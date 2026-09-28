import torch
import numpy as np
import wandb
import os

from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from T5CLIP import ConditioningEncoders
from model import mmDiT
from vae import VAE
from ema import EMA


ddp = int(os.environ.get('RANK', -1)) != -1 # using ddp torch run?
if ddp:
    assert torch.cuda.is_available(), "you need CUDA for DDP"
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # which gpu 0,1,2,..
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # multinode
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total gpus
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(device)

master_process = ddp_rank == 0
if master_process:
    print(ddp_world_size)

# ---------------HYPER-PARAMS-------------------------------------
lr = 1e-4
batch_size = 16
total_steps = 30000
lr_cooldown_steps: int = 15000

grad_accum_steps: int = 4
save_dir = "./saved_models/"

@dataclass
class mmDiTConfig:
    d_model: int = 256
    n_heads: int = 4
    n_blocks: int = 1
    patch_size: int = 16
    color_channels: int = 3 
    height: int = 32 # latent 
    width: int = 32 # latent 
    t_channels: int = 32
    p_cond: float = 0.2
    
@dataclass
class VAEConfig:
    kl_beta: float = 5e-2
    fft_weight: float = 0.05
    lpips_weight: float = 0.5
    latent_channels: tuple[int, ...] = [128, 256, 512, 512]
    z_channels: int = 32
    n_heads: int = 8
    resnet_blocks: int  = 2

def getDataLoader(img_size: int = 256, batch_size: int = 16):
    img_size = 256
    batch_size = 16

    dataset = load_dataset("arrow", data_files="../data/filtered/*.arrow")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RGB(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ]
    )
    def transform_batch(examples):
        images = []
        captions = []
        for img, json in zip(examples["jpg"], examples["json"]):
            caption = json['caption']
            if img is not None and caption is not None:
                images.append(transform(img))
                captions.append(caption)
        return {"pixel_values": images, "caption": captions}

    dataset = dataset.with_transform(transform_batch)['train']
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        sampler=DistributedSampler(dataset) if ddp else None,
        persistent_workers=True, # for epochs
        num_workers=4, # cpu cores
        prefetch_factor=2,
    )

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(device)
    torch.backends.cudnn.benchmark = True
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".inductor_cache"
    # ------------ MODELS --------------
    vae = VAE(
        n_channels=3,
        latent_channels=VAEConfig().latent_channels,
        z_channels=VAEConfig().z_channels,
        n_heads=VAEConfig().n_heads,
        resnet_blocks=VAEConfig().resnet_blocks
    ).to(device).to(memory_format=torch.channels_last)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    model = mmDiT(
        d_model=mmDiTConfig().d_model,
        n_heads=mmDiTConfig().n_heads,
        n_blocks=mmDiTConfig().n_blocks,
        patch_size=mmDiTConfig().patch_size,
        color_channels=mmDiTConfig().color_channels, 
        height=mmDiTConfig().height, 
        width=mmDiTConfig().width, 
        t_channels=mmDiTConfig().t_channels,
        pooled_dim=T5CLIP.pooled_dim,
        t5_dim=T5CLIP.t5_dim
    ).to(device)
    ema = EMA(model)
    T5CLIP = ConditioningEncoders(device=device)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = getDataLoader(batch_size = batch_size)
    dataloader_iter = iter(dataloader)
    # -----------------------------------
    
    if master_process:
        run = wandb.init(
            project="mmDiT",
            config={
                "architecture": "FLUX.V1",
                "dataset": "stanford-vision-lab/gpic",
                "steps": total_steps,
            },
        )
        
        print("Starting training", flush=True)
        wandb.watch(model, log_freq=100)

    model.train()
    for step in range(total_steps + 1):
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            images = batch['pixel_values'].to(device, non_blocking=True)
            captions = batch['caption']

            time_steps = torch.rand(batch_size, device=device)
            epsilon = torch.randn_like(images, device=device)
            # t * z + (1 - t) * epsilon
            interpolated = torch.einsum('b, b c h w -> b c h w', time_steps, images) + torch.einsum('b, b c h w -> b c h w', 1 - time_steps, epsilon)
            t5, pooled, attn_mask = T5CLIP.encode(captions=captions)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                velocity_pred = model(interpolated, t5, time_steps, pooled, attn_mask, mmDiTConfig().p_cond)
            loss = ((velocity_pred - (images - epsilon))**2).mean()

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if ddp: # wait and sync with all gpus at the end
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward() # changed due to ddp wrap
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (step <= lr_cooldown_steps):
            lr = max((15000-step)/15e7 , 3e-7)# 5e-4 -> 3e-6 15k steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        
        optimizer.step()
        
        if master_process:
            ema.update()
            
            if step % 100 == 0:
                loss_item = loss.item()
                norm_item = norm.item()
                run.log({"loss": loss_item, "norm": norm_item, "lr": lr})
                print(f"step: {step:5d} | loss: {loss_item:8.4f} | norm: {norm_item:8.4f} | lr: {lr:.9f}", flush=True)
                
            if step % 2000 == 0:
                torch.save(model.state_dict(), f"{save_dir}vae_compiled_{step}.pt")
                torch.save(optimizer.state_dict(), f"{save_dir}vae_optim_compiled_{step}.pt")
                ema.checkpoint(f"{save_dir}vae_compiled_{step}_ema.pt")
    
    if master_process:  
        run.finish()
        torch.save(model.state_dict(), f"{save_dir}vae_compiled_{step}.pt")
        torch.save(optimizer.state_dict(), f"{save_dir}vae_optim_compiled_{step}.pt")
        ema.checkpoint(f"{save_dir}vae_compiled_{step}_ema.pt")
