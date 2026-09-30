import torch
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

# torchrun --standalone --nproc_per_node=2 train.py
# ---------------HYPER-PARAMS-------------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 32
    total_steps: int = 30000
    lr_cooldown_steps: int = 20000
    starting_step: int = 0
    logging_interval: int = 100
    grad_accum_steps: int = 2 # not really needed
    save_dir: str = "./saved_models/"

@dataclass
class VAEConfig:
    kl_beta: float = 5e-2
    fft_weight: float = 0.05
    lpips_weight: float = 0.5
    latent_channels: tuple[int, ...] = (128, 256, 512, 512)
    z_channels: int = 32
    n_heads: int = 8
    resnet_blocks: int  = 2

@dataclass
class mmDiTConfig:
    d_model: int = 512
    n_heads: int = 4
    n_blocks: int = 16
    patch_size: int = 2
    color_channels: int = 32 # z_channels
    height: int = 32 # latent
    width: int = 32 # latent 
    t_channels: int = 64
    p_cond: float = 0.2
    
def getDataLoader(ddp: bool, img_size: int = 256, batch_size: int = 16):
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
    sampler = DistributedSampler(dataset) if ddp else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None),
        persistent_workers=True, # for epochs
        num_workers=1, # cpu cores
        prefetch_factor=2,
    )

def main():
    ddp = int(os.environ.get('RANK', -1)) != -1 # torch run
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

    torch.backends.cudnn.benchmark = True
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".inductor_cache"
    # ------------ MODELS --------------
    vae_cfg = VAEConfig()
    dit_cfg = mmDiTConfig()
    train_cfg = TrainConfig()
    
    T5CLIP = ConditioningEncoders(device=device, debugging=False)
    
    model = mmDiT(
        d_model=dit_cfg.d_model,
        n_heads=dit_cfg.n_heads,
        n_blocks=dit_cfg.n_blocks,
        patch_size=dit_cfg.patch_size,
        color_channels=dit_cfg.color_channels, 
        height=dit_cfg.height, 
        width=dit_cfg.width, 
        t_channels=dit_cfg.t_channels,
        pooled_dim=T5CLIP.pooled_dim,
        t5_dim=T5CLIP.t5_dim
    ).to(device)
    vae = VAE(
        n_channels=3,
        latent_channels=vae_cfg.latent_channels,
        z_channels=vae_cfg.z_channels,
        n_heads=vae_cfg.n_heads,
        resnet_blocks=vae_cfg.resnet_blocks
    ).to(device).to(memory_format=torch.channels_last)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae.load_state_dict(torch.load(train_cfg.save_dir + "vae_compiled_10000.pt", weights_only=True))

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) 
    raw_model = model.module if ddp else model
    ema = EMA(raw_model)
    
    lr = train_cfg.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    dataloader = getDataLoader(ddp=ddp, batch_size = train_cfg.batch_size)
    dataloader_iter = iter(dataloader)
    # -----------------------------------
    
    if master_process:
        run = wandb.init(
            project="mmDiT-FlowMatching",
            config={
                "architecture": "FLUX.V1",
                "dataset": "stanford-vision-lab/gpic",
                "steps": train_cfg.total_steps,
            },
        )
        
        print("Starting training", flush=True)
        wandb.watch(model, log_freq=100)

    epoch = 0
    model.train()
    for step in range(train_cfg.starting_step, train_cfg.total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        
        loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(train_cfg.grad_accum_steps):
            if ddp: # wait and sync with all gpus at the end
                model.require_backward_grad_sync = (micro_step == train_cfg.grad_accum_steps - 1)
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                epoch += 1
                if ddp and hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            images = batch['pixel_values'].to(device, non_blocking=True)
            captions = batch['caption']
            
            with torch.no_grad():
                t5, pooled, attn_mask = T5CLIP.encode(captions=captions)
                images, _, _ = vae.encode(images)
            
            time_steps = torch.rand(images.shape[0], device=device)
            epsilon = torch.randn_like(images, device=device)
            # t * z + (1 - t) * epsilon
            interpolated = torch.einsum('b, b c h w -> b c h w', time_steps, images) + torch.einsum('b, b c h w -> b c h w', 1 - time_steps, epsilon)
            
            device_type = "cuda" if "cuda" in str(device) else str(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                velocity_pred = model(interpolated, t5, time_steps, pooled, attn_mask, dit_cfg.p_cond)
            
            loss = ((velocity_pred - (images - epsilon))**2).mean()
            loss = loss / train_cfg.grad_accum_steps
            loss_accum += loss.detach()

            loss.backward() # changed due to ddp wrap
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (step >= train_cfg.lr_cooldown_steps):
            lr = max((15000-step)/15e7 , 3e-7)# 5e-4 -> 3e-6 15k steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        optimizer.step()
        
        if master_process:
            ema.update()
            if step % train_cfg.logging_interval == 0:
                loss_item = loss_accum.item()
                norm_item = norm.item()
                run.log({"loss": loss_item, "norm": norm_item, "lr": lr})
                print(f"step: {step:5d} | loss: {loss_item:8.4f} | norm: {norm_item:8.4f} | lr: {lr:.9f}", flush=True)
                
            if step % 2000 == 0:
                torch.save(raw_model.state_dict(), f"{train_cfg.save_dir}mmDiT_{step}.pt")
                torch.save(optimizer.state_dict(), f"{train_cfg.save_dir}optim_{step}.pt")
                ema.checkpoint(f"{train_cfg.save_dir}mmDiT_{step}_ema.pt")
    
    if master_process:  
        print("Completed Training")
        run.finish()
        torch.save(raw_model.state_dict(), f"{train_cfg.save_dir}mmDiT_{step}.pt")
        torch.save(optimizer.state_dict(), f"{train_cfg.save_dir}optim_{step}.pt")
        ema.checkpoint(f"{train_cfg.save_dir}mmDiT_{step}_ema.pt")
        
        raw_model.eval()
        from torchvision.utils import save_image
        with torch.no_grad():
            t5, pooled, attn_mask = T5CLIP.encode(captions=["A cat resting in a grassy field."])
            total_timesteps = 50
            output = raw_model.sample(total_timesteps, t5, pooled, attn_mask)
            
            print(output[total_timesteps - 1].unsqueeze(0).shape)
            final_image = vae.decode(output[total_timesteps - 1].unsqueeze(0))
            save_image(final_image, "sampled_image.png")
        print("Finished Sampling")
            
    if ddp:
        # dist.barrier() 
        dist.destroy_process_group()
        
if __name__ == '__main__':
    main()
