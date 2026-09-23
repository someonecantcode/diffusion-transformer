import os
import torch
import lpips
import wandb
from datasets import load_dataset
import torchvision.transforms.v2 as transforms
from vae import VAE
from ema import EMA

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(device)
    torch.backends.cudnn.benchmark = True
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".inductor_cache"
    # ------------------------------------------------------------
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
        images = [transform(img) for img in examples["jpg"] if img is not None]
        return {"pixel_values": images}

    dataset = dataset.with_transform(transform_batch)
    dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, pin_memory=False)
    data_iter = iter(dataloader)
    # ------------------------------------------------------------
    total_steps: int = 30000
    lr_cooldown_steps: int = 15000
    lr = 1e-4
    kl_beta = 1e-4
    fft_weight = 0.05
    lpips_weight = 0.05
    latent_channels = [128, 256, 512, 512]
    z_channels = 32
    model = VAE(
        n_channels=3,
        latent_channels=latent_channels,
        z_channels=z_channels,
        n_heads=8,
        resnet_blocks=2
    ).to(device).to(memory_format=torch.channels_last)
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    starting_step = 15000
    model.load_state_dict(torch.load(f"./saved_models/vae_compiled_{starting_step}.pt", weights_only=True))
    optimizer.load_state_dict(torch.load(f"./saved_models/vae_optim_compiled_{starting_step}.pt", weights_only=True))
    ema = EMA(model)
    ema.load(f"./saved_models/vae_compiled_{starting_step}_ema.pt")
    print(sum(p.numel() for p in model.parameters()), "total parameters")
    # ------------------------------------------------------------

    lpips_loss_function = lpips.LPIPS(net='alex').to(device)
    lpips_loss_function.requires_grad_(False)
    lpips_loss_function.eval()

    run = wandb.init(
        entity="someonecantcode-",
        project="vae",
        config={
            "architecture": "FLUX.V1",
            "dataset": "stanford-vision-lab/gpic",
            "steps": total_steps,
            "img_size": img_size,
            "latent_channels": latent_channels,
            "z_channels": z_channels,
            "n_heads": 8,
            "kl_beta": kl_beta
        },
    )
    
    print("Starting training", flush=True)
    wandb.watch(model, log_freq=100)
    for step in range(starting_step, total_steps + 1):
        try:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                data = next(data_iter)['pixel_values'].to(device)
                output, loss = model(data, lpips_loss_function, beta=kl_beta, fft_weight=fft_weight, lpips_weight=lpips_weight)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (step <= lr_cooldown_steps):
                lr = max((15000-step)/15e7 , 3e-7)# 5e-4 -> 3e-6 15k steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                    
            optimizer.step()
            ema.update()
            
            if step % 100 == 0:
                loss_item = loss.item()
                norm_item = norm.item()
                run.log({"loss": loss_item, "norm": norm_item, "lr": lr})
                print(f"step: {step:5d} | loss: {loss_item:8.4f} | norm: {norm_item:8.4f} | lr: {lr:.9f}", flush=True)
                
            if step % 5000 == 0:
                torch.save(model.state_dict(), f"./saved_models/vae_compiled_{step}.pt")
                torch.save(optimizer.state_dict(), f"./saved_models/vae_optim_compiled_{step}.pt")
                ema.checkpoint(f"./saved_models/vae_compiled_{step}_ema.pt")
        except StopIteration:
            data_iter = iter(dataloader)
            continue
        
    run.finish()
    torch.save(model.state_dict(), f"./saved_models/vae_compiled_{step}.pt")
    torch.save(optimizer.state_dict(), f"./saved_models/vae_optim_compiled_{step}.pt")
    ema.checkpoint(f"./saved_models/vae_compiled_{step}_ema.pt")
    # -------------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from einops import rearrange

    @torch.no_grad()
    def display(tensor):
        img = rearrange(tensor.to(torch.float32), "c h w -> h w c").cpu()
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(0, 0, 1, 1)
        plt.axis("off")
        plt.imshow(img)
        plt.savefig("reconstructed_vae.png", bbox_inches='tight', pad_inches=0, dpi=300)
    display(output[0])
    
if __name__ == '__main__':
    main()
