import torch
import lpips
from torchvision.transforms import ToTensor, transforms
from datasets import load_dataset
from vae import VAE
# from einops import rearrange

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(device)
    torch.backends.cudnn.benchmark = True
    
    # ------------------------------------------------------------
    img_size = 256
    batch_size = 8
    
    dataset = load_dataset("arrow", data_files={"data/filtered_ds/*.arrow"}, split="train")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RGB(),
            transforms.ToTensor(),
        ]
    )
    def transform_batch(examples):
        images = [transform(img) for img in examples["jpg"] if img is not None]
        return {"pixel_values": images}

    dataset = dataset.with_transform(transform_batch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=False)

    
    # ------------------------------------------------------------
    total_steps: int = 10000
    lr_cooldown_steps: int = 4000
    lr = 5e-3
    kl_beta = 0.5
    model = VAE(
        n_channels=3,
        latent_channels=[16, 32, 64, 64],
        z_channels=32,
        n_heads=8,
        resnet_blocks=2
    ).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # ------------------------------------------------------------
    
    lpips_loss_function = lpips.LPIPS(net='alex').to(device)
    lpips_loss_function.requires_grad_(False)
    lpips_loss_function.eval()

    data = torch.load("minidata/smallImageTensors.pt", weights_only=True).to(device) # pass through VAE
    for step in range(0, total_steps + 1):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(data, lpips_loss_function, beta=kl_beta)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (total_steps - step) <= lr_cooldown_steps:
            lr = (15000-step)/3e6 # 5e-3 -> 3e-6 15k steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(lr, 3e-6)
                
        optimizer.step()
        
        if step % 100 == 0:
            print(f"step: {step:8d} | loss: {loss.item():8.4f} | norm: {norm.item():8.4f}")
            
        if step % 1000 == 0:
            torch.save(model.state_dict(), "vae.pt")

if __name__ == '__main__':
    main()