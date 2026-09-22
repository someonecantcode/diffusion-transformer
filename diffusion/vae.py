import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResNet(nn.Module):
    def __init__(self, dim: int, n_groups: int = 4):
        super().__init__()
        self.layers = nn.Sequential(                
            nn.GroupNorm(num_groups=n_groups, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=n_groups, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resnet_blocks: int = 4):
        super().__init__()
        self.resnets = nn.Sequential(*[ResNet(dim=in_channels) for _ in range(resnet_blocks)])
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets(x)
        return self.down_conv(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resnet_blocks: int = 4):
        super().__init__()
        self.resnets = nn.Sequential(*[ResNet(dim=in_channels) for _ in range(resnet_blocks)])
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets(x)
        return self.up_conv(x)
    
class Attention(nn.Module):
    def __init__(self, latent_channels: int, n_heads: int):
        """
        input: (B, C, Z, Z)
        """
        super().__init__()
        self.n_heads = n_heads
        self.w_qkv = nn.Conv2d(latent_channels, 3 * latent_channels, kernel_size=1)
        self.wo = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # c = nh hs, attn over every latent pixel
        assert (x.shape[1] % self.n_heads == 0), f"Latent Channels {x.shape[1]} must be divisible by n_heads {self.n_heads}"
        q, k, v = rearrange(self.w_qkv(x), "b (k nh hs) h w -> k b nh (h w) hs", k=3, nh=self.n_heads).unbind(0)
        attn = rearrange(F.scaled_dot_product_attention(q, k, v), "b nh (h w) hs -> b (nh hs) h w", w=x.shape[-1])
        return x + self.wo(attn)
    
class Encoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], z_channels: int, n_heads: int = 1, resnet_blocks: int = 2, bottleneck_layers: int = 4):
        """
        input: (B, C, H, W)
        output: (B, z_channels, H/f, W/f)
        """
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            *[DownBlock(channels[i], channels[i+1], resnet_blocks) for i in range(len(channels) - 1)]
        )
    
        self.bottleneck = nn.Sequential(
            *[
                nn.Sequential(Attention(channels[-1], n_heads), ResNet(channels[-1]))
                for _ in range(bottleneck_layers)
            ], 
            nn.Conv2d(channels[-1], z_channels, kernel_size=3, padding=1)
        )
        
        self.mu_proj = nn.Conv2d(z_channels, z_channels, kernel_size=3, padding=1)
        self.logvar_proj = nn.Conv2d(z_channels, z_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.downsample(x)
        x = self.bottleneck(x)
        return self.mu_proj(x), self.logvar_proj(x)
    
class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], z_channels: int, n_channels: int, resnet_blocks: int = 4):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(z_channels, channels[-1], kernel_size=3, padding=1),
            *[UpBlock(channels[i], channels[i-1], resnet_blocks) for i in range((len(channels) - 1), 0, -1)],
            nn.Conv2d(channels[0], n_channels, kernel_size=3, padding=1) # RGB is 3 output channels
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)

class VAE(nn.Module):
    def __init__(self, n_channels: int, latent_channels: tuple[int, ...], z_channels: int, n_heads: int, resnet_blocks: int = 4, bottleneck_layers: int = 4):
        """
        input: (B, C, H, W)
        latent: (B, z_channels, H/f, W/f), f = 2^len(latent_channels)
        """
        super().__init__()
        self.encoder = Encoder(latent_channels, z_channels, n_heads, resnet_blocks, bottleneck_layers)
        self.decoder = Decoder(latent_channels, z_channels, n_channels)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mu, logvar = self.encoder(x)
        reparam = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        return reparam, mu, logvar
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor, lpips_loss_function, beta: float = 1.0, fft_weight: float = 1.0, lpips_weight: float = 1.0) -> tuple[torch.Tensor, ...]: #  lpips_loss_function: lpips.LPIPS,
        latent, mu, logvar = self.encode(x)
        output = self.decode(latent)
        
        if self.training is False:
            loss = None
        else:
            recon_loss = F.l1_loss(input=output, target=x)
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            fft_loss = F.l1_loss(input=torch.log1p(torch.fft.rfft2(output, dim=(-2, -1)).abs()), target=torch.log1p(torch.fft.rfft2(x, dim=(-2, -1)).abs()))
            loss = recon_loss + (beta) * kl_loss + (fft_weight) * fft_loss
            if lpips_loss_function is not None: # lpips is kinda slow
                loss += (lpips_weight) * lpips_loss_function(x, output).mean()  
        return output.detach(), loss