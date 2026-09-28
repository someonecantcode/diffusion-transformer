import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class Patchify2D(nn.Module):
    def __init__(self, d_model: int, patch_size: int, color_channels: int):
        """
        input: (B, C, H, W)
        output: (B, N, D)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=color_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LINEAR CxHxW -> (H/p * W/p)xCPP -> (NxCPP) @ (CPP x d) -> NxD
        # CONV   CxHxW -> DxH/pxW/p       -> DxN -> NxD
        return rearrange(self.conv(x), "... d h w -> ...  (h w) d")
        
class Depatchify2D(nn.Module):
    def __init__(self, d_model: int, patch_size: int, color_channels: int, height: int, width: int):
        """
        input: (B, N, D)
        output: (B, C, H, W)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        
        self.deconv = nn.ConvTranspose2d(in_channels=d_model, out_channels=color_channels, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.height // self.patch_size
        w1 = self.width // self.patch_size
        # x # B, N, D -> B, H/p, W/p, D -> B, D, H/p, W/p
        # N = h // p * w // p
        x = rearrange(x, "... (h1 w1) d ->... d h1 w1", h1 = h1, w1 = w1)
        return self.deconv(x)
    
class FourierTimeEmbedding(nn.Module):
    def __init__(self, t_channels: int, w_min: float = 1.0, w_max: float = 10000.0):
        """
        input: (B)
        output: (B, T)
        """
        super().__init__()
        self.t_channels = t_channels
        half_dim = t_channels // 2
        
        w = w_min * ((w_max / w_min) ** ((torch.linspace(start=1, end=half_dim, steps=half_dim) - 1) / (half_dim - 1)))
        self.register_buffer("w", w) # non trainable

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = 2 * math.pi * torch.einsum('t, w -> tw', t, self.w)
        emb = math.sqrt(2.0 / self.t_channels) * torch.cat([torch.cos(args), torch.sin(args)], dim=1) # Batch x t_channels
        
        if self.t_channels % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    
class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),      
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class AdaLN(nn.Module):
    def __init__(self, normalized_shape: int):
        """
        x: (B, N, D)
        scale/shift: (B, 1, D)
        """
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape, elementwise_affine=False)
    
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        return x * (1 + scale) + shift

class DoubleStreamAttention(nn.Module): # TODO: RoPE 
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        
        self.n_heads = n_heads
        self.qkv_image_weights = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qkv_text_weights = nn.Linear(d_model, 3 * d_model, bias=False)
        
        self.proj_image = nn.Linear(d_model, d_model)
        self.proj_text = nn.Linear(d_model, d_model)
    
    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),))
    
    def forward(self, data: torch.Tensor, cond: torch.Tensor, attn_mask=None) -> torch.Tensor:
        q_image, k_image, v_image = rearrange(self.qkv_image_weights(data), "b t (k nh hs) -> k b nh t hs", k=3, nh=self.n_heads).unbind(0)
        q_text, k_text, v_text = rearrange(self.qkv_text_weights(cond), "b t (k nh hs) -> k b nh t hs", k=3, nh=self.n_heads).unbind(0)
        
        # b, nh, t, hs dim = 2
        dim = 2
        Q = torch.cat([self.norm(q_image), self.norm(q_text)], dim = dim)
        K = torch.cat([self.norm(k_image), self.norm(k_text)], dim = dim)
        V = torch.cat([v_image, v_text], dim = dim)
        
        # Q, K = apply_rope(Q, K, pe)
        attn_outs = rearrange(F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask), 'b nh t hs -> b t (nh hs)')
        
        N = data.size(1)
        attn_image, attn_text = attn_outs[:, :N], attn_outs[:, N:]
        
        return self.proj_image(attn_image), self.proj_text(attn_text)

class mmDiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        data (B, N, D)
        cond (B, T, D)
        time_steps (B, D)
        """
        super().__init__()
        
        self.adaln_mlp_text = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        self.adaln_mlp_image = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        for mlp in (self.adaln_mlp_text, self.adaln_mlp_image):
            nn.init.zeros_(mlp[1].weight)
            nn.init.zeros_(mlp[1].bias)
        
        self.adaln_text_1 = AdaLN(d_model)
        self.adaln_image_1 = AdaLN(d_model)
        
        self.double_stream_attn = DoubleStreamAttention(d_model, n_heads)
        
        self.adaln_text_2 = AdaLN(d_model)
        self.mlp_text = MLP(d_model, d_model)
        self.adaln_image_2 = AdaLN(d_model)
        self.mlp_image = MLP(d_model, d_model)
        
        
    def forward(self, data: torch.Tensor, cond: torch.Tensor, time_steps: torch.Tensor, attn_mask=None):
        cond_text_scalars = rearrange(self.adaln_mlp_text(time_steps), "b (n d) -> b n d", n=6)
        cond_image_scalars = rearrange(self.adaln_mlp_image(time_steps), "b (n d) -> b n d", n=6)
        
        alpha_text, beta_text, gamma_text, delta_text, epsilon_text, zeta_text = cond_text_scalars.chunk(6, dim=1)
        alpha_image, beta_image, gamma_image, delta_image, epsilon_image, zeta_image = cond_image_scalars.chunk(6, dim=1)
    
        # (QKV) [img, text]
        attn_image, attn_text = self.double_stream_attn(
            self.adaln_image_1(data, alpha_image, beta_image),
            self.adaln_text_1(cond, alpha_text, beta_text),
            attn_mask
        )
        
        cond = gamma_text * attn_text + cond
        data = gamma_image * attn_image + data
        
        cond = zeta_text * self.mlp_text(self.adaln_text_2(cond, delta_text, epsilon_text)) + cond
        data = zeta_image * self.mlp_image(self.adaln_image_2(data, delta_image, epsilon_image)) + data
        
        return data, cond
    
class SingleStreamBlock(nn.Module):
    """
    input: img + t5 concatted (B, N_1 + N_2, D)
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        self.adaln1 = AdaLN(d_model)
        self.adaln2 = AdaLN(d_model)
        
        self.qkv_weights = nn.Linear(d_model, 3 * d_model)
        self.mlp = MLP(d_model, d_model)
        
        nn.init.zeros_(self.adaln_mlp[1].weight)
        nn.init.zeros_(self.adaln_mlp[1].bias)
        
    def norm(self, x: torch.Tensor) -> torch.Tensor:
            return F.rms_norm(x, (x.size(-1),))
        
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, attn_mask=None):
        scalars = rearrange(self.adaln_mlp(time_steps), "b (n d) -> b n d", n=6)
        alpha, beta, gamma, delta, epsilon, zeta = scalars.chunk(6, dim=1)
        
        q, k, v = rearrange(self.qkv_weights(self.adaln1(x, alpha, beta)), "b t (k nh hs) -> k b nh t hs", k=3, nh=self.n_heads).unbind(0)
        q, k = self.norm(q), self.norm(k)
        # Q, K = apply_rope(Q, K, pe)
        # be wary of attn_mask and order of the
        attn = rearrange(F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask), 'b nh t hs -> b t (nh hs)')
        
        x = x + attn * gamma
        x = x + self.mlp(self.adaln2(x, epsilon, delta)) * zeta
        return x
    
class mmDiT(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_blocks: int, patch_size: int, color_channels: int, height: int, width: int, t_channels: int, pooled_dim: int, t5_dim: int):
        """
        inputs:
        data (B, C, H, W)
        t5_cond (B, T, 512)
        
        time_steps (B)
        pooled_cond (B, 1280)
        """
        super().__init__()
        self.color_channels = color_channels
        self.height = height
        self.width = width
        self.patchify = Patchify2D(d_model, patch_size, color_channels)
        self.depatchify = Depatchify2D(d_model, patch_size, color_channels, height, width)
        
        self.time_embed = nn.Sequential(
            FourierTimeEmbedding(t_channels),
            MLP(t_channels, d_model)
        )
        
        self.pooled_mlp = MLP(pooled_dim, d_model)
        self.t5_proj = nn.Linear(t5_dim, d_model)
        self.null_cond_vector = nn.Parameter(torch.zeros(1, pooled_dim))
        
        # self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.mmDiTBlocks = nn.ModuleList([mmDiTBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.singlestreamBlocks = nn.ModuleList([SingleStreamBlock(d_model, n_heads) for _ in range(n_blocks)])
        
        self.mlp_adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.final_adaln = AdaLN(d_model)
        
    def forward(self, data: torch.Tensor, t5_cond: torch.Tensor, time_steps: torch.Tensor, pooled_cond: torch.Tensor, attn_mask, p_drop: float = 0.2) -> torch.Tensor:
        data = self.patchify(data)
        B, N, _ = data.shape
        device = data.device
        
        # cfg, concat order [img, text]
        cond_mask = (torch.randn(B, 1, device=device) < p_drop)
        attn_mask = ((
            torch.cat(
                (torch.ones(B, N, device=device).bool(), attn_mask * cond_mask), dim=-1
            ).view(B, 1, 1, -1)).bool()
            if attn_mask is not None
            else None
        )
        pooled_cond = torch.where(cond_mask, self.null_cond_vector, pooled_cond)
        t5_cond = self.t5_proj(t5_cond)
        time_steps = self.time_embed(time_steps) + self.pooled_mlp(pooled_cond)
        
        # positional embedding?
        for block in self.mmDiTBlocks:
            data, t5_cond = block(data, t5_cond, time_steps, attn_mask)
            
        x = torch.cat([data, t5_cond], dim=1)
        for block in self.singlestreamBlocks:
            x = block(x, time_steps, attn_mask)
        
        # trash the conditioning part
        data = x[:, :N]
        
        final_scale, final_shift = rearrange(self.mlp_adaln(time_steps), "b (n d) -> b n d", n=2).chunk(2, dim=1)
        data = self.final_adaln(data, final_scale, final_shift)
        return self.depatchify(data)
    
    def alpha(self, t: torch.Tensor, gamma: float, lambda_max: float) -> torch.Tensor:
        return lambda_max * t ** gamma
    
    @torch.no_grad()
    def sample(self, total_steps: int, t5_cond: torch.Tensor, time_steps: torch.Tensor, pooled_cond: torch.Tensor, attn_mask):
        """
        Rectified CFG++
        https://arxiv.org/abs/2510.07631
        """
        outputs = []
        device = t5_cond.device
        batch_size = time_steps.shape[0]
        X_0 = torch.randn((batch_size, self.color_channels, self.height, self.width), device=device)
        
        for _, time_steps in enumerate(torch.linspace(0, 1, total_steps, device=device), start=1):
            outputs.append(X_0)
            time_steps = time_steps.expand(batch_size)
            half_step_ts = time_steps + (1 / (2 * total_steps)) 
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # X_0 = X_0 + (1 / total_steps) * self(X_0, t5_cond, time_steps, pooled_cond, attn_mask)
                v_cond = self(X_0, t5_cond, time_steps, pooled_cond, attn_mask, p_cond=0.0)
                half_step_x = X_0 + (1 / total_steps) * v_cond / 2

                v_cond_half_step = self(half_step_x, t5_cond, half_step_ts, pooled_cond, attn_mask, p_cond=0.0)
                v_uncond_half_step = self(half_step_x, t5_cond, half_step_ts, pooled_cond, attn_mask, p_cond=1.0)

                v_corrected = v_cond + self.alpha(time_steps, gamma=0, lambda_max=0.8) * (v_cond_half_step - v_uncond_half_step)
                X_0 = X_0 + (1 / total_steps) * v_corrected
        return outputs # X_0, ..., X_1