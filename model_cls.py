import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms as T
import numpy as np
from torch import nn, einsum
import torchvision.transforms.functional as TF
import torch.nn.utils.weight_norm as weight_norm

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helpers
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class WindowPartition(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = pair(window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.window_size #16 16
        y = x.unfold(2, kH, kH).unfold(3, kW, kW) # B, C, nH, nW, kH, kW
        windows = y.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, kH*kW*C)
        return windows
    
    
class RearrangeAndPermute(nn.Module):
    def __init__(self, window_size, resolution):
        super(RearrangeAndPermute, self).__init__()
        self.window_size = window_size
        self.resolution = resolution
    
    def forward(self, windows):
        H, W = self.resolution  # 32, 32
        kH, kW = self.window_size  # 4, 4
        B = windows.shape[0]  # Batch size: 1
        C = windows.shape[-1] // (kH * kW)
        x = windows.view(B, H // kH, W // kW, kH, kW, C)
        y = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return y
    
    
def to_patch_embedding(x, patch_height, patch_width, patch_dim, dim):
    x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width) # [1,64,48]
    linear = nn.Linear(patch_dim, dim, bias=False)
    x = linear(x)  # [1,64,768]
    
    return x

def window_reverse(windows, window_size, resolution):
    H, W = pair(resolution)  # 32, 32
    kH, kW = pair(window_size)  # 4, 4
    num_patches = (H // kH) * (W // kW)  # 64
    B = windows.shape[0]  # Batch size: 1
    total_elements_per_patch = windows.shape[-1]  # 48

    C = total_elements_per_patch // (kH * kW)  # 48 / (4*4) = 3

    windows = windows.reshape(B, num_patches, kH, kW, C)
    
    x = windows.view(B, H // kH, W // kW, kH, kW, C)  
    y = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W) 
    
    return y

def reverse_patch1(x, patch_height, patch_width, dim):
    reverse_linear = nn.Linear(dim, patch_height * patch_width * 3, bias=False)
    # reverse_linear.weight.data = linear.weight.data.T  # 가중치의 전치 행렬 사용
    
    x = reverse_linear(x) # [B, N, patch_height * patch_width * 3]   [1,64,48]
    # x = torch.clamp(x, 0, 1)
    # x = torch.sigmoid(x)
    return x


def reverse_patch(x, patch_height, patch_width, dim):
    reverse_linear = nn.Linear(dim, patch_height * patch_width * 3, bias=False)  
    x = reverse_linear(x)  # [B, N, patch_height * patch_width * 3]   [1,64,48]
    # x = torch.clamp(x, 0, 1)
    # x = torch.sigmoid(x)
    return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        dim_head = dim//heads
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        
        #q, k, v 따로 입력받음.
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, q,k,v):
        h = self.heads
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q,k,v))

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') 

        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        dim_head = dim//heads
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') 
        return self.to_out(out)#, self_score
    
    

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim , dropout=dropout)


    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return self.norm(x)#, score


class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)


    def forward(self, q, k, v):
        x = self.attn(q, k, v) + q
        x = self.ff(x) + x
        return self.norm(x)
    
 
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, hidden_dim, dropout):
        super().__init__()

        self.self_attn1 = Transformer(dim, heads, dim_head, hidden_dim, dropout)
        self.self_attn2 = Transformer(dim, heads, dim_head, hidden_dim, dropout)
        self.self_attn3 = Transformer(dim, heads, dim_head, hidden_dim, dropout)
        self.cross_attn_Q = CrossTransformer(dim, heads, dim_head, hidden_dim, dropout)
        self.cross_attn_K = CrossTransformer(dim, heads, dim_head, hidden_dim, dropout)
        self.self_attn_V = Transformer(dim, heads, dim_head, hidden_dim, dropout)

    def forward(self, q, k, v):
        q_self = self.self_attn1(q)
        k_self = self.self_attn2(k)
        v_self = self.self_attn3(v)
        q_result = self.cross_attn_Q(q_self, k_self, v_self)
        k_result = self.cross_attn_K(k_self, k_self, v_self) #이건 문제 없음음
        v_result = self.self_attn_V(v_self)
        return q_result, k_result, v_result


class SRResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(SRResidualBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out

class TokenizationBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(TokenizationBlock, self).__init__()
        num_residual_blocks = int(np.log2(patch_size))
        dim1 = embed_dim // 2**num_residual_blocks

        self.init_conv = nn.Conv2d(in_channels, dim1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        res_blocks = [SRResidualBlock(dim1 * 2**(i+1)) for i in range(num_residual_blocks)]
        self.res_blocks = nn.ModuleList(res_blocks)

        pool_layers = [nn.Conv2d(dim1*2**(i+1), dim1*2**(i+2), kernel_size=3, stride=2, padding=1) 
                       for i in range(num_residual_blocks-1)]
        pool_layers = [nn.Conv2d(dim1, dim1 * 2, kernel_size=3, stride=2, padding=1)] + pool_layers
            
        self.pool = nn.ModuleList(pool_layers)

    def forward(self, x):
        f = []
        x = self.init_conv(x)
        x = self.relu(x)
        f.append(x)
        for pooling, res in zip(self.pool, self.res_blocks):
            x = pooling(x)
            x = res(x)
            f.append(x)
        f.pop()
        x = rearrange(x, 'b c h w -> b (h w) c') 
        return x, f

class DetokenizationBlock(nn.Module):
    def __init__(self, out_channels, embed_dim, patch_size):
        super(DetokenizationBlock, self).__init__()
        
        num_residual_blocks = int(np.log2(patch_size))
        dim1 = embed_dim // 2**num_residual_blocks

        # Initial Projection (Token -> Feature Map)
        self.init_proj = nn.Conv2d(embed_dim, dim1 * 2**num_residual_blocks, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Upsampling + Residual Blocks
        upsample_layers = [
            nn.ConvTranspose2d(dim1 * 2**(i+2), dim1 * 2**(i+1), kernel_size=3, stride=2, padding=1, output_padding=1)
            for i in reversed(range(num_residual_blocks - 1))
        ]
        upsample_layers.append(
            nn.ConvTranspose2d(dim1*2, dim1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.upsample = nn.ModuleList(upsample_layers)
        self.res_blocks = nn.ModuleList([SRResidualBlock(dim1 * 2**(i+1)) for i in reversed(range(num_residual_blocks - 1))])
        self.res_blocks.append(
            SRResidualBlock(dim1)
        )

        self.final_conv = nn.Conv2d(dim1, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, f):
        x = self.init_proj(x)
        x = self.relu(x)

        for upsample, res, skip in zip(self.upsample, self.res_blocks, reversed(f)):
            x = upsample(x)
            x = x + skip
            x = res(x)

        x = self.final_conv(x)
        return x


class Net(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.token_h, self.token_w = image_height // patch_height, image_width // patch_width
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = nn.ModuleList([
            TokenizationBlock(channels, dim, patch_width),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)]
        )

        self.to_patch_embedding_hq = nn.ModuleList([
            TokenizationBlock(channels, dim, patch_width),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)]
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.to_latent = nn.Identity()

        self.blocks = nn.ModuleList([
            AttentionBlock(dim, heads, dim_head, mlp_dim, dropout) for i in range(depth)
        ])
        
        self.reverse_patch1 = DetokenizationBlock(channels, dim, patch_width)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


    def forward(self, q, k, v):
        # q, k, v는 이제 [batch_size, 3, 32, 32] 형식으로 입력됨
        # q,k,v = [1,3,32,32]
        res = q.clone()
        for i, (token, token_hq) in enumerate(zip(self.to_patch_embedding, self.to_patch_embedding_hq)):
            if i == 0:
                q, f = token(q)  
                k, _ = token(k) 
                v, _ = token_hq(v)  
            else:
                q = token(q)
                k = token(k)
                v = token_hq(v)
        b, n, _ = q.shape

        # q,k,v = [1,64,192] 배치 사이즈, 
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        q = torch.cat((cls_tokens, q), dim=1)
        k = torch.cat((cls_tokens, k), dim=1)
        v = torch.cat((cls_tokens, v), dim=1)
        
        
        q += self.pos_embedding[:, :(n + 1)]  # [batch_size, num_patches, dim]
        k += self.pos_embedding[:, :(n + 1)] 
        v += self.pos_embedding[:, :(n + 1)] 
        

        # 각 block에서 나온 Attention weight를 저장
        for block in self.blocks:
            q, k, v= block(q, k, v)
            
        q_block = q[:, 1:, :]
        q_block = rearrange(q_block, 'b (h w) c -> b c h w', h=self.token_h, w=self.token_w)
        out = self.reverse_patch1(q_block, f) 

        return out + res
    

# target_size = 32
# patch_dict = {32: 4, 64: 8,  128: 8, 256: 16, 512: 16, 1024: 32}
# dim_dict = {32: 64, 64: 128,  128: 128, 256: 256, 512: 256, 1024: 512}
# model = Net(image_size=(32, 32), patch_size=4, dim=dim_dict[target_size], 
#             depth=6, heads=4, mlp_dim=dim_dict[target_size]//4 * 2, dropout=0.0)
# model(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))