import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, input_dim, out_dim, patch_dim, dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        assert input_dim % patch_dim == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (input_dim // patch_dim)
        self.patch_dim = patch_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.layer_out = nn.Sequential(
            nn.Linear(dim*num_patches, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim),
        )
        self.batchnorm = nn.BatchNorm1d(out_dim)
        self.R = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.R.data.fill_(1)

    def forward(self, data, mask=None):
        pd = self.patch_dim
        x = rearrange(data, 'b (n p) -> b n p', p=pd)
        x = self.patch_to_embedding(x)
        x += self.pos_embedding
        x = self.transformer(x)
        x = rearrange(x,'b p d ->b (p d)')
        z = self.layer_out(x)
        # z = self.batchnorm(x)
        feature_norm = torch.norm(z, dim=1, keepdim=True) + 1e-6
        return z
class ViT_Swin(nn.Module):
    def __init__(self, *, input_dim, out_dim, patch_dim, dim, depth, heads, mlp_dim, dim_head, roll_shifts):
        super().__init__()
        assert input_dim % patch_dim == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (input_dim // patch_dim)
        self.patch_dim = patch_dim
        self.roll_shifts= roll_shifts
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.embedding_to_patch = nn.Linear(dim, patch_dim)
        self.transformer1 = Transformer(dim, 1, heads, dim_head, mlp_dim)
        self.transformer2 = Transformer(dim, 1, heads, dim_head, mlp_dim)
        self.transformer3 = Transformer(dim, 1, heads, dim_head, mlp_dim)
        self.layer_out = nn.Sequential(
            nn.Linear(dim*num_patches, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim),
        )
        self.batchnorm = nn.BatchNorm1d(out_dim)
        self.R = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.R.data.fill_(1)

    def forward(self, data, mask=None):
        pd = self.patch_dim
        x = rearrange(data, 'b (n p) -> b n p', p=pd)
        x = self.patch_to_embedding(x)
        x += self.pos_embedding
        x = self.transformer1(x)
        x = self.embedding_to_patch(x)
        x = rearrange(x, 'b n p -> b (n p)')
        x = torch.roll(x, shifts=self.roll_shifts, dims=1)
        x = rearrange(x, 'b (n p) -> b n p', p=pd)
        x = self.patch_to_embedding(x)
        x = self.transformer2(x)
        x = self.embedding_to_patch(x)
        x = rearrange(x, 'b n p -> b (n p)')
        x = torch.roll(x, shifts=self.roll_shifts, dims=1)
        x = rearrange(x, 'b (n p) -> b n p', p=pd)
        x = self.patch_to_embedding(x)
        x = self.transformer3(x)
        x = rearrange(x,'b p d ->b (p d)')
        z = self.layer_out(x)
        # z = self.batchnorm(x)
        feature_norm = torch.norm(z, dim=1, keepdim=True) + 1e-6
        return z


class ForwardNet(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class ViT_backbone(nn.Module):
    def __init__(self, *, input_dim, out_dim, patch_dim, dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        assert input_dim % patch_dim == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (input_dim // patch_dim)
        self.patch_dim = patch_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.layer_out = nn.Sequential(
            nn.Linear(dim*num_patches, out_dim),
            nn.GELU(),
        )

    def forward(self, data, mask=None):
        pd = self.patch_dim
        x = rearrange(data, 'b (n p) -> b n p', p=pd)
        x = self.patch_to_embedding(x)
        x += self.pos_embedding
        x = self.transformer(x)
        x = rearrange(x,'b p d ->b (p d)')
        z = self.layer_out(x)
        return z

# graph attension network
class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_fix, patch_size, dropout = 0.0):
        super().__init__()
        self.transformer = ViT_backbone(input_dim=input_dim, out_dim=hidden_dim, patch_dim=patch_size, dim=16, depth=3, heads=8, mlp_dim=16, dim_head=16)
        self.transformer2 = ViT_backbone(input_dim=input_dim, out_dim=hidden_dim, patch_dim=patch_size, dim=16, depth=3, heads=8, mlp_dim=16, dim_head=16)

        self.MLP1 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)
        self.MLP2 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)
        self.MLP3 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)

        self.aw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.norm_att = nn.LayerNorm(hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(hidden_dim)

    def cal_attention(self, x, y, W):
        x=self.norm_att(x)
        y=self.norm_att(y)
        attention_out = torch.mm(torch.mm(x,W),torch.t(y))
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x_input):
        x = self.transformer(x_input)
        x2 = self.transformer2(x_input)
        xq = self.MLP1(x)
        xk = self.MLP2(x2)
        xv = self.MLP3(x2)
        att_weight = self.cal_attention(xq, xk,self.aw)
        hidden = self.norm_pre_ff(xv + att_weight.mm(xv))
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden)


class ProxyGAT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_fix, patch_size, dropout = 0.0):
        super().__init__()
        self.transformer1 = ViT_backbone(input_dim=input_dim, out_dim=hidden_dim, patch_dim=patch_size, dim=16, depth=3, heads=8, mlp_dim=32, dim_head=16)
        self.transformer2 = ViT_backbone(input_dim=input_dim, out_dim=hidden_dim, patch_dim=patch_size, dim=16, depth=3, heads=8, mlp_dim=32, dim_head=16)
        self.transformer3 = ViT_backbone(input_dim=input_dim, out_dim=hidden_dim, patch_dim=patch_size, dim=16, depth=3, heads=8, mlp_dim=32, dim_head=16)
        self.transformer4 = ViT_backbone(input_dim=input_dim, out_dim=hidden_dim, patch_dim=patch_size, dim=16, depth=3, heads=8, mlp_dim=32, dim_head=16)

        self.MLP1 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)
        self.MLP2 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)
        self.MLP3 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)

        self.MLP4 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)
        self.MLP5 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)
        self.MLP6 = ForwardNet(hidden_dim,hidden_dim, hidden_dim)

        self.x_fix = nn.Parameter(torch.randn(num_fix, input_dim))
        self.aw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.cw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.dw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fix = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fix2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fix3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.norm_att = nn.LayerNorm(hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(hidden_dim)
    def cal_attention(self, x, y, W):
        x=self.norm_att(x)
        y=self.norm_att(y)
        attention_out = torch.mm(torch.mm(x,W),torch.t(y))
        # attention_out = torch.mm(x, torch.t(y))
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x_input):
        #learning proxy from the data
        zq = self.MLP1(self.transformer1(x_input))
        zk = self.MLP2(self.transformer2(x_input))
        zv = self.MLP3(self.transformer2(x_input))
        z_fixq = self.MLP4(self.transformer3(self.x_fix))
        z_fixv = self.MLP5(self.transformer4(self.x_fix))
        att_weight = self.cal_attention(z_fixq, zk, self.aw)
        z_fix = self.norm_pre_ff(z_fixv + att_weight.mm(zv))
        z_fix = self.fc_fix(z_fix)
        z_fixk = self.leaky_relu(z_fix)
        att_weight = self.cal_attention(zq, z_fixk, self.bw)
        z = self.norm_pre_ff(zv + att_weight.mm(z_fixv))
        z = self.fc(z)
        z = self.leaky_relu(z)
        return self.fc_out(z)

