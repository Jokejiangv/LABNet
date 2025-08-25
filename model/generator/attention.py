import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        n_heads=4,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm([emb_dim,])
        self.norm_kv = nn.LayerNorm([emb_dim,])
        self.Wq = nn.Linear(emb_dim, hidden_dim)
        self.Wk = nn.Linear(emb_dim, hidden_dim)
        self.Wv = nn.Linear(emb_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, emb_dim)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        # x: [B, C, D, T, F]
        B, C, D, T, F = x.size()

        x = x.permute(0, 3, 4, 1, 2) # [B,T,F,C,D]
        
        x_ref = x[..., :1, :] # [B,T,F,1,D]

        q = self.Wq(self.norm_q(x_ref)).reshape(B, T, F, 1, self.n_heads, self.head_dim).transpose(-2, -3)  # (b,t,f,h,1,d/h)
        x = self.norm_kv(x)
        k = self.Wk(x).reshape(B, T, F, C, self.n_heads, self.head_dim).transpose(-2, -3)  # (b,t,f,h,c,d/h)
        v = self.Wv(x).reshape(B, T, F, C, self.n_heads, self.head_dim).transpose(-2, -3)  # (b,t,f,h,c,d/h)

        attn = torch.matmul(q, k.transpose(-2, -1)) # (b,t,f,h,1,c)
        attn = torch.mul(attn, self.scale)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (b,t,f,h,1,d/h)

        out = out.transpose(-2, -3).flatten(-2, -1) # (b,t,f,1,d)
        out = self.Wo(out)  # (b,t,f,1,d)
        
        return (out + x_ref).permute(0, 3, 4, 1, 2)  # [B, 1, D, T, F]
        

if __name__=='__main__':
    x = torch.randn(4, 1, 16, 63, 32)
    block = AttentionBlock(16, 32, 4)
    y = block(x)
    print(y.shape)
