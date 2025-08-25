from .dpr_layer import DPR, CustomLayerNorm
import torch
import torch.nn as nn
from .attention import AttentionBlock



class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3)),
        )
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)
    
    def forward(self, x):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]
        
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)

        x = torch.cat([x_low, x_high], dim=-1)
        x = self.norm(x)
        x = self.act(x)
        return x


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)
    
    def forward(self, x):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels//4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        
        self.conv_2 = DSConv(num_channels//4, num_channels//2, n_freqs=257)
        self.conv_3 = DSConv(num_channels//2, num_channels//4*3, n_freqs=128)
        self.conv_4 = DSConv(num_channels//4*3, num_channels, n_freqs=64)


    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x)
        out_list.append(x)  # 128
        x = self.conv_3(x)
        out_list.append(x)  # 64
        x = self.conv_4(x)
        out_list.append(x)  # 32
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super(MaskDecoder, self).__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64)  # 128
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128)  # 256
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)), # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))  # 64
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))  # 128
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))  # 256
        x = self.mask_conv(x)  # (B,out_channel,T,F)
        x = x.permute(0, 3, 2, 1)  # (B,F,T,out_channel)
        x = self.lsigmoid(x).permute(0, 3, 2, 1)
        return x


class ABNet(nn.Module):
    def __init__(self, num_channels=16, n_fft=512, hop_length=256, compress_factor=0.3):
        super(ABNet, self).__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.compress_factor = compress_factor

        self.encoder = Encoder(in_channels=3, num_channels=num_channels)

        '''
        self.ref_dpr = DPR(
            emb_dim=num_channels,
            hidden_dim=num_channels // 2 * 3,
            n_freqs=self.n_freqs // (2 ** 3),
            dropout_p=0.1,
        )

        self.other_dpr = DPR(
            emb_dim=num_channels,
            hidden_dim=num_channels // 2 * 3,
            n_freqs=self.n_freqs // (2 ** 3),
            dropout_p=0.1,
        )
        '''
        
        self.all_dpr = DPR(
            emb_dim=num_channels,
            hidden_dim=num_channels // 2 * 3,
            n_freqs=self.n_freqs // (2 ** 3),
            dropout_p=0.1,
        )

        self.attn_block1 = AttentionBlock(
            emb_dim=num_channels,
            hidden_dim=num_channels // 2 * 3,
            n_heads=4,
        )

        self.align_dpr = nn.Sequential(
            nn.Conv2d(2*num_channels, num_channels, 1),
            DPR(
                emb_dim=num_channels,
                hidden_dim=num_channels // 2 * 3,
                n_freqs=self.n_freqs // (2 ** 3),
                dropout_p=0.1,
            ),
        )

        self.attn_block2 = AttentionBlock(
            emb_dim=num_channels,
            hidden_dim=num_channels // 2 * 3,
            n_heads=4,
        )
        
        self.final_dpr = DPR(
            emb_dim=num_channels,
            hidden_dim=num_channels // 2 * 3,
            n_freqs=self.n_freqs // (2 ** 3),
            dropout_p=0.1,
        )


        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels, out_channel=2, beta=1)

    def apply_stft(self, x, return_complex=True):
        # x:(B,T)
        assert x.ndim == 2
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)  # (B,T,F)
        return spec

    def apply_istft(self, x, length=None):
        # x:(B,T,F)
        assert x.ndim == 3
        x = x.transpose(1, 2)  # (B,F,T)
        audio = torch.istft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            length=length,
            return_complex=False
        )  # (B,T)
        return audio

    def power_compress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** self.compress_factor
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def power_uncompress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** (1.0 / self.compress_factor)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    
    @staticmethod
    def cal_gd(x):
        # x: (B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(b, t, 1, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_gd.sin(), x_gd.cos())

    @staticmethod
    def cal_if(x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_if.sin(), x_if.cos())
    
    def cal_ifd(self, x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  # (-2pi, 2pi]
        x_ifd = x_if - 2 * torch.pi * (self.hop_length / self.n_fft) * torch.arange(f, device=x.device)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def griffinlim(self, mag, pha=None, length=None, n_iter=1, momentum=0.99):
        mag = mag.detach()
        mag = mag ** (1.0 / self.compress_factor) # uncompress
        assert 0 <= momentum < 1
        momentum = momentum / (1 + momentum)
        if pha is None:
            pha = torch.rand(mag.size(), dtype=mag.dtype, device=mag.device)

        tprev = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)
        for _ in range(n_iter):
            inverse = self.apply_istft(torch.complex(mag * pha.cos(), mag * pha.sin()), length=length)
            rebuilt = self.apply_stft(inverse)
            pha = rebuilt
            pha = pha - tprev.mul_(momentum)
            pha = pha.angle()
            tprev = rebuilt

        return pha
    

    def forward(self, src, tgt=None):
        # src: (B, C, time_length), tgt: (B, time_length)
        if tgt == None:
            tgt = src[:, 0, :]
        
        # Note reference channel idx is always 0
        batch_size, num_mics, time_length = src.size()
        src = src.reshape(batch_size*num_mics, time_length)

        src_spec = self.power_compress(self.apply_stft(src))  # (B*C, T, F)
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)

        tgt_spec = self.power_compress(self.apply_stft(tgt))  # (B,T,F)
        tgt_mag = tgt_spec.abs()

        x = torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)  # (B*C,3,T,F)

        encoder_out_list = self.encoder(x)
        x = encoder_out_list[-1]  # (B*C,D,T,F)
        # x = x.reshape(batch_size, num_mics, *x.size()[1:]) # [B,C,D,T,F]
        # x_ref = x[:, :1].flatten(0, 1)  # [B*1,D,T,F]
        # x_ref = self.ref_dpr(x_ref)  # [B*1,D,T,F]

        x = self.all_dpr(x) # (B*C,D,T,F)
        x = x.reshape(batch_size, num_mics, *x.size()[1:])  # (B,C,D,T,F)
        
        x_ref = self.attn_block1(x)  # (B,1,D,T,F)
        
        x = torch.cat([x_ref.expand_as(x), x], dim=2)  # (B,C,2D,T,F)
        x = self.align_dpr(x.flatten(0, 1))  # (B*C,D,T,F)
        x = x.reshape(batch_size, num_mics, *x.size()[1:])  # (B,C,D,T,F)

        x = self.attn_block2(x)  # (B,1,D,T,F)
        x = self.final_dpr(x.flatten(0, 1))  # (B,D,T,F)

        for idx in range(len(encoder_out_list)):
            encoded = encoder_out_list[idx]
            encoder_out_list[idx] = encoded.reshape(batch_size, num_mics, *encoded.size()[1:])[:, 0]  # fetch reference
        x = self.decoder(x, encoder_out_list)  # (B,2,T,F)

        src_mag = src_mag.reshape(batch_size, num_mics, *src_mag.size()[1:])[:, 0]  # (B,T,F)
        est_mag = (x[:, 0] + 1e-8) * src_mag + (x[:, 1] + 1e-8) * src_mag

        src_pha = src_pha.reshape(batch_size, num_mics, *src_pha.size()[1:])[:, 0]  # (B,T,F)
        est_pha = self.griffinlim(est_mag.detach(), src_pha, time_length)
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est = self.apply_istft(self.power_uncompress(est_spec), length=time_length)

        results = {
            'tgt': tgt,
            'tgt_spec': tgt_spec,
            'tgt_mag': tgt_mag,
            'est': est,
            'est_spec': est_spec,
            'est_mag': est_mag,
        }

        return results
