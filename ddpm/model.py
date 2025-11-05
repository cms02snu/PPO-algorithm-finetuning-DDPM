import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def swish(x):
    return x * torch.sigmoid(x)

def get_timestep_embedding(t, channel):
    half = channel // 2
    device = t.device
    inv_freq = torch.exp(-math.log(10000) * torch.arange(half) / half).to(device)
    args = t.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=8, eps=1e-6):
        super().__init__(num_groups, num_channels, eps=eps)

def conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, init_scale=1.0):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
    with torch.no_grad():
        conv.weight.data *= init_scale
    return conv

def nin(in_ch, out_ch, init_scale=1.0):
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
    with torch.no_grad():
        layer.weight.data *= init_scale
    return layer

def linear(in_features, out_features, init_scale=1.0):
    fc = nn.Linear(in_features, out_features)
    with torch.no_grad():
        fc.weight.data *= init_scale
    return fc

class DownsampleBlock(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class UpsampleBlock(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=256, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.dropout = dropout
        self.norm1 = GroupNorm(self.in_channels)
        self.conv1 = conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, init_scale=1.0)
        self.temb_proj = linear(self.temb_channels, self.out_channels, init_scale=1.0)
        self.norm2 = GroupNorm(self.out_channels)
        self.conv2 = conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, init_scale=1.0)
        self.conv_shortcut = nin(self.in_channels, self.out_channels)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h_temb = swish(temb)
        h_temb = self.temb_proj(h_temb)
        h_temb = h_temb[:, :, None, None]
        h = h + h_temb
        h = self.norm2(h)
        h = swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)
        x = self.conv_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.q = nin(channels, channels)
        self.k = nin(channels, channels)
        self.v = nin(channels, channels)
        self.proj_out = nin(channels, channels, init_scale=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W).permute(0, 2, 1)
        v = v.view(B, C, H * W).permute(0, 2, 1)
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        h_ = torch.bmm(attn, v)
        h_ = h_.permute(0, 2, 1).reshape(B, C, H, W)
        h_ = self.proj_out(h_)
        return x + h_

class DDPMModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={32},
        dropout=0.0,
        resamp_with_conv=False,
        init_resolution=64
    ):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.num_levels = len(ch_mult)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resamp_with_conv = resamp_with_conv
        self.init_resolution = init_resolution
        self.temb_ch = ch * 4
        self.temb_dense0 = linear(self.ch, self.temb_ch)
        self.temb_dense1 = linear(self.temb_ch, self.temb_ch)
        self.conv_in = conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList()
        curr_ch = ch
        curr_res = init_resolution
        for level in range(self.num_levels):
            level_blocks = nn.ModuleList()
            out_ch = ch * ch_mult[level]
            for i in range(num_res_blocks):
                level_blocks.append(ResnetBlock(curr_ch, out_ch, temb_channels=self.temb_ch, dropout=dropout))
                if curr_res in attn_resolutions:
                    level_blocks.append(AttnBlock(out_ch))
                curr_ch = out_ch
            self.down_blocks.append(level_blocks)
            if level != self.num_levels - 1:
                self.down_blocks.append(DownsampleBlock(curr_ch, with_conv=resamp_with_conv))
                curr_res //= 2
        self.mid_block = nn.ModuleList([
            ResnetBlock(curr_ch, curr_ch, temb_channels=self.temb_ch, dropout=dropout),
            AttnBlock(curr_ch),
            ResnetBlock(curr_ch, curr_ch, temb_channels=self.temb_ch, dropout=dropout)
        ])
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            level_blocks = nn.ModuleList()
            out_ch = ch * ch_mult[level]
            level_blocks.append(ResnetBlock(curr_ch + out_ch, out_ch, temb_channels=self.temb_ch, dropout=dropout))
            if (init_resolution // (2 ** level)) in attn_resolutions:
                level_blocks.append(AttnBlock(out_ch))
            curr_ch = out_ch
            for i in range(num_res_blocks):
                level_blocks.append(ResnetBlock(curr_ch, curr_ch, temb_channels=self.temb_ch, dropout=dropout))
                if (init_resolution // (2 ** level)) in attn_resolutions:
                    level_blocks.append(AttnBlock(curr_ch))
            if level != 0:
                level_blocks.append(UpsampleBlock(curr_ch, with_conv=resamp_with_conv))
            self.up_blocks.append(level_blocks)
        self.norm_out = GroupNorm(curr_ch)
        self.conv_out = conv2d(curr_ch, out_channels, kernel_size=3, stride=1, padding=1, init_scale=0.0)

    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = swish(temb)
        temb = self.temb_dense1(temb)
        skips = []
        h = self.conv_in(x)
        down_iter = iter(self.down_blocks)
        for level in range(self.num_levels):
            blocks = next(down_iter)
            for layer in blocks:
                h = layer(h, temb) if isinstance(layer, ResnetBlock) else layer(h)
            skips.append(h)
            if level != self.num_levels - 1:
                downsample = next(down_iter)
                h = downsample(h)
        for layer in self.mid_block:
            h = layer(h, temb) if isinstance(layer, ResnetBlock) else layer(h)
        for level in range(self.num_levels):
            blocks = self.up_blocks[level]
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = blocks[0](h, temb)
            for layer in blocks[1:]:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, temb)
                else:
                    h = layer(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h