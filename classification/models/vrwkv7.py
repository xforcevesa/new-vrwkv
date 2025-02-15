# Copyright (c) Shanghai AI Lab. All rights reserved.
from collections import OrderedDict
from typing import Sequence
import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import random

SCAN_FORWARD = 0
SCAN_BACKWARD = 1
SCAN_UPWARD = 2
SCAN_DOWNWARD = 3

from einops import rearrange
from einops.layers.torch import Rearrange

# from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # x = x.permute(0, 2, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # x = x.permute(0, 2, 1)
        return x
    
class Permute(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        # exit(0)
        return self.func(x)

# from mmcls.models.builder import BACKBONES
# from mmcls.models.utils import resize_pos_embed
# from mmcls.models.backbones.base_backbone import BaseBackbone

# from mmcls_custom.models.utils import DropPath

logger = logging.getLogger(__name__)

T_MAX = 10000
HEAD_SIZE = 32

BaseModule = nn.Module
BaseBackbone = nn.Module

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

CHUNK_LEN = 16
def seqlen_ceil_chunk(x: torch.Tensor):
    seqlen = x.shape[1]
    concat_seqlen = CHUNK_LEN - seqlen % CHUNK_LEN if seqlen % CHUNK_LEN != 0 else 0
    if concat_seqlen != 0:
        concat_tensor = torch.zeros(x.shape[0], concat_seqlen, *x.shape[2:]).to(x.device).to(x.dtype)
        x = torch.concat([x, concat_tensor], dim=1)
    return x

from torch.utils.cpp_extension import load
CUDA_FLAGS = ["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

# Set wind_cuda to True to reproduce the crash
wind_cuda = False

def q_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE, 
                      patch_resolution=None, with_cls_token=False):
    B, N, C = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.transpose(1, 2).reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1])  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0:int(head_dim*1/4), :, shift_pixel:W] = \
        input[:, :, 0:int(head_dim*1/4), :, 0:W-shift_pixel]
    output[:, :, int(head_dim/4):int(head_dim/2), :, 0:W-shift_pixel] = \
        input[:, :, int(head_dim/4):int(head_dim/2), :, shift_pixel:W]
    output[:, :, int(head_dim/2):int(head_dim/4*3), shift_pixel:H, :] = \
        input[:, :, int(head_dim/2):int(head_dim/4*3), 0:H-shift_pixel, :]
    output[:, :, int(head_dim*3/4):int(head_dim), 0:H-shift_pixel, :] = \
        input[:, :, int(head_dim*3/4):int(head_dim), shift_pixel:H, :]
    if with_cls_token:
        output = output.reshape(B, C, N-1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N).transpose(1, 2)
    return output

if wind_cuda:
    # This code may cause SIGABRT on NVIDIA H800 & CUDA 12.5
    load(name="wind", sources=['models/cuda_v7/wind_rwkv7.cu', 'models/cuda_v7/wind_rwkv7.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=CUDA_FLAGS+[f'-D_C_={HEAD_SIZE}'])

    class WindRWKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx,w,q,k,v,a,b):
            B,T,H,C = w.shape
            s0 = torch.zeros(B,H,C,C,dtype=w.dtype,device=w.device)
            assert T%16 == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
            w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
            y = torch.empty_like(v)
            sT = torch.empty_like(s0)
            s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
            torch.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
            ctx.save_for_backward(w,q,k,v,a,b,s)
            return y
        
        @staticmethod
        def backward(ctx,dy):
            w,q,k,v,a,b,s = ctx.saved_tensors
            B,T,H,C = w.shape
            dsT = torch.zeros(B,H,C,C,dtype=dy.dtype,device=dy.device)
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            dy,dsT = [i.contiguous() for i in [dy,dsT]]
            dw,dq,dk,dv,da,db,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
            torch.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
            return dw,dq,dk,dv,da,db
    
    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        dtype = q.dtype
        q,w,k,v,a,b = [seqlen_ceil_chunk(i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE)) for i in [q,w,k,v,a,b]]
        return WindRWKV7.apply(w,q,k,v,a,b).view(B,T,HC)[:, :T, :].to(dtype)

else:
    load(name="wind_backstepping", sources=[f'models/cuda_v7/backstepping_f32_{1 if HEAD_SIZE < 128 else 2}.cu', 'models/cuda_v7/backstepping_f32.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=CUDA_FLAGS+[f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}"])

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            w,q,k,v,z,b = [i.contiguous() for i in [w,q,k,v,z,b]]
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert dy.dtype == torch.bfloat16
            dy = dy.contiguous()
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        dtype = q.dtype
        q,w,k,v,a,b = [seqlen_ceil_chunk(i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE)) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)[:, :T, :].to(dtype)

class RWKV7(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        dim_att = n_embd

        self.head_size = HEAD_SIZE
        self.n_head = dim_att // self.head_size
        assert dim_att % self.n_head == 0

        self.shift_func = eval(shift_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -7 + 5 * (n / (dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,dim_att))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_MIX_LORA = 28
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(ortho_init(torch.zeros(4, D_MIX_LORA, n_embd), 0.1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, dim_att), 0.1))

            D_AAA_LORA = 24
            self.time_aaa_w1 = nn.Parameter(torch.zeros(n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, dim_att), 0.1))

            D_KKK_LORA = 24
            self.time_kkk_w1 = nn.Parameter(torch.zeros(n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(ortho_init(torch.zeros(D_KKK_LORA, dim_att), 0.1))

            D_GATE_LORA = 120
            self.gate_w1 = nn.Parameter(torch.zeros(n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, dim_att), 0.1))

            D_MA_LORA = 24
            self.ma_w1 = nn.Parameter(torch.zeros(n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, dim_att), 0.1))
            self.time_misc_a = nn.Parameter(torch.zeros(1,1,n_embd))
            D_MK_LORA = 24
            self.mk_w1 = nn.Parameter(torch.zeros(n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, dim_att), 0.1))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,n_embd))
            if layer_id != 0:
                D_MV_LORA = 24
                self.mv_w1 = nn.Parameter(torch.zeros(n_embd, D_MV_LORA))
                self.mv_w2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, dim_att), 0.1))
                self.time_misc_v = nn.Parameter(torch.zeros(1,1,n_embd)+1.0)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(n_embd, dim_att, bias=False)
            self.key = nn.Linear(n_embd, dim_att, bias=False)
            self.value = nn.Linear(n_embd, dim_att, bias=False)
            self.output = nn.Linear(dim_att, n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, dim_att, eps=6.4e-5)

            self.receptance.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.key.weight.data.uniform_(-0.05/(self.n_embd**0.5), 0.05/(self.n_embd**0.5))
            self.value.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.output.weight.data.zero_()

            self.n_layer = n_layer

    def forward(self, x, v1, res, scan):
        scan = scan[self.layer_id % len(scan)]
        trans = Rearrange('b (h w) c -> b (w h) c', h=res[0])
        restore = Rearrange('b (w h) c -> b (h w) c', h=res[0])
        if scan == SCAN_FORWARD:
            x, v1 = self._forward(x, v1, res)
        elif scan == SCAN_BACKWARD:
            x, v1 = torch.flip(x, dims=[1]), torch.flip(v1, dims=[1]) if v1 is not None else None
            x, v1 = self._forward(x, v1, res)
            x, v1 = torch.flip(x, dims=[1]), torch.flip(v1, dims=[1])
        elif scan == SCAN_DOWNWARD:
            x, v1 = trans(x), trans(v1) if v1 is not None else None
            x, v1 = self._forward(x, v1, (res[1], res[0]))
            x, v1 = restore(x), restore(v1)
        elif scan == SCAN_UPWARD:
            x, v1 = torch.flip(trans(x), dims=[1]), torch.flip(trans(v1), dims=[1]) if v1 is not None else None
            x, v1 = self._forward(x, v1, (res[1], res[0]))
            x, v1 = restore(torch.flip(x, dims=[1])), restore(torch.flip(v1, dims=[1]))
        return x, v1

    def _forward(self, x, v1, res):
        B, T, C = x.size()
        H = self.n_head
        xx = self.shift_func(x, self.shift_pixel, patch_resolution=res, 
                             with_cls_token=self.with_cls_token) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        xrg, xwa, xk, xv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + xrg)
        xwa = x + xx * (self.time_maa_wa + xwa)
        xk = x + xx * (self.time_maa_k + xk)
        xv = x + xx * (self.time_maa_v + xv)

        r = self.receptance(xrg)
        w = -F.softplus(-(self.time_decay + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v1 = v
        else:
            v = v + (v1 - v) * torch.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        g = torch.sigmoid(xrg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        a = torch.sigmoid(self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2)

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k*a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * torch.clamp(w*mk, max=0).exp()

        x = RUN_CUDA_RWKV7g(r.bfloat16(), w.bfloat16(), k.bfloat16(), v.bfloat16(), -kk.bfloat16(), (kk*a).bfloat16())
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v1
    
class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False, 
                 with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        print(f'{self.attn_sz = }, {self.n_head = }')
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.shift_func = eval(shift_mode)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
                                 with_cls_token=self.with_cls_token)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
    
class RWKV7Block(nn.Module):

    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.attn = RWKV7(n_embd, n_head, n_layer, layer_id, shift_mode,
                 shift_pixel, drop_path, hidden_rate, init_mode,
                 init_values, post_norm, key_norm, with_cls_token,
                 with_cp)
        self.mlp = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                    with_cls_token=with_cls_token)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, v1, x0, res, scan):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(self.ln1(x), v1, res, scan)
        x = x + x1
        x = x + self.mlp(self.ln2(x), res)
        return x, v1

def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)

# @BACKBONES.register_module()
class VRWKV7(BaseBackbone):
    def __init__(self,
                 img_size=224,
                 dims=[96, 192, 384, 768],
                 patch_size=16,
                 in_channels=3,
                #  out_indices=-1,
                 drop_rate=0.,
                 embed_dims=192,
                 depth=12,
                 drop_path_rate=0.,
                 shift_pixel=1,
                 shift_mode='q_shift_multihead',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 output_cls_token=False,
                 with_cls_token=False,
                 with_cp=False,
                 norm_layer="LN", # "BN", "LN2D"
                 num_classes=1000, 
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = len(depth)
        self.drop_path_rate = drop_path_rate

        num_heads = embed_dims // HEAD_SIZE

        # if isinstance(dims, int):
        #     dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        # self.num_features = dims[-1]
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        
        self.scan_method = [
            SCAN_FORWARD,
            SCAN_BACKWARD,
            SCAN_UPWARD,
            SCAN_DOWNWARD
        ]
        
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))
        
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # if isinstance(out_indices, int):
        #     out_indices = [out_indices]
        # assert isinstance(out_indices, Sequence), \
        #     f'"out_indices" must by a sequence or int, ' \
        #     f'get {type(out_indices)} instead.'
        # for i, index in enumerate(out_indices):
        #     if index < 0:
        #         out_indices[i] = self.num_layers + index
        #     assert 0 <= out_indices[i] <= self.num_layers, \
        #         f'Invalid out_indices {index}'
        # self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]
        self.layers = nn.ModuleList()
        # from . import vrwkv6
        for i in range(self.num_layers):
            self.layers.append(RWKV7Block(
                n_embd=embed_dims,
                n_head=num_heads,
                n_layer=self.num_layers,
                layer_id=i,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cls_token=with_cls_token,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)
        def permute_reshape(x):
            x = x.permute(0, 2, 1)
            seqlen = x.shape[-1]
            permute_head_size = math.ceil(seqlen ** 0.5)
            while seqlen % permute_head_size != 0:
                permute_head_size -= 1
            x = x.view(*x.shape[:2], seqlen // permute_head_size, permute_head_size)
            return x
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embed_dims, num_classes)
        )


    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # self.scan_method = list(reversed(self.scan_method))

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)  # post cls_token

        x = self.drop_after_pos(x)

        v1 = None
        x0 = x

        for i, layer in enumerate(self.layers):
            if isinstance(layer, RWKV7Block):
                x, v1 = layer(x, v1, x0, patch_resolution, self.scan_method)
            else:
                x = layer(x, patch_resolution)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

        # print(x.shape)
        x = x.mean(dim=1)
        x = self.classifier(x)
                
        return x

