import torch
from torch import nn
import math

from torch.nn import functional as F

SCAN_FORWARD = 0
SCAN_BACKWARD = 1
SCAN_UPWARD = 2
SCAN_DOWNWARD = 3

from einops import rearrange
from einops.layers.torch import Rearrange

from wind_rwkv.rwkv7 import attn, load_attn

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


class RWKV7(nn.Module):
    def __init__(self, dim, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False, **kwargs):
        super().__init__()
        n_embd = dim
        self.layer_id = layer_id
        self.n_embd = n_embd
        dim_att = n_embd

        self.head_size = HEAD_SIZE
        self.n_head = dim_att // self.head_size
        assert dim_att % self.n_head == 0

        load_attn(HEAD_SIZE=HEAD_SIZE)

        # self.shift_func = eval(shift_mode)
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

    @torch.compile
    def forward(self, x, v1, res=None, scan=[SCAN_FORWARD, SCAN_BACKWARD]):
        scan = scan[self.layer_id % len(scan)]
        # trans = Rearrange('b (h w) c -> b (w h) c', h=res[0])
        # restore = Rearrange('b (w h) c -> b (h w) c', h=res[0])
        if scan == SCAN_FORWARD:
            x, v1 = self._forward(x, v1)
        elif scan == SCAN_BACKWARD:
            x, v1 = torch.flip(x, dims=[1]), torch.flip(v1, dims=[1]) if v1 is not None else None
            x, v1 = self._forward(x, v1)
            x, v1 = torch.flip(x, dims=[1]), torch.flip(v1, dims=[1])
        # elif scan == SCAN_DOWNWARD:
        #     x, v1 = trans(x), trans(v1) if v1 is not None else None
        #     x, v1 = self._forward(x, v1, (res[1], res[0]))
        #     x, v1 = restore(x), restore(v1)
        # elif scan == SCAN_UPWARD:
        #     x, v1 = torch.flip(trans(x), dims=[1]), torch.flip(trans(v1), dims=[1]) if v1 is not None else None
        #     x, v1 = self._forward(x, v1, (res[1], res[0]))
        #     x, v1 = restore(torch.flip(x, dims=[1])), restore(torch.flip(v1, dims=[1]))
        return x, v1

    def _forward(self, x, v1, res=None):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

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

        x = attn(r.bfloat16(), w.bfloat16(), k.bfloat16(), v.bfloat16(), -kk.bfloat16(), (kk*a).bfloat16(), HEAD_SIZE=HEAD_SIZE)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v1