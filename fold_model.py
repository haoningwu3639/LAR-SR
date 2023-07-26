import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from blocks import Unet, ResidualBlockBN
from utils import fold, unfold



class gate_layer(nn.Module):
    def __init__(self, channel, scale):
        super(gate_layer, self).__init__()
        self.blocks = nn.ModuleList([])
        for i in range(scale * scale):
            self.blocks.append(ResidualBlockBN(num_feat = (i+1) * channel, out_feat = channel, dropout=0.1))
        self.n = scale * scale
    def forward(self, x):
        '''

        :param x: (N, C, h * w, H, W)
        :return: (N, C, h * w, H, W)
        '''
        N, C, l, H, W = x.shape
        # if l < self.n:
        #     input = x.reshape(N, -1, H, W)
        #     out = self.blocks[l-1](input).unsqueeze(2)
        #     return out
        x = torch.split(x, 1, dim=2)   #list([N, C, H, W],..., [])
        output = []
        for i in range(self.n):
            input = torch.cat(x[:i+1], dim=2)  #(N, C, i, H, W)
            # input = x[:, :, :i+1]
            input = input.reshape(N, -1, H, W)
            out = self.blocks[i](input).unsqueeze(2)  #(N, C, )
            output.append(out)
            # if output is None:
            #     output = out
            # else:
            #     #print(out.device, output.device)
            #     output = torch.cat([output, out], dim=2)
        output = torch.cat(output, dim=2)
        return output






class split_gate_layer(nn.Module):
    def __init__(self, channel, scale):
        super(split_gate_layer, self).__init__()
        self.w_blocks = nn.ModuleList([])
        self.h_blocks = nn.ModuleList([])

        for i in range(scale * scale):
            self.w_blocks.append(ResidualBlockBN(num_feat=channel, out_feat=channel, dropout=0.1))
            self.h_blocks.append(nn.Sequential(
                nn.Conv2d(( i + 1) * channel, channel, kernel_size=1),
                nn.Dropout(0.1),
                nn.ELU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=1)
            ))

        self.n = scale * scale

    def forward(self, x):
        '''

        :param x: (N, C, h * w, H, W)
        :return: (N, C, h * w, H, W)
        '''
        N, C, l, H, W = x.shape
        # if l < self.n:
        #     input = x.reshape(N, -1, H, W)
        #     out = self.blocks[l-1](input).unsqueeze(2)
        #     return out
        x = torch.split(x, 1, dim=2)  # list([N, C, H, W],..., [])
        output = []
        for i in range(self.n):
            input = torch.cat(x[:i + 1], dim=2)  # (N, C, i, H, W)

            # input = x[:, :, :i+1]
            w_input = input.permute(0, 2, 1, 3, 4)
            w_input = w_input.reshape(N * (i + 1), C, H, W)
            w_input = self.w_blocks[i](w_input).reshape(N, i+1, -1, H, W).permute(0, 2, 1, 3, 4)

            h_input = w_input.reshape(N, -1, H, W)
            h_out = self.h_blocks[i](h_input).unsqueeze(2)  # (N, C, )
            output.append(h_out)
            # if output is None:
            #     output = out
            # else:
            #     #print(out.device, output.device)
            #     output = torch.cat([output, out], dim=2)
        output = torch.cat(output, dim=2)
        return output


class Local_Ar(nn.Module):
    def __init__(self, scale = 4, channel = 64, n_embed = 128, block = 10):
        super(Local_Ar, self).__init__()


        self.n_embed = n_embed

        self.scale = scale

        self.sr_encode = nn.Sequential(
            Unet(3, channel, channel),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
        )

        num_ds = int(np.log2(self.scale))
        ds_blocks = []
        ds_blocks.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1))
        ds_blocks.append(nn.ELU(inplace=True))
        for i in range(num_ds):
            ds_blocks.append(nn.Conv2d(channel, channel, kernel_size=4, padding=1, stride=2))
            ds_blocks.append(nn.ELU(inplace=True))
            ds_blocks.append(nn.Conv2d(channel, channel, kernel_size=1))
        self.init_conv = nn.Sequential(*ds_blocks)

        self.x_encode = nn.Sequential(
            nn.Linear(n_embed, channel),
            nn.ELU(inplace=True),
            nn.Linear(channel, channel),
        )

        self.aggressive_encode = nn.Sequential(
            nn.Linear(2 * channel, channel),
            nn.ELU(inplace=True),
            nn.Linear(channel, channel),
            nn.ELU(inplace=True),
            nn.Linear(channel, channel),
        )

        self.gate_conv_group = nn.ModuleList([])
        for i in range(block):
            self.gate_conv_group.append(split_gate_layer(channel, scale))

        self.out_conv = nn.Sequential(
            nn.Linear(2 * channel, channel),
            nn.ELU(inplace=True),
            nn.Linear(channel, channel),
            nn.ELU(inplace=True),
            nn.Linear(channel, n_embed)
        )
    def forward(self, x, sr):
        B, H, W = x.shape

        sr = self.sr_encode(sr)
        sr_feature = unfold(sr, self.scale, H // self.scale, W // self.scale)
        sr_feature = sr_feature.permute(0, 2, 3, 4, 1)

        init_feature = self.init_conv(sr)

        x = F.one_hot(x, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        x = unfold(x, self.scale, H // self.scale, W // self.scale)
        x = x.permute(0, 2, 3, 4, 1)
        x_feature = self.x_encode(x)

        agg_feature = torch.cat([x_feature, sr_feature], dim=-1)
        agg_feature = self.aggressive_encode(agg_feature)

        agg_feature = agg_feature.permute(0, 4, 1, 2, 3)
        agg_feature = agg_feature[:, :, :-1]

        agg_feature = torch.cat([init_feature.unsqueeze(2), agg_feature], dim=2)

        for conv in self.gate_conv_group:
            agg_feature = conv(agg_feature)

        agg_feature = agg_feature.permute(0, 2, 3, 4, 1)

        out = self.out_conv(torch.cat([agg_feature, sr_feature], dim=-1))
        out = out.permute(0, 4, 1, 2, 3)

        out = fold(out, self.scale, H // self.scale, W // self.scale)
        return out

    def pred(self, sr, thed = 0.):
        B, _, H, W = sr.shape
        H = H // 2
        W = W // 2

        sr = self.sr_encode(sr)
        sr_feature = unfold(sr, self.scale, H // self.scale, W // self.scale)
        sr_feature = sr_feature.permute(0, 2, 3, 4, 1)

        init_feature = self.init_conv(sr).unsqueeze(2)

        cache = torch.zeros(init_feature.shape).repeat(1, 1, self.scale * self.scale, 1, 1).float().to(init_feature.device)
        out_cache = torch.zeros(B, self.scale * self.scale, H // self.scale , W // self.scale ).long().to(init_feature.device)

        for i in range(self.scale * self.scale):
            input = torch.cat([init_feature, cache[:, :, :-1]], dim=2)
            for conv in self.gate_conv_group:
                input = conv(input)
            input = input.permute(0, 2, 3, 4, 1)
            out = self.out_conv(torch.cat([input, sr_feature], dim=-1))
            out = out[:, i].squeeze(1)
            out = F.softmax(out, dim=-1)
            out = out.reshape(-1, self.n_embed)

            np.random.seed(int(random.random() * 9999999))
            max_flag = np.random.rand(out.shape[0])

            max_mask = torch.from_numpy(( max_flag < thed ).astype(np.float)).float().to(out.device)
            inv_max_mask = 1 - max_mask

            arg_out = torch.argmax(out, dim=1, keepdim=False)
            out = torch.multinomial(out, 1).squeeze(-1)
            out = out * inv_max_mask + arg_out * max_mask

            out = torch.round(out.reshape(B, H // self.scale, W // self.scale)).long()
            out_cache[:, i] = out

            if i != self.scale * self.scale - 1:
                input = F.one_hot(out, num_classes = self.n_embed).float()
                input = self.x_encode(input)
                input = torch.cat([input, sr_feature[:, i]], dim=-1)
                input = self.aggressive_encode(input)
                input = input.permute(0, 3, 1, 2)
                cache[:, :, i] = input

        out_cache = out_cache.reshape(-1, self.scale, self.scale, H // self.scale, W // self.scale)
        out_cache = out_cache.permute(0, 3, 1, 4, 2)
        out_cache = out_cache.reshape(B, H, W)

        return out_cache









