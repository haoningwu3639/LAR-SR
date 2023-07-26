import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import unfold, fold


class ResidualBlockBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, out_feat=None, res_scale=1, dropout=0., pytorch_init=False):
        super(ResidualBlockBN, self).__init__()
        if out_feat is None:
            out_feat = num_feat
        if res_scale != 1:
            print('warning scale ', res_scale)
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, out_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=num_feat)
        if out_feat != num_feat:
            self.identity = nn.Conv2d(num_feat, out_feat, kernel_size=1)
        else:
            self.identity = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x)
        x = self.dropout(x)
        if self.identity is None:
            identity = x
        else:
            identity = self.identity(x)
        out = self.conv2(self.bn(self.relu(self.conv1(x))))
        return identity + out * self.res_scale


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, mid_channel=32, dropout=0.):
        super(Unet, self).__init__()
        self.conv1 = ResidualBlockBN(in_ch, mid_channel, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ResidualBlockBN(mid_channel, mid_channel, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ResidualBlockBN(mid_channel, mid_channel, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ResidualBlockBN(mid_channel, mid_channel, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ResidualBlockBN(mid_channel, mid_channel, dropout=dropout)

        self.up6 = nn.ConvTranspose2d(mid_channel, mid_channel, 2, stride=2)
        self.conv6 = ResidualBlockBN(mid_channel * 2, mid_channel, dropout=dropout)
        self.up7 = nn.ConvTranspose2d(mid_channel, mid_channel, 2, stride=2)
        self.conv7 = ResidualBlockBN(mid_channel * 2, mid_channel, dropout=dropout)
        self.up8 = nn.ConvTranspose2d(mid_channel, mid_channel, 2, stride=2)
        self.conv8 = ResidualBlockBN(mid_channel * 2, mid_channel, dropout=dropout)
        self.up9 = nn.ConvTranspose2d(mid_channel, mid_channel, 2, stride=2)
        self.conv9 = ResidualBlockBN(mid_channel * 2, mid_channel, dropout=dropout)
        self.conv10 = nn.Conv2d(mid_channel, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        return c10


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, Train=True, decay=0.99, eps=1e-5, init=None):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)  # initialize
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

        if init:
            self.embed.data.copy_(init['embed'])
            self.cluster_size.data.copy_(init['cluster_size'])
            self.embed_avg.data.copy_(init['embed_avg'])
        self.Train = Train

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )  # (flatten - embed)^2
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.Train:
            # print(input.shape)
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()  # ?

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
