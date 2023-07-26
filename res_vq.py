import torch
from torch import nn
from torch.nn import functional as F

from backbone import _NetG, RRDBNet

class simple_csr(nn.Module):
    def __init__(self):
        super(simple_csr, self).__init__()
    def forward(self, x):
        return x

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, Train=True, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)  # initialize
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
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


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        # Downsample Multi-scale + ResBlock
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]
        elif stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 1:
            blocks = [
                nn.Conv2d(in_channel, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    # TransConv Upsample
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )
        elif stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel, out_channel, 4, stride=2, padding=1
                    )
                ]
            )
        elif stride == 1:
            blocks.extend(
                [
                    nn.Conv2d(channel, channel, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, out_channel, kernel_size=1),
                ]
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class res_vqvae(nn.Module):
    def __init__(self,
                 in_channel=3,
                 channel=128,
                 n_res_block=4,
                 n_res_channel=32,
                 embed_dim=64,
                 n_embed=128,
                 stage='train',
                 cause_sr_resume = '',
                 backbone = 'RRDB',
                 decay=0.99,
                 ):
        super(res_vqvae, self).__init__()
        print('vqvae n_embed ', n_embed)
        print('stride ', 8)
        print('backbone', backbone)


        if backbone == 'srresnet':
            self.cause_sr = _NetG()
        elif backbone == 'bicubic':
            self.cause_sr = simple_csr()
        elif backbone == 'RRDB':
            self.cause_sr = RRDBNet()
        else:
            raise NotImplemented()
        if cause_sr_resume:
            try:
                self.cause_sr.load_state_dict(torch.load(cause_sr_resume))
            except:
                self.cause_sr.load_state_dict(torch.load(cause_sr_resume)["model"].state_dict())
        else:
            if backbone != 'bicubic' and stage != 'sample':
                raise FileNotFoundError()

        for name, param in self.cause_sr.named_parameters():
            param.requires_grad = False

        self.enc_b = Encoder(in_channel * 2, channel, n_res_block, n_res_channel, stride=2)

        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        if stage != 'train':
            self.quantize_b = Quantize(embed_dim, n_embed, Train=False)
        else:
            self.quantize_b = Quantize(embed_dim, n_embed)

        self.enc_t = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=2)
        self.dec = Decoder(
            embed_dim + channel,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=2,
        )

    def encode(self, hr_input, lr_input):

        sr_input = self.cause_sr(lr_input)

        enc_b = self.enc_b(torch.cat([hr_input, sr_input.detach()], dim=1))

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)

        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        return quant_b, diff_b, id_b, sr_input

    def decode_code(self, lr_input, code_b):
        cause_sr = self.cause_sr(lr_input)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        enc_sr = self.enc_t(cause_sr.detach())
        quant = torch.cat([enc_sr, quant_b], dim=1)

        dec = self.dec(quant)

        return dec + cause_sr


    def forward(self, hr_input, lr_input):
        quant_b, diff_b, id_b, cause_sr = self.encode(hr_input, lr_input)
        enc_sr = self.enc_t(cause_sr.detach())
        quant = torch.cat([enc_sr, quant_b], dim=1)
        dec = self.dec(quant)
        dec = dec + cause_sr
        return dec, diff_b, cause_sr


if __name__ == '__main__':
    img_path = ''