import os
import time
import argparse
import torch
import torch.nn as nn

from fold_model import Local_Ar
from res_vq import res_vqvae


from prepare_data import imread
from imresize import imresize




all_imgs = os.scandir('input_imgs')
class get_args():
    def __init__(self):
        self.resume = 'ckpt/0.100427023103448.pth'
        self.res_vq_resume = 'ckpt/vqvae_030_0.000112.pt'
        self.bottom_class = 1024

        self.gpus = '0'



args = get_args()
vqvae = res_vqvae(stage='sample', backbone='RRDB', n_embed=args.bottom_class)
ckpt = torch.load(args.res_vq_resume)
vqvae.load_state_dict(ckpt)
vqvae = vqvae.cuda()
print("load successfully")
vqvae.eval()

# model = Local_Ar(n_embed=args.bottom_class, block = 10, scale=4).cuda()
model = Local_Ar(n_embed=args.bottom_class, block=10, scale=4).cuda()
model.load_state_dict(torch.load(args.resume))
gpus = args.gpus.split(',')
gpus = [int(x) for x in gpus]
model = nn.DataParallel(model, device_ids=gpus)
model.eval()

import numpy as np
import cv2

save_path = 'result/sr'
gt_path = 'result/gt'
lr_path = 'result/lr'

vqvae.eval()
model.eval()
all_time = 0.
count = 0

for i, img_file in enumerate(all_imgs):
    img = imread(img_file.path)
    if img.shape[0] % 32 != 0:
        img = img[:-(img.shape[0] % 32)]
    if img.shape[1] % 32 != 0:
        img = img[:, :-(img.shape[1] % 32)]


    img_lr = imresize(img, scalar_scale=0.25)

    with torch.no_grad():
        # img = imresize(img_lr, scalar_scale=4.)
        img = img / 255.0
        img_lr = img_lr / 255.0
        lr = torch.Tensor(img_lr).cuda().float().permute(2, 0, 1).unsqueeze(0)
        hr = torch.Tensor(img).cuda().float().permute(2, 0, 1).unsqueeze(0)
        torch.cuda.synchronize()
        start = time.time()
        cause_sr_img = vqvae.cause_sr(lr)

        sample_id = model.module.pred(cause_sr_img.detach().float()[:8], 0.05)
        gen_img = vqvae.decode_code(lr[:8], sample_id)
        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start
        print(end - start)
        count += 1
        gen_img = gen_img[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
    gen_img = np.clip(gen_img * 255.0, a_min=0., a_max=255.)

    gt = hr[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
    gt = np.clip(gt * 255.0, a_min=0., a_max=255.)

    img_lr = lr[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
    img_lr = np.clip(img_lr * 255.0, a_min=0., a_max=255.)

    cv2.imwrite(os.path.join(lr_path, img_file.name), img_lr.astype(np.uint8))
    cv2.imwrite(os.path.join(save_path, str(i + 6) + '.png'), gen_img.astype(np.uint8))
    cv2.imwrite(os.path.join(gt_path, img_file.name), gt.astype(np.uint8))
print(all_time / count)