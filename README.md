# LAR-SR: A Local Autoregressive Model for Image Super-Resolution (CVPR2022)

Here is the official implementation for CVPR 2022 paper "LAR-SR: A Local Autoregressive Model for Image Super-Resolution".

## NOTE

res_vq.py: textural VQVAE in Stage 1

fold_model.py: LAR-module in Stage 2

fold_attention_model.py: LAR-attn-layer based LAR-module

test.py: test the model with pretrained checkpoint

Due to the size limitation, the checkpoints are not uploaded

## Citation
If you use this code for your research or project, please cite:

    @inproceedings{guo2022lar,
    title={Lar-sr: A local autoregressive model for image super-resolution},
    author={Guo, Baisong and Zhang, Xiaoyun and Wu, Haoning and Wang, Yu and Zhang, Ya and Wang, Yan-Feng},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={1909--1918},
    year={2022}
    }