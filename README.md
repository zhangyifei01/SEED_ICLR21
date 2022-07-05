## SEED: Self-supervised Distillation for Visual Representation

This is a PyTorch implementation of the **SEED** (ICLR-2021):

We implement SEED based on the official code of [MoCo](https://github.com/facebookresearch/moco).

```
@inproceedings{fang2021seed,
  author  = {Zhiyuan Fang, Jianfeng Wang, Lijuan Wang, Lei Zhang, Yezhou Yang, and Zicheng Liu},
  title   = {SEED: Self-supervised Distillation for Visual Representation},
  booktitle = {ICLR},
  year    = {2021},
}
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on MoCo. Running by:
```
sh train.sh
```
