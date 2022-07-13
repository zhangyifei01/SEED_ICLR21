## SEED: Self-supervised Distillation for Visual Representation

This is an unofficial PyTorch implementation of the **SEED** (ICLR-2021):

We implement SEED based on the official code of [MoCo](https://github.com/facebookresearch/moco).



### **Implementation Results**
Teacher model is MoCo-v2 (top-1: 67.6) pretrained on ImageNet-1k with ResNet-50.
We distill it to ResNet-18. 
Results show that our code is credible.

<table>
  <tr>
    <th>SEED</th>
    <th>Top-1 acc</th>
    <th>Top-5 acc</th>
  </tr>
  <tr>
    <td>Official results (hidden_dim=512)</td>
    <td>57.60</td>
    <td>81.80</td>
  </tr>
  <tr>
    <td>**Ours** (hidden_dim=512)</td>
    <td>58.03</td>
    <td>82.44</td>
  </tr>
  <tr>
    <td>**Ours** (hidden_dim=2048)</td>
    <td>60.31</td>
    <td>83.56</td>
  </tr>
</table>

Hidden dimension (hidden_dim) can be modified by 
```
self.encoder_q.fc = nn.Sequential(nn.Linear(dim_smlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
```

### **Start Training**

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on [MoCo](https://github.com/facebookresearch/moco). Running by:
```
sh train.sh
```

### **To Do**
More student architectures.



### **Citation**
```
@inproceedings{fang2021seed,
  author  = {Zhiyuan Fang, Jianfeng Wang, Lijuan Wang, Lei Zhang, Yezhou Yang, and Zicheng Liu},
  title   = {SEED: Self-supervised Distillation for Visual Representation},
  booktitle = {ICLR},
  year    = {2021},
}
```
