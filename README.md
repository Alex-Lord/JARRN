# Joint Absolute-Relative Relation Network for Generalizable Depth Completion

**[paper](https://github.com/Alex-Lord/Joint-Absolute-Relative-Relation-Network-for-Generalizable-Depth-Completion)**

**Bingyuan Chen, Ruizhe Zhang, Haotian Wang, Lingzhi Pan, Meng Yang, Xinhu Zheng, Gang Hua**

## News

- Training code will be released soon! `2025`

## Abstract

Completing dense depth maps from 2D visual images and sparse depth measurements is crucial to robustly perceive 3D scenes. Recent depth completion methods have made significant advances in single scenes, nevertheless, they do not well generalize across unseen scenes.
One fundamental limitation lies in that current methods directly use conventional network architectures from 2D visual tasks.
However, depth completion predicts depth maps with additional absolute distances in 3D space.
To bridge this gap, we propose a new network architecture to effectively leverage conventional networks of 2D images for generalizable depth completion.
We first formulate a novel joint absolute-relative relation with three learnable factors between output depth and GT depth. Based on this relation, we successfully decouple the task of generalizable depth completion in 3D space into two sequential branches in 2D space, namely, visual and refine branches. The visual branch utilizes a conventional modified U-Net to generate dense depth from sparse ones. The refine branch utilizes another modified U-Net to learn three maps based on the three learnable factors, namely, scale, offset, and probability, from sparse depth and dense depth, and then rectifies the absolute output depth with the learned maps.

Our model is trained on a mixture of two large-scale datasets and extensively tested on eight unseen datasets of different scenes with a large range of randomly sampled and LiDAR pattern sparse depth points. The results demonstrate that our model consistently achieves significant gains with moderate low cost, when compared to both officially released and retrained models of state-of-the-art baselines.

## Requirments

Python=3.8

Pytorch=2.3

## Train
coming soon
#### Model weights
**[google drive](https://drive.google.com/drive/folders/1uWVAgbockgQOVohXyKIJXbiX89tdUUvA?usp=sharing)**
