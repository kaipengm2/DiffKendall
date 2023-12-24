# NeurIPS 2023 | DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation

paper link: [https://arxiv.org/abs/2307.15317](https://arxiv.org/abs/2307.15317)

------

## ABSTRACT

Few-shot learning aims to adapt models trained on the base dataset to novel tasks where the categories were not seen by the model before. This often leads to a relatively concentrated distribution of feature values across channels on novel classes, posing challenges in determining channel importance for novel tasks. Standard few-shot learning methods employ geometric similarity metrics such as cosine similarity and negative Euclidean distance to gauge the semantic relatedness between two features. However, features with high geometric similarities may carry distinct semantics, especially in the context of few-shot learning. In this paper, we demonstrate that the importance ranking of feature channels is a more reliable indicator for few-shot learning than geometric similarity metrics. We observe that replacing the geometric similarity metric with Kendall’s rank correlation only during inference is able to improve the performance of few-shot learning across a wide range of methods and datasets with different domains. Furthermore, we propose a carefully designed differentiable loss for meta-training to address the non-differentiability issue of Kendall’s rank correlation. By replacing geometric similarity with differentiable Kendall’s rank correlation, our method can integrate with numerous existing few-shot approaches and is ready for integrating with future state-of-the-art methods that rely on geometric similarity metrics. Extensive experiments validate the efficacy of the rank-correlation-based approach, showcasing a significant improvement in few-shot learning.

------

## Dataset

- miniImageNet
- tiered-ImageNet

Following the previous [setup](https://github.com/Frankluox/Channel_Importance_FSL), we also evaluate the models trained on the mini-ImageNet training set on novel datasets with greater domain differences.

- [CUB](https://docs.google.com/uc?export=download&id=1B8jmZin9teye7Lte9ZKsQ3lyMASbxune)
- [Traffic Signs](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)
- [VGG Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz), [Labels](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
- [Quick Draw](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip)
- [Fungi](https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz)

------

## Pretraining 

We follow this [repo](https://github.com/Sha-Lab/FEAT) to obtain different backbones on mini-ImageNet pretrained with standard cross-entropy loss, and the same training strategies in this [repo](https://github.com/nupurkmr9/S2M2_fewshot) for the S2M2 model. You can directly download the pre-trained model weights from the link [https://drive.google.com/open?id=14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE](https://drive.google.com/open?id=14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE) and the link [https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing](https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing) respectively.

## Using Kendall’s Rank Correlation During Inference

```bash
#!/bin/bash
python eval.py -metric kendall
```

------

## Meta Learning with Differentiable Kendall’s Rank Correlation

```bash
#!/bin/bash
python meta.py -beta 1 -bs 4 -lr 0.001
```

## Reference

[CIM](https://github.com/Frankluox/Channel_Importance_FSL)

[FEAT](https://github.com/Sha-Lab/FEAT)

[DeepEMD](https://github.com/icoz69/DeepEMD/tree/master)

[S2M2](https://github.com/nupurkmr9/S2M2_fewshot)

