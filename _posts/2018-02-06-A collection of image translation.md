---
layout:     post
title:      A collection of image translation
date:       2018-02-06
author:     Will Hsia
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - image translation
    - index
---

## A collection of Image Translation methods.
- `pix2pix`: [[homepage](https://phillipi.github.io/pix2pix/)] [[Code](https://github.com/phillipi/pix2pix)]  [[Paper](https://arxiv.org/pdf/1611.07004.pdf)]
- `CycleGAN`: [[homepage](https://junyanz.github.io/CycleGAN/)] [[CycleGAN](https://github.com/junyanz/CycleGAN)] [[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)] [[Full Paper](https://arxiv.org/pdf/1703.10593.pdf)]
- `DualGAN`: [[Code](https://github.com/duxingren14/DualGAN)] [[Paper](https://arxiv.org/abs/1704.02510)]
- `DiscoGAN`: [[Code](https://github.com/carpedm20/DiscoGAN-pytorch)] [[Paper](https://arxiv.org/abs/1703.05192)]
- `DTN`: [[Code](https://github.com/yunjey/domain-transfer-network)] [[Paper](https://arxiv.org/abs/1611.02200)]
- `FaderNets`: [[Code](https://github.com/facebookresearch/FaderNetworks)] [[Paper](https://arxiv.org/abs/1706.00409)]
- `IcGAN`: [[Code](https://github.com/Guim3/IcGAN)] [[Paper](https://arxiv.org/abs/1611.06355)]
- `GeneGAN`: [[Code](https://github.com/Prinsphield/GeneGAN)] [[Paper](https://arxiv.org/abs/1705.04932)]
- `Face-Age-cGAN`: [[Paper](https://arxiv.org/abs/1702.01983)]
- `BicycleGAN`: [[Code](https://github.com/junyanz/BicycleGAN)] [[Tensorflow](https://github.com/gitlimlab/BicycleGAN-Tensorflow)]
- `StarGAN`: CVPR 2018. [[Code](https://github.com/yunjey/StarGAN)]  [[Paper](https://arxiv.org/abs/1711.09020)]
- `VAE-GAN`: [[Code](http://github.com/andersbll/autoencoding_beyond_pixels)] [[Paper](https://arxiv.org/pdf/1611.07004.pdf)]
- `cVAE-GAN`: [[Paper](https://arxiv.org/pdf/1703.10155.pdf)]
## datasets
Please cite their papers if you use the data.
### pix2pix Datasets
Some datasets can also be downloaded manually from the website or **automatically** using the following script:
```python
python download-dataset.py datasetname
```
- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). (31MB) 
- `sketch`: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/
- `oil-chinese`: http://www.cs.mun.ca/~yz7241/dataset/
- `day-night`: http://www.cs.mun.ca/~yz7241/dataset/
- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade). [[Citation](datasets/bibtex/facades.tex)]
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com). [[Citation](datasets/bibtex/cityscapes.tex)]
- `maps`: 1096 training images scraped from Google Maps
- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. [[Citation](datasets/bibtex/shoes.tex)]
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. [[Citation](datasets/bibtex/handbags.tex)]
### CycleGAN Datasets
- `facades`: 400 images from the [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade). [[Citation](datasets/bibtex/facades.tex)]
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com). [[Citation](datasets/bibtex/cityscapes.tex)]
- `maps`: 1096 training images scraped from Google Maps.
- `horse2zebra`: 939 horse images and 1177 zebra images downloaded from [ImageNet](http://www.image-net.org) using keywords `wild horse` and `zebra`
- `apple2orange`: 996 apple images and 1020 orange images downloaded from [ImageNet](http://www.image-net.org) using keywords `apple` and `navel orange`.
- `summer2winter_yosemite`: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.
- `monet2photo`, `vangogh2photo`, `ukiyoe2photo`, `cezanne2photo`: The art images were downloaded from [Wikiart](https://www.wikiart.org/). The real photos are downloaded from Flickr using the combination of the tags *landscape* and *landscapephotography*. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
- `iphone2dslr_flower`: both classes of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316.

### Attribute Editing
- `CelebA`. The CelebFaces Attributes (CelebA) dataset contains 202,599 face images of celebrities, each annotated with 40 binary attributes. size 178×218. hair color (black, blond, brown),gender (male/female), and age (young/old).
- `RaFD`. The Radboud Faces Database (RaFD) consists of 4,824 images collected from 67 participants. Each participant makes eight facial expressions in three different gaze directions, which are captured from three different angles.

## Reference：
1. Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[J]. arXiv preprint arXiv:1611.07004, 2016.
2. Zhu J Y, Park T, Isola P, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks[J]. arXiv preprint arXiv:1703.10593, 2017.
3. Yi Z, Zhang H, Gong P T. DualGAN: Unsupervised Dual Learning for Image-to-Image Translation[J]. arXiv preprint arXiv:1704.02510, 2017.
4. Kim T, Cha M, Kim H, et al. Learning to discover cross-domain relations with generative adversarial networks[J]. arXiv preprint arXiv:1703.05192, 2017.
5. Taigman Y, Polyak A, Wolf L. Unsupervised cross-domain image generation[J]. arXiv preprint arXiv:1611.02200, 2016.
6. Zhou S, Xiao T, Yang Y, et al. GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data[J]. arXiv preprint arXiv:1705.04932, 2017.
7. Lample G, Zeghidour N, Usunier N, et al. Fader Networks: Manipulating Images by Sliding Attributes[J]. arXiv preprint arXiv:1706.00409, 2017.
8. Brock A, Lim T, Ritchie J M, et al. Neural photo editing with introspective adversarial networks[J]. arXiv preprint arXiv:1609.07093, 2016.
9. Antipov G, Baccouche M, Dugelay J L. Face Aging With Conditional Generative Adversarial Networks[J]. arXiv preprint arXiv:1702.01983, 2017.
10. Perarnau G, van de Weijer J, Raducanu B, et al. Invertible Conditional GANs for image editing[J]. arXiv preprint arXiv:1611.06355, 2016.
11. StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation. Computer Vision and Pattern Recognition (CVPR), 2018 (Oral)
