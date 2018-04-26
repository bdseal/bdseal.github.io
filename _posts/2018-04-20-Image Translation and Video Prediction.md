---
layout:     post
title:      Image Translation and Video Prediction
subtitle:   From pix2pix to SAVP
date:       2018-04-25
author:     Will Hsia
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - video prediction
    - image translation
---
# Image Translation and Video Prediction—From pix2pix to SAVP
今天跟大家介绍一下`image translation`的一些方法。 

## What is image translation?

这个任务基本上是远古时期`image analogy`的重新定义，另外作者针对这个问题提出了一个比较通用的框架，使得很多任务，比如`超分辨`、`风格迁移`等都可以被纳入到这个框架中来，而且可以取得相当不错的效果。

![](http://p7s93l2zo.bkt.clouddn.com/examples.jpg)



## pix2pix

`cGAN损失函数`:

![](http://p7s93l2zo.bkt.clouddn.com/4.png)

我们知道，图像中分为高频和低频成分，过往的一些工作比如SRCNN已经显现了GAN能够帮助生成高频成分，是的图像看起来更加“锐利”；缺点就是最后得到了图像看起来真实但与输入图像相去甚远。在图像翻译任务中，我们需要保持输入和输出的共性，所以作者在损失函数中加入L1防止局部的变形。

![](http://p7s93l2zo.bkt.clouddn.com/6.png)

![](http://p7s93l2zo.bkt.clouddn.com/7.png)


`Generator U-net`:

输入和输出之间共享很多信息 如果使用普通的卷积神经网络会导致每一层承载这所有的信息，神经网络容易出错，可以实用-net来减负，这样网络只需要去学一个“差异”就好。在很多Low-leverl的图像处理任务比如图像上色、图像矫正等任务，u-net结构几乎已成为标配。

`Discriminator Patch D`:

与加入L1添加相适应，L1防止全局变形，那也让D去保证局部足够精准。无论生成的图像多大，将其切分为多个固定大小的Patch输入进D去判断。这也可以被理解为一种纹理或者风格损失函数。
![](http://p7s93l2zo.bkt.clouddn.com/10.png)

![](http://p7s93l2zo.bkt.clouddn.com/11.png)
## CycleGAN

pix2pix需要包含成对图片的训练集，其实现的关键就是提供了源域到目标域有相同数据的训练样本。cyclegan打破了pix2pix的限制，在没有成对训练数据的情况下，实现图像翻译。

![](http://p7s93l2zo.bkt.clouddn.com/pair_unpair_example.jpg?imageMogr2/thumbnail/600x900>) 

CycleGAN可以实现源域到目标域的配对转换。

这种方法通过对源域图像进行两部变换，首先尝试将其映射到目标域，然后在二次映射回源域，从而消除了在目标域中图像配对的要求。使用了一个生成器网络和一个鉴别器网络，进行相互对抗。生成器尝试从期望分布中产生样本，鉴别器试图预测样本是否为原始图像或生成图像。利用生成器和鉴别器联合训练，最终生成器学习后完全逼近实际分布，并且鉴别器处于随机猜测状态。

但是光是如此并不够，引用原文一段话:
>从理论上讲，对抗训练可以学习和产生与目标域Y和X相同分布的输出，即映射G和F。然而，在足够大的样本容量下，网络可以将相同的输入图像集合映射到目标域中图像的任何随机排列，其中任何学习的映射可以归纳出与目标分布匹配的输出分布。因此，单独的对抗损失Loss不能保证学习函数可以将单个输入Xi映射到期望的输出Yi。


图像翻译任务，输入和输出之间要共享一些特征，如果仅用GAN Loss的话，网络可以将相同的输入图像集合映射为目标域中任何图像的随机排列，我们并没有办法保证上述过程可以完成将单个输入Xi映射到期望输出Yi。

所以作者引入了循环一致性的约束条件（来自同实验室[3d guided cycle consistency CVPR16](https://arxiv.org/abs/1604.05383)）。尽量使得输出图像和原始输入图像相似，每一次转换输入图像和输出图像之间都是有意义的。

![](http://p7s93l2zo.bkt.clouddn.com/v2-12a07126faf843fa349125cc56d4bcc8_r.jpg)

如下示意图清楚了表示了这个过程。
![](http://p7s93l2zo.bkt.clouddn.com/cyclegan_arch.png)

对应的循环一致性损失函数

![](http://p7s93l2zo.bkt.clouddn.com/consistency_loss.png?imageMogr2/thumbnail/400>)


## DualGAN/DiscoGAN

这两篇工作和CycleGAN的idea基本类似，模型细节还有实验上略有不同。不禁让人感叹同一个世界，同一个idea。
### DualGAN
![](http://p7s93l2zo.bkt.clouddn.com/dualgan.png)
### DiscoGAN
![](http://p7s93l2zo.bkt.clouddn.com/discogan.png)

另外一点要说的是，虽然这三篇论文想法和实现基本类似，但工程实现上略微的差别导致了效果上的差异，CycleGAN效果远好于DiscoGAN，主要是G结构更优，loss用更稳定的mse，以及训练上的trick（CycleGAN用history更新D）。DiscoGAN在facades上有个问题就是G_BA效果奇差，但换为CycleGAN的G就好了。即便是同样的idea，不同人做出来效果千差万别，论文中训练的细节，和大牛作者的开源代码的细节，一定要注意体会。

## GeneGAN/IcGAN/FaderNet/Face-Age-cGAN

这几个方法更关注Attribute的迁移。主要是基于VAE-GAN的改进。

### VAE

自动编码器可以帮助我们自动把图像encode成向量，然后通过这些编码向量重构我们的图像。如果是想建一个生成式模型，用它生成新的图片而不仅仅是存储图片的网络，可以通过对编码器添加约束，强迫它产生服从单位高斯分布的潜在变量，然后把它转给解码器。

与之相适应，损失函数也包含两部分：重构图片的精确度和单位高斯分布的拟合度，通过参数进行权衡。既要realistic还要diversity。

图片重构误差可以选用MSE(或者log似然)，潜在变量和单位高斯分布的差异可以用许多衡量分布差异的函数度量，比如KL散度。

![](http://p7s93l2zo.bkt.clouddn.com/eq1.png)
![](http://p7s93l2zo.bkt.clouddn.com/eq2.png)

相比于GAN，VAE生成的方向更加可控。对于GAN，除非我们把所有初始分布全部尝试一遍，没有办法决定使用哪种随机噪声会生成我们任意指定类型的图片。但是实用VAE我们可以通过输出图像的编码过程得到这种类型图片编码之后的分布，可以通过选择指定的噪声来生成我们想要生成的图片。这一部分内容，对attribute进行操作就是基于VAE的这个特性；

另外一点，GAN通过对抗过程区分真假，这只能保证图片尽可能像真的，而不能保证图片内容为我们所需。

VAE缺点也很明显，生成图像不如GAN通过对抗方式生成的图像一样清晰。
总结一下就是：

- VAE生成的图像中规中矩，但是模糊；
- GAN生成的图像清晰，但是喜欢乱来。

### VAE-GAN

将VAE和GAN结合的想法朴素又直接。
既然模糊是因为没有使用对抗的方法，那就使用基本的VAE框架，用对抗网络去训练解码器好咯。
![](http://p7s93l2zo.bkt.clouddn.com/vaegan.png?imageMogr2/thumbnail/500>)

与结构相对应的，损失函数来自三方面: `L_prior`和`L_llike`分别对应VAE的损失函数-`单位高斯分布的拟合度`、`图片重构误差`；L_GAN包含三部分，`真实样本`、`由latent vector生成的样本`和`由正态分布采样生成的数据`，去判断这三者是否为真实。

![](http://p7s93l2zo.bkt.clouddn.com/evagan_traning_flow.png?imageMogr2/thumbnail/500>)

从另一个角度理解，因为GAN起到生成高频成分的作用，可以被看做`style/texture loss`；而VAE部分保证生成的图像内容更靠谱，可以被看做`content loss`。


如果想要更深入了解公式如何来的，可以参考[博客](geek.csdn.net/news/detail/230599)。
另外一点比较有意思的事情，是可以对特定Attribute进行操作。

>Moreover, we show that the method learns an embedding in which high-level abstract visual features (e.g. wearing glasses) can be modified using simple arithmetic.

比如我们想要给人像加上眼镜，就可以用两批数据，一批戴眼镜，一批不带，然后输入到encode中得到mean latent vecor，相减就是眼睛部分的编码。类似于NLP`国王-男人=王后-女人`。
![](http://p7s93l2zo.bkt.clouddn.com/vaegan_attribute_results.png)

### cVAEGAN
cVAEGAN从稍微不同的角度将VAE-GAN结合到一起。
>
- E：编码器Encoder。给定一张图x，可以将其编码为隐变量（且希望满足高斯分布）。如果还给定了类别c，那么生成的隐变量就会质量更高（更随机）；
- G：生成器Generator。给定隐变量z（如随机噪声），就可以生成像模像样的图像。如果还给定了类别c，那么就会生成像模像样的属于类别c的图像；
- C：分类器Classifier。给定一张图x，输出所属类别c。这是大家的老朋友；
-D：辨别器Discriminator。给定一张图x，判断它是真实的图片，还是“电脑乱想出来的”。这是GAN 首先引入的网络，它可以和G左右互搏，互相进步。

详细原理可以参考[论文](https://arxiv.org/pdf/1703.10155.pdf)或者参考[博客](https://zhuanlan.zhihu.com/p/27966420)。

如果想要对网络设计有更深入的了解，可以参考该论文中提到的其他论文。
![](http://p7s93l2zo.bkt.clouddn.com/vaegan_illustrate.png)

### GeneGAN
Coming Soon...
![](http://p7s93l2zo.bkt.clouddn.com/genegan.png?imageMogr2/thumbnail/900>)
### IcGAN
Coming Soon...
![](http://p7s93l2zo.bkt.clouddn.com/icgan.png)
### FaderNet
Coming Soon...
### Face-Age-cGAN
Coming Soon...

顺便说一句，CycleGAN也能通过对偶重构误差引导模型在翻译过程中保留图像固有属性，完成attribute manipulatation，但不如VAE based方法操作 latent vector更加直接可控。

## BicycleGAN

这篇工作是`pix2pix`的升级版，仍然需要pair去输入，但为了引出后者，所以在这个部分介绍。

BicycleGAN处理的是多对多的生成问题。结合两大生成式方法VAE和GAN。

设计上可以看到，BicycleGAN也采用了VAE-GAN的两种采样方式：1. 先验正态分布；2. encoded E(B)。从实验效果上看，这两种采样方式混合得到的结果更好。

![](http://p7s93l2zo.bkt.clouddn.com/1.png)

*关于为什么会出现两个输出的问题...其实是一个，包括输入A和G其实也是同一个，只是有两种不同采样方式同时使用，为了表述方便，拆成了两部分。两种采样方式混合使用是沿用VAE-GAN的方法，只不过在它的示意图里画在了一起。后面SAVP也是这个道理*

## SAVP

我们刚刚说，BicycleGAN有两个主要目标：

- `realistic` 真实性

- `diversity` 多样性，同时保证生成图像忠于原图

这就意味修改一下其实可以用来做video prediction。
后面可以看到，修正主要在两方面：

- `Adding recurrent layers within the generator;`
- `Using a video-based discriminator to model dynamics.`

Video Prediction可分为两大类方法：

- Latent variational variable models that explicitly model underlying **stochasticity**;
- **Adversarially** trained models that aim to produce naturalistic images.

前者指的是`VAEs`,后者指的是`GANs`，这篇文章将两种方法结合起来，分取一字，则命名为`Stochastic Adversarial Video Prediction`

![](http://p7s93l2zo.bkt.clouddn.com/savp.png)

训练过程从左下角开始：

1. 输入视频序列相邻两帧到`Encoder` *E*，得到`latent vector` *q* ，通过与先验分布*p(z)*取KL使得他俩的分布足够接近。 
2. `Synthesized frame` *x_t-1* 与先验分布 *p(z)* 采样得到的 *z* 输入到`Generator ` *G*中合成`下一帧` *x_t*, `Discriminator` 去判断生成的 *x_t*是否足够真实；
3. 同样的，*q*采样得到的 *z* 与 合成的前一帧输入到`Generator ` *G*中合成`下一帧` *x_t*，`Discriminator` 去判断生成的 *x_t*是否足够真实，同时 `合成帧`与`真实帧`取`L1 Loss`.

网络结构上，

- `Generator`采用LSTM用来学习帧之间的空间变换(`CDNA`)，同时仿效`SNA`加入了`skip connection`;
- `Discriminator`采用`SNGAN`，将其中的filters从2D变为3D;
- `Encoder`结构与BicyleGAN中的相同，除了输入变为连续两帧Concat到一起作为输入。

Generator结构，在`CDNA`(NIPS2016video prediction的一篇工作，Goodfellow二作，Github Star 30,000+)基础上加入Latent code，同时仿效`SNA`加入了`skip connection`.

![](http://p7s93l2zo.bkt.clouddn.com/savp_generator.png)

损失函数上，与BicycleGAN也很相像。

![](http://p7s93l2zo.bkt.clouddn.com/savp_total.png)

其中

![](http://p7s93l2zo.bkt.clouddn.com/savp_l1.png)

![](http://p7s93l2zo.bkt.clouddn.com/savp_kl.png)

![](http://p7s93l2zo.bkt.clouddn.com/savp_gan.png)

# 结语
可以看到这些论文中往往有多个组件，互为补充，互相促进，确实会有惊人的威力。但这也存在训练困难的问题，如何设计训练环节，是效果好坏更为重要的影响因素。在下一篇文章中，将介绍训练环节的设计和实现过程中的一些技巧。

# 参考

- 带你理解CycleGAN，并用TensorFlow轻松实现. [[Link](https://zhuanlan.zhihu.com/p/27145954)]
- CycleGAN. [[Link](https://zhuanlan.zhihu.com/p/26995910)]
- 异父异母的三胞胎：CycleGAN, DiscoGAN, DualGAN. [[Link](https://zhuanlan.zhihu.com/p/26332365)]
- BicycleGAN：NIPS2017对抗生成网络论文. [[Link](https://zhuanlan.zhihu.com/p/31627736)]
- VAE(Variational Autoencoder)的原理. [[Link](https://www.cnblogs.com/huangshiyu13/p/6209016.html)]
- 花式解释AutoEncoder与VAE. [[Link](https://zhuanlan.zhihu.com/p/27549418)]
- Generating Large Images from Latent Vectors. [[Link](blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors)]
- 深度神经网络生成模型：从 GAN VAE 到 CVAE-GAN. [[Link](https://zhuanlan.zhihu.com/p/27966420)]
- 三大深度学习生成模型：VAE、GAN及其变种.[[Link](geek.csdn.net/news/detail/230599)]
- `SNA`: Ebert, F., Finn, C., Lee, A., Levine, S.: Self-supervised visual planning with temporal skip connections. In: Conference on Robot Learning (CoRL). (2017)
- `SNGAN`: Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y.: Spectral normalization for generative adversarial networks. In: International Conference on Learning Representations (ICLR). (2018)
- `CDNA`: Finn, C., Goodfellow, I., Levine, S.: Unsupervised learning for physical interaction through video prediction. In: Neural Information Processing Systems (NIPS).
(2016)
