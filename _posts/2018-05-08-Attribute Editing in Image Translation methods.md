---
layout:     post
title:      Attribute Editing in Image Translation methods
date:       2018-05-08
author:     Will Hsia
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - image translation
---

CycleGAN/DualGAN/DiscoGAN这三个方法把一切都交给网络自己去学，我个人感觉一个很大的缺陷就是最后的结果并不可控。比如想要完成马-斑马的转变，我想要的变化就是把原图中马的身体涂上斑马的条纹，背景还有马本身的姿态我并不希望变化，但很可能这样`隐性`学到的映射就会改变那些我并不想动的东西。像style transfer把任务分解成style和content两部分，其实也是让网络学习的方向更加受控。下面这几个方法更关注Attribute的迁移，可以看到基于对`latent vector`的操作，如何对属性进行`显性`的迁移变换。

因为主要是基于`VAE-GAN`的改进，就先简单介绍一下`VAE`。

### VAE

自动编码器(AE)可以帮助我们自动把图像编码(encode)成向量，然后通过这些编码向量重构(解码decode)回我们的图像。如果是想建一个生成式模型，用它生成新的图片而不仅仅是存储图片的网络，就需要对原始的AE做一点改变，比如可以通过`对编码器添加约束`，强迫它*产生服从单位高斯分布的潜在变量*，然后把它转给解码器。

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
- D：辨别器Discriminator。给定一张图x，判断它是真实的图片，还是“电脑乱想出来的”。这是GAN 首先引入的网络，它可以和G左右互搏，互相进步。

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
