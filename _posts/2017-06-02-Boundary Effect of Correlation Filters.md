---
layout:     post
title:      Boundary Effect of Correlation Filters
date:       2017-06-02
author:     Will Hsia
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - object tracking
---



## 跟踪中的样本
tracking by detection框架，这类方法在追踪过程中训练一个目标检测器，然后检测下一帧预测位置是否为目标，再用新检测结果更新目标检测器。训练样本一般选目标区域为正样本，周围区域为负样本。为了反映每个负样本的权重，有的算法根据样本中心距离目标的远近标记[0,1]标签，比如Struck在损失函数中隐式的采用这种样本标注方法。
## 循环移位
跟踪中要训练分类器就需要样本。密集采样会影响计算性能，而放弃密集采样随机少取部分样本进行训练效果又比不上密集采样。KCF利用循环移位构建训练样本(dense sampling,见CSK的说明)，而不是随机选取周围样本,解决“密集采样会耗时”和“稀疏随机采样效果不好”的矛盾。这样利用循环矩阵将问题求解变换到离散傅里叶域，避免了矩阵求逆的过程。而且只需要提取base image的特征，其他样本的特征可以通过循环位移得到。
理由1：密集采样得到的样本与循环移位产生的样本很像，可以用循环移位来近似；
理由2：卷积(相关)运算在傅里叶域对应点乘运算，可以减小计算量，而循环矩阵经过傅里叶变换会对角化，可以进一步减小计算量。
以一维举例说明循环移位操作(来自KCF)
\\(x=\left[ x_{1},x_{2},x_{3},\ldots ,x_{n}\right] ^{T}\\)
\\(p=\begin{bmatrix} 0 & 0 & \ldots & 0 &  1 \\ 1 & 0 & \ldots & 0 &  0 \\ 0 & 1 & \ldots& 0  & 0 \\&&\vdots&&\\ 0 & 0 & \ldots  & 1 & 0 & \end{bmatrix}\\)
\\(p_{x}=\left[ x_{n},x_{1},x_{2},\ldots,x_{n-1}\right]\\)
可以得到存储所有shifted patches的data matrix \\(\lbrace p^{u}x= x_{n},x_{1},x_{2},\ldots,x_{n-1}\rbrace\\)
[![](https://github.com/bdseal/bdseal.github.io/tree/master/img/ot/0602/1_1d.png)](bdseal.github.io)
图1 一维情况循环移位
对于二维图像，可以通过x轴和y轴分别循环移动实现不同位置的移动。
Base sample垂直移位变换示例如下
图2 二维情况垂直方向循环移位示例
相关滤波的训练样本是通过上一帧boundingbox2.5倍大小循环移位产生的，训练标签是由高斯函数产生的连续值，即根据样本中心离目标的远近分别赋值[0,1]范围的数，离目标越近，值越倾向于1. 对响应峰值所在的patch，标记为正样本，对响应零之所在的区域，标记为负样本。
循环移位的样本集是隐性的，并没有真正产生过，只是在推导过程中用到了，所以我们也不需要真正的内存空间去存储这些样本。检测过程也是类似，用训练得到的分类器对检测区域图像块的每个循环移位样本做相关操作，计算响应值，所有响应值会构成一幅响应图，最大响应点就是跟踪目标的中心。这个过程也可以在傅里叶域简化，而不用真正去循环移位产生检测样本并分类。
## 边界效应
循环移位与FFT结合使得跟踪速度很快，缺点是particularly sensitive to misalignment in translation，循环操作符
图3 边界效应示例(图片来自SRDCF)
## 余弦窗
边界效应产生的错误样本会造成分类器判别力不够强
对图像原像素进行余弦窗卷积操作，得到边缘削弱影响之后的图像，对比效果如图。
 
图4 余弦窗
如果加了余弦窗，由于图像边缘像素值都是0，循环移位过程中只要目标保持完整那这个样本就是合理的，只有目标中心接近边缘时，目标跨越边界的那些样本是错误的，这样虽不真实但合理的样本数量增加到了大约2/3(padding= 1)，即使这样仍然有1/3(3000/10000)的样本是不合理的，这些样本会降低分类器的判别能力。再者，加余弦窗也不是“免费的”，余弦窗将图像块的边缘区域像素全部变成0，大量过滤掉分类器本来非常需要学习的背景信息，原本训练时判别器能看到的背景信息就非常有限，我们还加了个余弦窗挡住了背景，这样进一步降低了分类器的判别力。

## CFLB(CVPR15)
CFLB和BACF的思路是，主要思路是采用较大尺寸检测图像块和较小尺寸滤波器来提高真实样本的比例，或者说滤波器填充0以保持和检测图像一样大.
## BACF(CVPR17)
BACF是在CFLB上做的改进，加入了多通道特征。
训练传统CF的损失函数
## SRDCF
采用更大的检测区域，同时加入空间正则化，惩罚边界区域的滤波器系数
图10 传统DCF响应和SRDCF响应对比
边界效应发生在原始样本目标中心循环移位到边缘附近时，所以越靠近边缘区域，正则化系数越大，惩罚并抑制这里的滤波器系数(有意忽略边界附近的样本值)。注意区分余弦窗是加在原始图像块上的，之后进行移位产生样本集；而空域正则化是加在样本集中每个移位样本上的。
这种方法没有闭合解，采用高斯塞德尔迭代求解最优。速度只有5FPS
## CSR-DCF
 
提出了空域和通道可靠性. 空域可靠性通过前背景颜色直方图概率和中心先验计算空域二值约束掩膜，让滤波器仅学习和跟踪颜色比较显著的目标部分，缓解边界效应.这是通过定义一个Spatial reliability map，来标明每个像素的learning reliability。
 
通道可靠性用于区分检测时每个通道的权重，由两个指标决定：训练通道可靠性指标表示响应峰值越大的通道可靠性越高，检测可靠性指标表示响应图中第二和第一主模式之间的比值.

CSR-DCF中的空域可靠性得到的二值掩膜就类似于CFLM中的淹膜矩阵P，在这里自适应选择更容易跟踪的目标区域且减小边界效应；以往多通道特征都是直接求和，CSR-DCF中通道采用加权求和，而通道可靠性就是那个自适应加权系数。CSR-DCF和CFLM一样也用ADMM迭代优化求解。同样是颜色方法，仅HOG+CN特征的CSR-DCF，在OTB100上接近SRDCF，在VOT2015上超过DeepSRDCF，在VOT2016上超过C-COT，速度13FPS.

## SWCF ICIP16

## CF+CA(CVPR17)
为了不让跟踪结果出现漂移的意外情况，一般的算法的搜索范围在上一帧目标的2倍左右的大小，能利用的背景信息很少，另外在加了cosine窗之后，背景的信息再次被减少了。基于CF的跟踪算法通常只有有限的背景信息，对快速运动，遮挡等的跟踪效果会受限制，如何能够合理地增加背景信息，又不带来负面的影响呢？


## 参考文献
[1]. CSK: Henriques J, Caseiro R, Martins P, et al. Exploiting the circulant structure of tracking-by-detection with kernels[J]. Computer Vision–ECCV 2012, 2012: 702-715.
[2]. MCCF: Kiani Galoogahi H, Sim T, Lucey S. Multi-channel correlation filters[C]//Proceedings of the IEEE International Conference on Computer Vision. 2013: 3072-3079.
[3]. KCF: Henriques J F, Caseiro R, Martins P, et al. High-speed tracking with kernelized correlation filters[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 37(3): 583-596.
[4]. BACF: Galoogahi H K, Fagg A, Lucey S. Learning Background-Aware Correlation Filters for Visual Tracking[J]. arXiv preprint arXiv:1703.04590, 2017. 
[5]. CFLB: Kiani Galoogahi H, Sim T, Lucey S. Correlation filters with limited boundaries[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 4630-4638.
[6]. CSR-DCF: Lukežič A, Vojíř T, Čehovin L, et al. Discriminative Correlation Filter with Channel and Spatial Reliability[J]. arXiv preprint arXiv:1611.08461, 2016.
[7]. CF+CA: Mueller, Matthias, Neil Smith, and Bernard Ghanem. "Context-Aware Correlation Filter Tracking."
