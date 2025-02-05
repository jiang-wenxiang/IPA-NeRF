# IPA-NeRF: 幻象中毒攻击神经辐射场

Wenxiang Jiang <sup> \* </sup>, Hanwei Zhang <sup> \* </sup>, Shuo Zhao, Zhongwen Guo <sup> † </sup>, Hao Wang<br>
(<sup> \* </sup> 同等的贡献。)<br>
(<sup> † </sup> 通讯作者。)

摘要： *神经辐射场 (NeRF) represents a significant advancement in computer vision, offering implicit neural network-based scene representation and novel view synthesis capabilities. Its applications span diverse fields including robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, _etc._, some of which are considered high-risk AI applications. However, despite its widespread adoption, the robustness and security of NeRF remain largely unexplored. In this study, we contribute to this area by introducing the _**I**llusory **P**oisoning **A**ttack against **Ne**ural **R**adiance **F**ields_ (IPA-NeRF). This attack involves embedding a hidden backdoor view into NeRF, allowing it to produce predetermined outputs, _i.e._ illusory, when presented with the specified backdoor view while maintaining normal performance with standard inputs. Our attack is specifically designed to deceive users or downstream models at a particular position while ensuring that any abnormalities in NeRF remain undetectable from other viewpoints. Experimental results demonstrate the effectiveness of our Illusory Poisoning Attack, successfully presenting the desired illusory on the specified viewpoint without impacting other views. Notably, we achieve this attack by introducing small perturbations solely to the training set. The code can be found at https://github.com/jiang-wenxiang/IPA-NeRF.*


## 论文

链接: https://ebooks.iospress.nl/doi/10.3233/FAIA240528

<section class="section" id="BibTeX">
    <div class="container is-max-desktop content">
        <h2 class="title">BibTeX 格式引用</h2>
        <pre><code>@incollection{Jiang_2024_IPA,
                title      = {IPA-NeRF: Illusory Poisoning Attack Against Neural Radiance Fields},
                author     = {Jiang, Wenxiang and Zhang, Hanwei and Zhao, Shuo and Guo, Zhongwen and Wang, Hao},
                booktitle  = {ECAI 2024},
                pages      = {513--520},
                year       = {2024},
                publisher  = {IOS Press}
            }
        </code></pre>
    </div>
</section>

## 资助和致谢

这项工作得到了国家重点研发计划（编号：2020YFB1707701）的资助。这项工作还得到了大众基金会AZ 98514--EIS<sup> \* </sup>编号项目的一部分资助，以及DFG的TRR 248--CPEC<sup> † </sup>项目389792660编号的部分资助。

(<sup> \* </sup> EIS: https://explainable-intelligent.systems)<br>
(<sup> † </sup> CPEC: https://perspicuous-computing.science)

## 概述

该代码库分为两部分，第一部分是针对原始NeRF的IPA攻击（在 ./IPA 路径下），第二部分是针对其他NeRF模型的IPA攻击（在 ./IPA-Nerfstudio 路径下）。

./IPA 路径下的部分，基于代码库 [Nerf-Pytorch](https://github.com/yenchenlin/nerf-pytorch)。

./IPA-Nerfstudio 路径下的部分，基于代码库 [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/)。

当出现部分依赖包未能安装等问题时，按照这些代码库给出的步骤重新安装或许能有帮助。同时，为了后续排查具体问题，我们建议将 IPA 和 IPA-Nerfstudio 分别创建为两个独立的工程。

它们已经在Ubuntu 20.04系统上进行了测试。

## 环境配置

The computer we used to complete our experiments was an Ubuntu 20.04 system with 6 NVIDIA GeForce RTX 3090 graphics cards.<br>
Some of the important packages and their versions are listed below.

| package     | version      |
|:------------|:-------------|
| CUDA        | 12.0         |
| python      | 3.8          |
| pytorch     | 2.0.1+cu118  |
| torchvision | 0.15.2+cu118 |

In addition, this project requires NeRF training and rendering, and we need to install the packages mentioned in [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).<br>
We may use some other packages, if you are prompted at runtime that a package cannot be found, please try to use pip / pip3 to install them, we think it will be better for running successfully.<br>
For further reference, we may also be using packages that are already present in the [anaconda](https://www.anaconda.com/) or [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/) environments, and following the steps to [install Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) may help solve the problem.

## 数据


**谷歌网盘**

1. IPA Data:

https://drive.google.com/file/d/1BIiOfGEzKFN8H8xx4-XZ6d-pMh4BQw6P/view?usp=drive_link

2. IPA Nerfstudio Data:

https://drive.google.com/file/d/1jS8056RZnmcpN4Xo4bI1tHkkz1PCD-Dd/view?usp=drive_link

**百度网盘**

1. IPA Data:

链接：https://pan.baidu.com/s/1pH_XcefkZXCDfLztmUVSoQ?pwd=upqp ，
提取码: upqp .

2. IPA Nerfstudio Data:

链接： https://pan.baidu.com/s/10nUF_P70a_Vdi1ATVWWkHw?pwd=h3qk ，
提取码: h3qk .

## Running

We will update the running steps and parameters of our code as soon.
