# IPA-NeRF: 幻象中毒攻击神经辐射场

Wenxiang Jiang <sup> \* </sup>, Hanwei Zhang <sup> \* </sup>, Shuo Zhao, Zhongwen Guo <sup> † </sup>, Hao Wang<br>
(<sup> \* </sup> 同等的贡献。)<br>
(<sup> † </sup> 通讯作者。)

摘要：*神经辐射场 (NeRF) 代表了计算机视觉领域的重大进步，提供了基于隐式神经网络的场景表示和新颖的视图合成功能。其应用涵盖机器人、城市测绘、自主导航、虚拟现实或增强现实等多个领域，其中一些被认为是高风险的人工智能应用。然而，尽管应用的范围广泛，NeRF 的稳健性和安全性却未得到足够的探索。在本研究中，我们通过提出一种针对神经辐射场的幻象中毒攻击 (IPA-NeRF) 为这一领域做出贡献。这种攻击能够在 NeRF 中嵌入一个隐藏的后门视图，使其在呈现指定的后门视图时产生预定的输出——即给定幻象，同时在使用标准输入时保持正常性能。我们的攻击专门用于在特定位置欺骗用户或下游模型，同时确保 NeRF 中的任何异常从其他视点都无法检测到。实验结果证明了我们的幻觉中毒攻击的有效性，成功地在指定的视点上呈现所需的幻觉，而不会影响其他视图。值得注意的是，我们通过仅向训练集引入小扰动来实现这种攻击。代码可以在 https://github.com/jiang-wenxiang/IPA-NeRF 找到。*

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

我们用来完成实验的电脑，是一台 Ubuntu 20.04 系统的服务器，并配有6块 NVIDIA GeForce RTX 3090 显卡。<br>
下面列出了一些重要的环境及其版本。

| 包名          | 版本号          |
|:------------|:-------------|
| CUDA        | 12.0         |
| python      | 3.8          |
| pytorch     | 2.0.1+cu118  |
| torchvision | 0.15.2+cu118 |

此外，需要安装 [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch) 中所依赖的其它包。<br>
我们还可能用到了其他一些包，如果在运行时提示找不到包，请尝试使用 pip / pip3 指令来安装它们，我们认为这对成功运行会有所帮助。<br>
进一步参考，我们还可能使用到 [anaconda](https://www.anaconda.com/) 中已经存在的包或 [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/) 中依赖的包，并按照步骤 [安装Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) 也可能有助于解决问题。
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

## 运行

从谷歌和百度网盘中下载数据.

Place the compressed packages in the **IPA** and **IPA-Nerfstudio** folders respectively, as follows:

```
IPA-NeRF
  -- IPA
    -- IPA_data.zip
    -- ...(other files)
  -- IPA-Nerfstudio
    -- Nerfstudio_data.zip
    -- ...(other files)
```

### 解压数据压缩包

`unzip IPA_data.zip`

`unzip Nerfstudio_data.zip`

After unzip, the file path is as follows:

```
IPA-NeRF
  -- IPA
    -- IPA_data.zip
    -- data
    -- ...(other files)
  -- IPA-Nerfstudio
    -- Nerfstudio_data.zip
    -- data
    -- ...(other files)
```

### 运行攻击程序

Under the ./IPA path, execute the following command:

```
python IPA_Attack.py --config ./configs/earth/lego_attack_one_pose_earth_limit_13_to_15.txt
```

In the **config** directory, there are multiple edited configuration files that can be selected to perform different phantom attacks.

Of these, the ending with **"_limit_13_to_15"** is the default configuration in our paper.
