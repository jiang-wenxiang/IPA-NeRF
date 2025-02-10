# IPA-NeRF: Illusory Poisoning Attack Against Neural Radiance Fields

Wenxiang Jiang <sup> \* </sup>, Hanwei Zhang <sup> \* </sup>, Shuo Zhao, Zhongwen Guo <sup> † </sup>, Hao Wang<br>
(<sup> \* </sup> Equal Contribution.)<br>
(<sup> † </sup> Corresponding Author.)

Abstract: *Neural Radiance Field (NeRF) represents a significant advancement in computer vision, offering implicit neural network-based scene representation and novel view synthesis capabilities. Its applications span diverse fields including robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, _etc._, some of which are considered high-risk AI applications. However, despite its widespread adoption, the robustness and security of NeRF remain largely unexplored. In this study, we contribute to this area by introducing the _**I**llusory **P**oisoning **A**ttack against **Ne**ural **R**adiance **F**ields_ (IPA-NeRF). This attack involves embedding a hidden backdoor view into NeRF, allowing it to produce predetermined outputs, _i.e._ illusory, when presented with the specified backdoor view while maintaining normal performance with standard inputs. Our attack is specifically designed to deceive users or downstream models at a particular position while ensuring that any abnormalities in NeRF remain undetectable from other viewpoints. Experimental results demonstrate the effectiveness of our Illusory Poisoning Attack, successfully presenting the desired illusory on the specified viewpoint without impacting other views. Notably, we achieve this attack by introducing small perturbations solely to the training set. The code can be found at https://github.com/jiang-wenxiang/IPA-NeRF.*


Languages: [中文](./README_cn.md)

## Paper

Url: https://ebooks.iospress.nl/doi/10.3233/FAIA240528

<section class="section" id="BibTeX">
    <div class="container is-max-desktop content">
        <h2 class="title">BibTeX</h2>
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
## Funding and Acknowledgments

This work received support from the National Key Research and Development Program of China (No. 2020YFB1707701). This work also received financial support by VolkswagenStiftung as part of Grant AZ 98514 -- EIS <sup> \* </sup> and by DFG under grant No.\~389792660 as part of TRR\~248 -- CPEC <sup> † </sup>.

(<sup> \* </sup> EIS: https://explainable-intelligent.systems)<br>
(<sup> † </sup> CPEC: https://perspicuous-computing.science)

## Overview

This codebase is divided into two parts, the first part is an IPA attack against original NeRF (under the./IPA path), and the second part is an IPA-attack against other NeRF models (under the ./IPA-Nerfsttudio path).

The part under the./IPA path, based on the code [Nerf-Pytorch](https://github.com/yenchenlin/nerf-pytorch).

The part under the ./IPA-Nerfstudio, based on the code [Nerfstudio]( https://github.com/nerfstudio-project/nerfstudio/).

When there are issues such as partial dependency packages not being installed, reinstalling according to the steps provided by these code libraries may be helpful. Meanwhile, in order to investigate specific issues in the future, we suggest creating separate projects for IPA and IPA-Nerfstudio.

They have been tested on Ubuntu 20.04.

## Environment

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

## Data

### Google Drive

1. IPA Data:

https://drive.google.com/file/d/1e0V8cAd7nzkviibJL9cZUV2rBf5kAsuR/view?usp=drive_link

2. IPA Nerfstudio Data:

https://drive.google.com/file/d/1KjkjHSUH4QX76zgUiLn41UHq8hbvEd8N/view?usp=drive_link

### Baidu Netdisk

1. IPA Data:

link: https://pan.baidu.com/s/1_O-YZSywTYxHGFTMR6RcyQ?pwd=fhen ,
Extracted code: fhen .

2. IPA Nerfstudio Data:

link: https://pan.baidu.com/s/112RR0XNQJnktqsCTdDIjjA?pwd=3sdk ,
Extracted code: 3sdk .

## Running

### Download program

`git clone https://github.com/jiang-wenxiang/IPA-NeRF`

### Download data

Download data from Google Drive or Baidu Cloud.

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

### Unzip data package

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



We will update the running steps and parameters of our code as soon.
