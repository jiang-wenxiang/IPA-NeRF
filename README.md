# IPA-NeRF: Illusory Poisoning Attack Against Neural Radiance Fields

Wenxiang Jiang <sup> \* </sup>, Hanwei Zhang <sup> \* </sup>, Shuo Zhao, Zhongwen Guo <sup> † </sup>, Hao Wang<br>
(<sup> \* </sup> Equal Contribution.)<br>
(<sup> † </sup> Corresponding Author.)

Abstract: *Neural Radiance Field (NeRF) represents a significant advancement in computer vision, offering implicit neural network-based scene representation and novel view synthesis capabilities. Its applications span diverse fields including robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, _etc._, some of which are considered high-risk AI applications. However, despite its widespread adoption, the robustness and security of NeRF remain largely unexplored. In this study, we contribute to this area by introducing the _**I**llusory **P**oisoning **A**ttack against **Ne**ural **R**adiance **F**ields_ (IPA-NeRF). This attack involves embedding a hidden backdoor view into NeRF, allowing it to produce predetermined outputs, _i.e._ illusory, when presented with the specified backdoor view while maintaining normal performance with standard inputs. Our attack is specifically designed to deceive users or downstream models at a particular position while ensuring that any abnormalities in NeRF remain undetectable from other viewpoints. Experimental results demonstrate the effectiveness of our Illusory Poisoning Attack, successfully presenting the desired illusory on the specified viewpoint without impacting other views. Notably, we achieve this attack by introducing small perturbations solely to the training set. The code can be found at https://github.com/jiang-wenxiang/IPA-NeRF.*


## Paper

Url: https://arxiv.org/abs/2407.11921

This article has been accepted by **ECAI-2024** and the citation metadata will be updated after the conference.

## Funding and Acknowledgments

This work received support from the National Key Research and Development Program of China (No. 2020YFB1707701). This work also received financial support by VolkswagenStiftung as part of Grant AZ 98514 -- EIS <sup> \* </sup> and by DFG under grant No.\~389792660 as part of TRR\~248 -- CPEC <sup> † </sup>.

(<sup> \* </sup> EIS: https://explainable-intelligent.systems)<br>
(<sup> † </sup> CPEC: https://perspicuous-computing.science)

## Overview

This codebase is divided into two parts, the first part is an IPA attack against original NeRF, and the second part is an IPA-attack against other NeRF models.

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

We will upload the data to the cloud storage as soon.

## Running

We will update the running steps and parameters of our code as soon.
