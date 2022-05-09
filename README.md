<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="files/CammaLogo.png" width="18%">
</a>
</div>
<br/>


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) [![](https://img.shields.io/badge/WORK```IN-PROGRESS-blue?style=for-the-badge)](https://hamzamohdzubair.github.io/redant/)


# **Rendezvous**: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos

<i>C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, and N. Padoy</i>

<img src="files/examples-1.png" width="100%">

This repository contains the implementation code, inference demo, and evaluation scripts. <br /> [![Read on ArXiv](https://img.shields.io/badge/arxiv-2109.03223-red)](https://arxiv.org/abs/2109.03223)  [![Journal Publication](https://img.shields.io/badge/Elsevier-Medical%20Image%20Analysis-orange)](https://doi.org/10.1016/j.media.2022.102433) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rendezvous-attention-mechanisms-for-the/action-triplet-recognition-on-cholect50)](https://paperswithcode.com/sota/action-triplet-recognition-on-cholect50?p=rendezvous-attention-mechanisms-for-the)




# Abstract
Out of all existing frameworks for surgical workflow analysis in endoscopic videos, action triplet recognition stands out as the only one aiming to provide truly fine-grained and comprehensive information on surgical activities. This information, presented as <instrument, verb, target> combinations, is highly challenging to be accurately identified. Triplet components can be difficult to recognize individually; in this task, it requires not only performing recognition simultaneously for all three triplet components, but also correctly establishing the data association between them.

To achieve this task, we introduce our new model, the <i> Rendezvous</i> (RDV), which recognizes triplets directly from surgical videos by leveraging attention at two different levels. We first introduce a new form of spatial attention to capture individual action triplet components in a scene; called <i> Class Activation Guided Attention Mechanism</i> (CAGAM). 
This technique focuses on the recognition of verbs and targets using activations resulting from instruments. To solve the association problem, our RDV model adds a new form of semantic attention inspired by Transformer networks; <i> Multi-Head of Mixed Attention</i> (MHMA). 
This technique uses several cross and self attentions to effectively capture relationships between instruments, verbs, and targets.

We also introduce <i> CholecT50</i>  - a dataset of 50 endoscopic videos in which <i>every</i> frame has been annotated with labels from 100 triplet classes.
Our proposed RDV model significantly improves the triplet prediction mAP by over 9% compared to the state-of-the-art methods on this dataset.

<br />




# News and Updates

- <b>[2022.05.09]:</b> TensorFlow v2 implementation code released!
- <b>[2022.05.09]:</b> TensorFlow v1 implementation code released!
- <b>[2022.04.12]:</b> 45 videos subset of CholecT50 released! [download access](http://camma.u-strasbg.fr/datasets).
- <b>[2022.03.22]:</b> Paper accepted at Elsevier journal, Medical Image Analysis 2022!
- <b>[2022.05.03]:</b> PyTorch implementation code released!
- <b>[2022.04.01]:</b> Demo code and pre-trained model released!

<br />




# Model Overview

<img src="files/rdv.png" width="38%" align="right" >

The RDV model is composed of:
* Feature Extraction layer: extract high and low level features from input image from a video
* Encoder: for triplet components encoding
    * Weakly-Supervised Localization (WSL) Layer: for localizing the instruments
    * Class Activation Guided Attention Mechanism (CAGAM): for detecting the verbs and targets leveraging attention resulting from instrument activations. (channel anad position spatial attentions are used here)
    * Bottleneck layer: for collecting unfiltered features for initial scene understanding
* Decoder: for triplet assocaition decoding over L successive layers
    * Multi-Head of Mixed Attention (MHMA): for learning to associate instrument-verb-target using successive self- and cross-attention mechanism
    * Feed-forward layer: for triplet feature refinement
* Classifier: for final triplet classification



We hope this repo will help researches/engineers in the development of surgical action recognition systems. For algorithm development, we provide training data, baseline models and evaluation methods to make a level playground. For application usage, we also provide a small video demo that takes raw videos as input without any bells and whistles.

<br />



# Performance

## Results Table


||Components AP ||||| Association AP |||
:---:|:---:|:---:|:---: |:---:|:---:|:---:|:---:|:---:|
AP<sub>I</sub> | AP<sub>V</sub> | AP<sub>T</sub> ||| AP<sub>IV</sub> | AP<sub>IT</sub> | AP<sub>IVT</sub> |
92.0 | 60.7 | 38.3 ||| 39.4 | 36.9 | 29.9|

<br />

## Video Demo



<a href="https://www.youtube.com/watch?v=d_yHdJtCa98&t=61s"><img src="files/vid.png" width="20.2%" ></a>

Available on Youtube.

<br />



# Installation

## Requirements

The model depends on the following libraries:
1. sklearn
2. PIL
3. Python >= 3.5
4. ivtmetrics
5. Developer's framework:
    1. For Tensorflow version 1:
        * TF >= 1.10
    2. For Tensorflow version 2:
        * TF >= 2.1
    3. For PyTorch version:
        - Pyorch >= 1.10.1
        - TorchVision >= 0.11

<br />

## System Requirements:
The code has been test on Linux operating system. It runs on both CPU and GPU.
Equivalence of basic OS commands such as _unzip, cd, wget_, etc. will be needed to run in Windows or Mac OS.

<br />

## Quick Start
* clone the git repository: ``` git clone https://github.com/CAMMA-public/rendezvous.git ```
* install all the required libraries according to chosen your framework.
* download the dataset
* download model's weights
* train
* evaluate

<br />


## Docker Example

coming soon . . .

<br />


# Dataset Zoo

* [CholecT45](https://github.com/CAMMA-public/cholect45) 
* CholecT50
* [Dataset splits](https://arxiv.org/abs/2204.05235)

<br />


## Data Preparation

* All frames are resized to 256 x 448 during training and evaluation.
* Image data are mean normalized.
* The dataset variants are tagged in this code as follows: 
   - cholect50 = CholecT50 with split used in the original paper.
   - cholect50-challenge = CholecT50 with split used in the CholecTriplet challenge.
   - cholect45-crossval = CholecT45 with official cross-val split **(currently public released)**.
   - cholect50-crossval = CholecT50 with official cross-val split.

<br />



## Evaluation Metrics

The *ivtmetrics* computes AP for triplet recognition. It also support the evaluation of the recognition of the triplet components.

```
pip install ivtmetrics
```

or

```
conda install -c nwoye ivtmetrics
```

Usage guide is found on [pypi.org](https://pypi.org/project/ivtmetrics/).

<br />


# Running the Model

The code can be run in a trianing mode (`-t`) or testing mode (`-e`)  or both (`-t -e`) if you want to evaluate at the end of training :

<br />

## Training on CholecT45/CholecT50 Dataset

Simple training on CholecT50 dataset:

```
python run.py -t  --data_dir="/path/to/dataset" --dataset_variant=cholect50 --version=1
```

You can include more details such as epoch, batch size, cross-validation and evaluation fold, weight initialization, learning rates for all subtasks, etc.:

```
python3 run.py -t -e  --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold=1 --epochs=180 --batch=64 --version=2 -l 1e-2 1e-3 1e-4 --pretrain_dir='path/to/imagenet/weights'
```

All the flags can been seen in the `run.py` file.
The experimental setup of the published model is contained in the paper.

<br />

## Testing

```
python3 run.py -e --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold 3 --batch 32 --version=1 --test_ckpt="/path/to/model-k3/weights"
```

<br />

 ## Training on Custom Dataset

Adding custom datasets is quite simple, what you need to do are:
- organize your annotation files in the same format as in [CholecT45](https://github.com/CAMMA-public/cholect45) dataset. 
- final model layers can be modified to suit your task by changing the class-size (num_tool_classes, num_verb_classes, num_target_classes, num_triplet_classes) in the argparse.

<br />



# Model Zoo

* **N.B.** Download links to models' weights will not be provided until after the CholecTriplet2022 challenge.


## PyTorch

| Network   | Base      | Resolution | Dataset   | Data split  |  Link             |
------------|-----------|------------|-----------|-------------|-------------------|
| Rendezvous| ResNet-18 | Low        | CholecT50 | RDV         |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | High       | CholecT50 | RDV         |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | Challenge   |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | crossval k1 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | crossval k2 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | crossval k3 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | crossval k4 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | crossval k5 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT45 | crossval k1 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT45 | crossval k2 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT45 | crossval k3 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT45 | crossval k4 |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT45 | crossval k5 |   [Google] [Baidu] |


<br />

## TensorFlow v1

| Network   | Base      | Resolution | Dataset   | Data split    | Link             |
------------|-----------|------------|-----------|---------------|------------------|
| Rendezvous| ResNet-18 | High       | CholecT50 | RDV           |  [Google] [Baidu] |
| Rendezvous| ResNet-18 | High       | CholecT50 | Challenge     |  [Google] [Baidu] |

<br />


## TensorFlow v2

| Network   | Base      | Resolution | Dataset   | Data split    | Link             |
------------|-----------|------------|-----------|---------------|------------------|
| Rendezvous| ResNet-18 | High       | CholecT50 | RDV           |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | Low        | CholecT50 | RDV           |   [Google] [Baidu] |
| Rendezvous| ResNet-18 | High       | CholecT50 | Challenge     |   [Google] [Baidu] |


<br />

## Baseline Models

TensorFlow v1
| Model | Layer Size | Ablation Component |AP<sub>IVT</sub> | Link |
------------|------|------------|------|-----|
|Rendezvous | 1 | Proposed | 24.6 | [Google] [Baidu] |
|Rendezvous | 2 | Proposed | 27.0 | [Google] [Baidu] |
|Rendezvous | 4 | Proposed | 27.3 | [Google] [Baidu] |
|Rendezvous | 8 | Proposed | 29.9 | [Google] [Baidu] |
|Rendezvous | 8 | Patch sequence | 24.1 | [Google] [Baidu] |
|Rendezvous | 8 | Temporal sequence | --.-- | [Google] [Baidu] |
|Rendezvous | 8 | Single Self Attention Head | 18.8 | [Google] [Baidu] |
|Rendezvous | 8 | Multiple Self Attention Head | 26.1 | [Google] [Baidu] |
|Rendezvous | 8 | CholecTriplet2021 Challenge Model | 32.7|   [Google] [Baidu] |
------------

Model weights are released periodically because some training are in progress.

<br /><br />


------------
# License


This code, models, and datasets are available for **non-commercial scientific research purposes** provided by [CC BY-NC-SA 4.0 LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/) attached as [LICENSE file](LICENSE). 
By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.


<br />

------

# Acknowledgment

This work was supported by French state funds managed within the Investissements d'Avenir program by BPI France in the scope of ANR project CONDOR, ANR Labex CAMI, ANR DeepSurg, ANR IHU Strasbourg and ANR National AI Chair AI4ORSafety.
We thank the research teams of IHU and IRCAD  for their help in the initial annotation of the dataset during the CONDOR project.

<br />

<img src="files/ihu.png" width="6%" align="right" > <img src="files/davenir.png" width="8%" align="right"> <img src="files/bpi.png.svg" width="14%" align="right"> <img src="files/ircad.png" width="10%" align="right">  <img src="files/hopital.png" width="7%"  align="right">  
<img src="files/condor.png" width="10%"  align="right">

<br />

<br />



------

# Related Resources
<b>
   
-  CholecT45 / CholecT50 Datasets
   [![Download dataset](https://img.shields.io/badge/download-camma-yellowgreen)](http://camma.u-strasbg.fr/datasets)    [![GitHub](https://img.shields.io/badge/github-CholecT45-blue)](https://github.com/CAMMA-public/cholect45)   
-  Offical Dataset Splits 
   [![Official dataset split](https://img.shields.io/badge/arxiv-2204.05235-red)](https://arxiv.org/abs/2204.05235)
-  Tripnet 
   [![ArXiv paper](https://img.shields.io/badge/arxiv-2007.05405-red)](https://arxiv.org/abs/2007.05405)    [![GitHub](https://img.shields.io/badge/github-tripnet-blue)](https://github.com/CAMMA-public/tripnet)   
-  Attention Tripnet
   [![ArXiv paper](https://img.shields.io/badge/arxiv-2109.03223-red)](https://arxiv.org/abs/2109.03223)    [![GitHub](https://img.shields.io/badge/github-attention.tripnet-blue)](https://github.com/CAMMA-public/attention-tripnet) 
-  CholecTriplet2021 Challenge
   [![Challenge website](https://img.shields.io/badge/website-2021.cholectriplet-lightgreen)](https://cholectriplet2021.grand-challenge.org)     [![ArXiv paper](https://img.shields.io/badge/arxiv-2204.04746-red)](https://arxiv.org/abs/2204.04746)    [![GitHub](https://img.shields.io/badge/github-2021.cholectriplet-blue)](https://github.com/CAMMA-public/cholectriplet2022) 
-  CholecTriplet2022 Challenge
   [![Challenge website](https://img.shields.io/badge/website-2022.cholectriplet-lightgreen)](https://cholectriplet2022.grand-challenge.org)    [![GitHub](https://img.shields.io/badge/github-2022.cholectriplet-blue)](https://github.com/CAMMA-public/cholectriplet2022)
 
</b>

<br />


------

# Citation
If you find this repo useful in your project or research, please consider citing the relevant publications:

- For the CholecT45/CholecT50 Dataset:
```
@article{nwoye2021rendezvous,
  title={Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos},
  author={Nwoye, Chinedu Innocent and Yu, Tong and Gonzalez, Cristians and Seeliger, Barbara and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
  journal={Medical Image Analysis},
  volume={78},
  pages={102433},
  year={2022}
}
```


- For the CholecT45/CholecT50 Official Dataset Splits:
```
@article{nwoye2022data,
  title={Data Splits and Metrics for Benchmarking Methods on Surgical Action Triplet Datasets},
  author={Nwoye, Chinedu Innocent and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2204.05235},
  year={2022}
}
```


- For the Rendezvous or Attention Tripnet Baseline Models or any snippet of code from this repo:
```
@article{nwoye2021rendezvous,
  title={Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos},
  author={Nwoye, Chinedu Innocent and Yu, Tong and Gonzalez, Cristians and Seeliger, Barbara and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
  journal={Medical Image Analysis},
  volume={78},
  pages={102433},
  year={2022}
}
```

- For the Tripnet Baseline Model:
```
@inproceedings{nwoye2020recognition,
   title={Recognition of instrument-tissue interactions in endoscopic videos via action triplets},
   author={Nwoye, Chinedu Innocent and Gonzalez, Cristians and Yu, Tong and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
   booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
   pages={364--374},
   year={2020},
   organization={Springer}
}
```


- For the models presented @ CholecTriplet2021 Challenge:
```
@article{nwoye2022cholectriplet2021,
  title={CholecTriplet2021: a benchmark challenge for surgical action triplet recognition},
  author={Nwoye, Chinedu Innocent and Alapatt, Deepak and Vardazaryan, Armine ... Gonzalez, Cristians and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2204.04746},
  year={2022}
}
```

#
This repo is maintained by [CAMMA](http://camma.u-strasbg.fr). Comments and suggestions on models are welcomed. Check this page for updates.