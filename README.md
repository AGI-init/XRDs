![alt text](https://i.imgur.com/0Qp4YOb.png)

**Welcome to the repo for classifying crystal structures & space groups from 1D X-ray diffraction (XRD) patterns.**

*Can machine learning identify crystals in light diffraction patterns?* </br>
**[Check out our paper for more details and information](https://www.nature.com/articles/s41524-023-01164-8)**, and be sure to cite us.
 
---

Paper:

```bibtex
@article{Crystals,
title   = {Automated Classification of Big X-ray Data Using Deep Learning Models},
author  = {Jerardo Salgado; Sam Lerman; Zhaotong Du; Chenliang Xu; and Niaz Abdolrahim},
journal = {npj Computational Materials},
year    = {2023}
}
```

---

# :point_up: Setup

This project is built with the **[UnifiedML](https://github.com/AGI-init/UnifiedML)** deep learning library/framework.

```console
pip install UnifiedML
```

---

# Reproducing The Work

To run, we have 3 model variants for predicting **7-way crystal types**:

**Model 1: No-pool CNN**

```console
python XRD.py task=NPCNN
```

**Model 2: Standard CNN**

```console
python XRD.py task=SCNN
```

**Model 3: MLP**

```console
python XRD.py task=MLP
```

:bulb: **To predict 230-way space groups instead**, add the ```num_classes=230``` flag.

```console
python XRD.py task=NPCNN num_classes=230
```

Plots automatically save to ```./Benchmarking/<experiment>/```.

#

The above scripts will launch training on the "souped" **synthetic + random 50% RRUFF experimental data**, & evaluation on the **remaining 50% RRUFF data**. The trained model is saved in a ```./Checkpoints``` directory and can be loaded with the ```load=true``` flag.

All model and dataset code can be found in [```XRD.py```](XRD.py)

Custom datasets can be evaluated with the ```Dataset=``` flag and ```train_steps=0 load=true``` from a saved model.

# Differences from and additions to paper

**Synthetic data**

This repo automatically downloads the public CIF database as opposed to ICSD as in the paper. If you’d rather use ICSD and have access, you can download it to the ```Data/Generated/CIFs_ICSD/``` directory, and this code will automatically use that instead as in the paper. If you’d like to use both, add the ```open_access=true``` flag.

**Souping and evaluation data**

This GitHub provides the experimental real-world data RRUFF. It will be detected and used for souping as described in the paper. That is, reserving a random 50% subset of the real-world data for training and the remaining 50% for evaluation. If you’d like to disable souping, use the ```soup=false``` flag. If you’d like to train only on a 0.9/0.1 split of the synthetic data, you can use ```rruff=false```.

---

# Citing

If you find this work useful, be sure to cite us:

```bibtex
@article{Crystals,
title   = {Automated Classification of Big X-ray Data Using Deep Learning Models},
author  = {Jerardo Salgado; Sam Lerman; Zhaotong Du; Chenliang Xu; and Niaz Abdolrahim},
journal = {npj Computational Materials},
year    = {2023}
}
```

---

<a href="http://www.github.com/AGI-init/UnifiedML" target="_blank">
<img width="15%" alt="flowchart" src="https://i.imgur.com/Ya9FpIJ.png">
</a>

All **[UnifiedML](https://github.com/AGI-init/UnifiedML)** features and syntax are supported.

#

[This code is licensed under the MIT license.](MIT_LICENSE)