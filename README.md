![neodroid](.github/images/header.png)

# Vision
This repository will host implementation computer vision algorithms applying the [Neodroid](https://github
.com/sintefneodroid/) platform.

---

_[Neodroid](https://github.com/sintefneodroid) is developed with support from Research Council of Norway Grant #262900. ([https://www.forskningsradet.no/prosjektbanken/#/project/NFR/262900](https://www.forskningsradet.no/prosjektbanken/#/project/NFR/262900))_

---

| [![Build Status](https://travis-ci.org/sintefneodroid/agent.svg?branch=master)](https://travis-ci.org/sintefneodroid/agent)  | [![Coverage Status](https://coveralls.io/repos/github/sintefneodroid/agent/badge.svg?branch=master)](https://coveralls.io/github/sintefneodroid/agent?branch=master)  | [![GitHub Issues](https://img.shields.io/github/issues/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/issues)  |  [![GitHub Forks](https://img.shields.io/github/forks/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/network) | [![GitHub Stars](https://img.shields.io/github/stars/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/stargazers) |[![GitHub License](https://img.shields.io/github/license/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/blob/master/LICENSE.md) |
|---|---|---|---|---|---|

<p align="center" width="100%">
  <a href="https://www.python.org/">
    <img alt="python" src=".github/images/python.svg" height="40" align="left">
  </a>
  <a href="https://opencv.org/" style="float:center;">
    <img alt="opencv" src=".github/images/opencv.svg" height="40" align="center">
  </a>
  <a href="http://pytorch.org/"style="float: right;">
    <img alt="pytorch" src=".github/images/pytorch.svg" height="40" align="right" >
  </a>
</p>
<p align="center" width="100%">
  <a href="http://www.numpy.org/">
    <img alt="numpy" src=".github/images/numpy.svg" height="40" align="left">
  </a>
  <a href="https://github.com/tqdm/tqdm" style="float:center;">
    <img alt="tqdm" src=".github/images/tqdm.gif" height="40" align="center">
  </a>
  <a href="https://matplotlib.org/" style="float: right;">
    <img alt="matplotlib" src=".github/images/matplotlib.svg" height="40" align="right" />
  </a>
</p>

# Contents Of This Readme
- [Algorithms](#algorithms)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
  - [Target Point Estimator](#target-point-estimator)
- [Contributing](#contributing)
- [Other Components](#other-components-of-the-neodroid-platform)

# Algorithms
- [REINFORCE (PG)](agents/pg_agent.py)
- [DQN](agents/dqn_agent.py)

# Requirements
- pytorch
- tqdm
- Pillow
- numpy
- matplotlib
- torchvision
- torch
- Neodroid
- pynput

To install these use the command:
````bash
pip3 install -r requirements.txt
````

# Usage
Export python path to the repo root so we can use the utilities module
````bash
export PYTHONPATH=/path-to-repo/
````
For training a agent use:
````bash
python3 procedures/train_agent.py
````
For testing a trained agent use:
````bash
python3 procedures/test_agent.py
````

# Results

## Target Point Estimator
Using Depth, Segmentation And RGB images to estimate the location of target point in an environment.

### [REINFORCE (PG)](agents/pg_agent.py)

### [DQN](agents/dqn_agent.py)


# Contributing
See guidelines for contributing [here](CONTRIBUTING.md).

# Citation

For citation you may use the following bibtex entry:

````
@misc{neodroid-agent,
  author = {Heider, Christian},
  title = {Neodroid Vision},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sintefneodroid/Vision}},
}
````
