# Deep learning based search method

## Overview
We investigated in planar object tracking task and present with an idea of using a deep convolutional neural network for estimating the relative updating warp parameters between a pair of images. We use a corner displacement parameterization which we can use to represent and recover the warping transformation between two corners. Initial theory workout and the training of the network have shown that developing such a neural network based search method for warping parameters updates is possible.
More detail of the project can be found in [report]().

## Prerequisites
+ Python3
+ Keras
+ Numpy
+ OpenCV

## Results
| Model                     | DeTone et. al. | Our Re-creation |
| ------------- | ----------------- | ---------------- |
| Mean corner error         |     9.3951        |      9.2      |

## Team members
| Author           |
| ---------------- |
| [Chen Jiang](https://github.com/zonetrooper32)       |
| Steven Weikai Lu |

## Acknowledgement
This is the final project report for CMPUT 428, instructed by [Prof. Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/) and Teaching Assistant [Abhineet Singh](http://webdocs.cs.ualberta.ca/~vis/asingh1/), University of Alberta. While we worked on our project in a deeply-cooperating fashion, a work distribution of each group member can be roughly listed as follows:
 * Chen Jiang: Overall project design, network architecture design, coding
 * Steven Weikai Lu: Synthetic dataset generation, illumination and occlusion perturbation.
