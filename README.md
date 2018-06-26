# UNREAL

[![CircleCI](https://circleci.com/gh/miyosuda/unreal.svg?style=svg)](https://circleci.com/gh/miyosuda/unreal)

## About
Code accompanying paper "Attend Before you Act: Leveraging human visual attention for continual learning" - at Lifelong Learning: A Reinforcement Learning Approach Workshop @ ICML 2018 

Paper Link: 

Implemented with TensorFlow and DeepMind Lab environment.

## Supplimentary Material 
[Visually-Attentive UNREAL agent navigating the 3D maze ](https://sites.google.com/view/attendbeforeyouact)

## Network
![Network](./doc/network0.png)

All weights of convolution layers and LSTM layer are shared.

## Requirements

- TensorFlow (Tested with r1.0)
- DeepMind Lab
- numpy
- cv2
- pygame
- matplotlib

## Result
"nav_maze_static_01" Level

![nav_maze_static_01_score](experiments/baseline/Unreal_BaselineRun_ScorePlot.png)


## How to train
First, download and install DeepMind Lab
```
$ git clone https://github.com/deepmind/lab.git
```
Then build it following the build instruction. 
https://github.com/deepmind/lab/blob/master/docs/build.md

Clone this repo in lab directory.
```
$ cd lab
$ git clone https://github.com/miyosuda/unreal.git
```
Add this bazel instruction at the end of `lab/BUILD` file

```
package(default_visibility = ["//visibility:public"])
```

Then run bazel command to run training.
```
bazel run //unreal:train --define headless=glx
```
`--define headlesss=glx` uses GPU rendering and it requires display not to sleep. (We need to disable display sleep.)

If you have any trouble with GPU rendering, please use software rendering with `--define headless=osmesa` option.

## How to show result

To show result after training, run this command.
```
bazel run //unreal:display --define headless=glx
```
