# Visually Attentive UNREAL
Motivation: Where do humans look while navigating in a 3D maze environment ? Does foveating around the regions where humans look helps the reinforcement learning process in the context of continual learning ? We hypothesise that knowing where to look in a task aids continual learning across tasks

![Where do we look in an image](https://github.com/kkhetarpal/unrealwithattention/blob/master/Where%20Humans%20Look.png)

We introduce the Visually-Attentive UNREAL agent 2 by foveating around the salient regions in each image. This is done in the base process of online A3C , as shown in the pseudo code in Algorithm 1 of the paper.

## About
Code accompanying paper "Attend Before you Act: Leveraging human visual attention for continual learning" - at Lifelong Learning: A Reinforcement Learning Approach Workshop @ ICML 2018 

[Link to Paper](https://arxiv.org/abs/1807.09664) 


## Supplimentary Material 
![Project Page](https://sites.google.com/view/attendbeforeyouact)


## Result
Learning with varying degrees of visual attention to navigate the 3D maze environment. Specific degrees of visual attention
helps in learning better than baseline UNREAL agent. Here α = 0.69 speeds up the learning as compared to other settings for
this instance of runs.

![Learning with varying degrees of visual attention to navigate the 3D maze environment](https://github.com/kkhetarpal/unrealwithattention/blob/master/Different_Degrees_of_Foveation.png)


## Requirements

- TensorFlow
- DeepMind Lab
- numpy
- cv2
- pygame
- matplotlib

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
$ git clone https://github.com/kkhetarpal/unrealwithattention.git
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
