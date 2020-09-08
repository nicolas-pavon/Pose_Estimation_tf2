# Pose_Estimation_tf2
Pose estimation using Tensorflow 2.x, based on:
- https://github.com/ildoonet/tf-pose-estimation
- https://github.com/gsethi2409/tf-pose-estimation
- Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

# Test 
test on:
- Intel Core i5 - Nvidia GTX 1050 - Intel core i Quad-Core

|cmu| mobilenet_thin | mobilenet_v2_large | mobilenet_v2_small |
|:-------|:---------|:--------------------|:----------------|
|**~4 FPS** (656x368)| **~15 FPS** | **~10 FPS** | **~12 FPS**|

# Step by Step 

## Linux

### 1. Clone repository

open a terminal (ctrl+alt+t) then go into your working directory, and put:

```
$ git clone https://github.com/nicolas-pavon/Pose_Estimation_tf2.git
```

### 2. Create a new conda enviorment.

you must have anaconda or miniconda installed

you need dependencies such as:
  - Python.
  - Tensorflow.
  - OpenCv.
  - Slidingwindow.
  
go into **conda-env.yml** file and see all dependencies and libraries.

#### to create the environment(conda):
after clone the repository put:
```
$ cd Pose_Estimation_tf2
$ conda create env -f conda-env.yml
$ conda activate tf2-gpu
```
note that you can change the enviroment's name, for example if your computer doesn't have GPU, you could change the name just for tf2 or whatever you want.

### 3. Build c++ library for post processing.
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python setup.py build_ext --inplace
```

### 4. Models 

#### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files(CUM). You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)(not in the repo)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so ildoonet uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder bt running:

```
$ cd models/graph/cmu
$ bash download.sh
$ cd ../..
```

### 5. Real time WebCam or video path

test by running your pc's camera or a video:
```
$ python Pose_estimation.py
```
if you want to use the CMU original model:
```
$ python Pose_estimation.py --model cmu --resize 656x368
```
or usign the sample video
```
$ python Pose_estimation.py --video videos/sample_video-mp4
```




