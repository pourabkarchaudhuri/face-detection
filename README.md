# Face Detection Implementations
Detects face from webcam using multiple algoritms and models

### Face Detection Algorithms
- Haarcascade Classifier
- MTCNN FaceNet
- ResNet50

### Technology

Oversight uses a number of open source projects to work properly:

* [Tensorflow] - A google open-source ML framework
* [Python] - awesome language we love

### Dependencies
- tensorflow 
- opencv 
- numpy

### Usage

```sh
$ git clone https://github.com/pourabkarchaudhuri/face-detection.git
cd face-detection
cd src
```

### Environment Setup

##### This was built on Windows 10.

These were the pre-requisities :

##### NVIDIA CUDA Toolkit
* [CUDA] - parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). Download and Install all the patches. During install, choose Custom and uncheck the Visual Studio Integration checkbox.

##### Download cuDNN
* [cuDNN] - The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. Create a NVIDIA developer account to download.

##### Set Path :
Add the following paths,
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\extras\CUPTI\libx64

##### Install [Anaconda](https://www.anaconda.com/download/) with 3.6 x64

```sh
$ conda update conda
```


##### Install C/C++ Build tools

* [C/C++ Build Tools] - Custom librarires required to build C based implementations to Python runnable builds


#### To Run HaarCascade on Webcam
```sh
python detect_face_haarcascade.py
```
#### To Run HaarCascade on Webcam with Multithreading
```sh
python detect_face_haarcascade_multithreading.py
```
#### To Run ResNet Classifier on Webcam
```sh
python detect_face_caffe_resnet.py -p ..\caffe\deploy.prototxt.txt -m ..\caffe\res10_300x300_ssd_iter_140000.caffemodel
```
#### To Run FaceNet with MTCNN on Webcam
```sh
python detect_face_mtcnn_gpu.py
```
License
----

Public


   [Tensorflow]: <https://www.tensorflow.org/>
   [Python]: <https://www.python.org/>
   [Google's FaceNet]: <https://arxiv.org/abs/1503.03832>
   [Anaconda]: <https://www.anaconda.com/download/>
   [CUDA]: <https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal>
   [cuDNN]: <https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7>
   [Pretrained Model]: <https://drive.google.com/open?id=1sOMaZYWyWJJKJkQFVf3TUTX6-1iyR-kV>
   [C/C++ Build Tools]: <https://go.microsoft.com/fwlink/?LinkId=691126>
