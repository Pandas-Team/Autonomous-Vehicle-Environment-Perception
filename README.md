# Autonomous-Vehicle-Environment-Perception
This repository contains Pandas Team implementation of Autonomous Vehicles Environment Perception Task.

Environment Perception is a crucial asset when it comes to Autonomous Vehicles. The system is required to perceive several entities in its field of view. Said entities include but are not limited to pedestrians, other vehicles,  traffic lights, traffic signs, distance relative to other objects on the road, cross-walks, and side-walks. In this work, we utilize various computer vision methods and algorithms to fulfill the sought-after task.

![output_img](https://user-images.githubusercontent.com/44018277/112603426-9db00680-8e32-11eb-87d8-6954337fe1b9.jpg)


## Inference
To run the program, first install the requirements using the code below:
```
$ pip install -r requirements.txt
```
Then create a folder named 'weights' in the main directory and download all the weights in [this](https://drive.google.com/drive/folders/1skKYyZMSIAmJv52jrV4kHujjHwfF_Sx4?usp=sharing) shared google drive folder.

Then, place your video in the main folder of this repo and then run the following command.
```
$ python main.py --video yourvideoname.mp4 [--save] [--noshow] [--output-name myoutputvideo.mp4] [--fps]
```
--save argument will save the output video.

--noshow will not show you a preview of the output.

--output-name will determine the name you want for your output video

--fps will plot the fps results on the output frames

"yourvideoname.mp4" is the name of your video file added to the main folder.
"myoutputvideo.mp4" is the name you want for your output video.

Afterwards, the program starts running and the output video will be saved in the specified directory. To view the output while running, do not use '--no-show' argument.

There you have it.

## Cited Works
1. Yolov5 [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)
2. SGDepth [GithubRepo](https://github.com/ifnspaml/SGDepth), Also [Paper](https://arxiv.org/abs/2007.06936)
3. PINet [GithubRepo](https://github.com/koyeongmin/PINet)
## Datasets
1. Traffic-Sign Detection and Classification in the Wild [Link](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
2. DFG Traffic Sign Data Set [Link](https://www.vicos.si/Downloads/DFGTSD#:~:text=Dataset%20consists%20of%20200%20traffic,around%207000%20high%2Dresolution%20images.&text=The%20images%20have%20been%20anonymized,with%20the%20EU%20GDPR%20legislation.)
