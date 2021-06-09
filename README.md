# Autonomous-Vehicle-Environment-Perception
This repository contains Pandas Team implementation of Autonomous Vehicles Environment Perception Task.

Environment Perception is a crucial asset when it comes to Autonomous Vehicles. The system is required to perceive several entities in its field of view. Said entities include but are not limited to pedestrians, other vehicles,  traffic lights, traffic signs, distance relative to other objects on the road, cross-walks, and side-walks. In this work, we utilize various computer vision methods and algorithms to fulfill the sought-after task.

# Abstract
In this project, we designed and coded an environmental perception system for an autonomous vehicle. Applications of this system include identifying pedestrians, traffic lights, and signs, identifying vehicles, and detecting distances from them, as well as identifying roadside and pedestrian lanes. A variety of neural networks and machine learning algorithms as well as classical machine vision techniques such as the huff algorithm have been used in this project.


Below You Can See Pictures of the Output:

![14](https://user-images.githubusercontent.com/61683254/121145468-3b24af00-c854-11eb-8597-1f8d1d64e57f.PNG)
![10](https://user-images.githubusercontent.com/61683254/121145068-d406fa80-c853-11eb-846c-94d4735f2569.PNG)
![11](https://user-images.githubusercontent.com/61683254/121145075-d5382780-c853-11eb-9e69-8b9c3f3add6f.PNG)
![12](https://user-images.githubusercontent.com/61683254/121145081-d6695480-c853-11eb-9f87-b0c74bc65f4b.PNG)


## Inference
To run the program, first install the requirements using the code below:
```
$ pip install -r requirements.txt
```
Then create a folder named 'weights' in the main directory and download all the weights in [this](https://drive.google.com/u/0/uc?id=1-MpEWgI-s1V5d6O5iq8cd29yKcrBkO_4&export=download) shared google drive folder.

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

## Colab Notebook
You can also use the provided colab notebook to automatically download all the weights and sample video, and run the program in a matter of seconds!

simply open the following colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pandas-Team/Autonomous-Vehicle-Environment-Perception/blob/main/Pandas_Team.ipynb)

## Cited Works
1. Yolov5 [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)
2. SGDepth [GithubRepo](https://github.com/ifnspaml/SGDepth), Also [Paper](https://arxiv.org/abs/2007.06936)
3. PINet [GithubRepo](https://github.com/koyeongmin/PINet_new)
## Datasets
1. Traffic-Sign Detection and Classification in the Wild [Link](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
2. DFG Traffic Sign Data Set [Link](https://www.vicos.si/Downloads/DFGTSD#:~:text=Dataset%20consists%20of%20200%20traffic,around%207000%20high%2Dresolution%20images.&text=The%20images%20have%20been%20anonymized,with%20the%20EU%20GDPR%20legislation.)

## Our Team
We as Team Pandas won 1st place in the National Rahneshan competition 2020-2021 for autonomous vehicles. This contest has been one of the most competitive and challenging contests in the Rahneshan tournaments with more than 15 teams competing from top universities in Iran.
![Pandas6](https://user-images.githubusercontent.com/44018277/113591619-5e12c700-9649-11eb-805d-dd504081456e.jpg)

### Contact us
Feel free to contact us via email or connect with us on linkedin.

- Milad Soltany --- [Linkedin](https://www.linkedin.com/in/milad-soltany/), [Github](https://github.com/miladsoltany) , [Email](mailto:soltany.m.99@gmail.com)
- Abbas Omidi --- [Linkedin](https://www.linkedin.com/in/abbasomidi77/), [Github](https://github.com/abbasomidi77), [Email](mailto:abbasomidi77@gmail.com)
- Amirhossein Kazerooni ---  [Linkedin](https://www.linkedin.com/in/amirhossein477/), [Github](https://github.com/amirhossein-kz), [Email](mailto:Amirhossein477@gmail.com )
- Amirhossein Heydarian ---  [Linkedin](https://www.linkedin.com/in/amirhosseinh77/), [Github](https://github.com/amirhosseinh77), [Email](mailto:amirhossein4633@gmail.com )
- Aida Mohammadshahi ---  [Linkedin](https://www.linkedin.com/in/aida-mohammadshahi-9845861b3/), [Github](https://github.com/aidamohammadshahi), [Email](mailto:aidamoshahi@gmail.com)
