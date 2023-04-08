# An Intelligent Modular Real-Time Vision-Based System for Environment Perception

A significant portion of driving hazards is caused by human error and disregard for local driving regulations; consequently, an intelligent assistance system can be beneficial. Hence, we propose a vision-based modular package to ensure drivers’ safety by perceiving the environment. Each module is designed based on accuracy and inference time to deliver real-time performance. As a result, the proposed system can be implemented on a wide range of vehicles with minimum hardware requirements. Our modular package comprises four main sections: lane detection, object detection, segmentation, and monocular depth estimation. Each section is accompanied by novel techniques to improve the accuracy of others along with the entire system. Furthermore, a GUI is developed to display perceived information to the driver. 

![overall_diagram](https://user-images.githubusercontent.com/61879630/199366037-69f5a025-73d5-428b-a2de-5742532946d3.jpg)


## Citation
```python

@article{kazerouni2023intelligent,
  title={An intelligent modular real-time vision-based system for environment perception},
  author={Kazerouni, Amirhossein and Heydarian, Amirhossein and Soltany, Milad and Mohammadshahi, Aida and Omidi, Abbas and Ebadollahi, Saeed},
  journal={arXiv preprint arXiv:2303.16710},
  year={2023}

```

## Updates

- October 20, 2022: Accepted in NeurIPS 2022 Workshop on Machine Learning for Autonomous Driving! :fire:
- February 5, 2021: Won 1st place in the National Rahneshan competition 2020-2021 for autonomous vehicles! :tada:
- January 10, 2021: First release.


## Results

### Results on BDD100K dataset

![bdd](https://user-images.githubusercontent.com/61879630/199363094-6149ddd8-d2e8-4343-bea1-fee052d8bd5b.jpg)

### Results on our local dataset

![local](https://user-images.githubusercontent.com/61879630/199363107-e4ebb719-ac51-49f7-ae61-f663caaad6c6.jpg)


## Inference
To run the program, first install the requirements using the code below:
```
$ pip install -r requirements.txt
```
Then create a folder named 'weights' in the main directory and download all the weights in [this](https://drive.google.com/uc?export=download&id=1X1uKaGENEBZamF6tOfx9eKLTIQLsBN5h) shared google drive folder.

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

Simply open the following colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pandas-Team/Autonomous-Vehicle-Environment-Perception/blob/main/Instructions.ipynb)

## Cited Works
1. Yolov5 ([Github](https://github.com/ultralytics/yolov5))
2. SGDepth ([Github](https://github.com/ifnspaml/SGDepth))
3. PINet ([Github](https://github.com/koyeongmin/PINet_new))

## Datasets

### Test Videos:
Please download from [here](https://drive.google.com/uc?export=download&id=1-bRFhDt5EZULnQaKO35U3oX-p6yZwteB).

### Sign Datasets:
1. Traffic-Sign Detection and Classification in the Wild [Link](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
2. DFG Traffic Sign Data Set [Link](https://www.vicos.si/Downloads/DFGTSD#:~:text=Dataset%20consists%20of%20200%20traffic,around%207000%20high%2Dresolution%20images.&text=The%20images%20have%20been%20anonymized,with%20the%20EU%20GDPR%20legislation.)


## Our Team
We as Team Pandas won 1st place in the National Rahneshan competition 2020-2021 for autonomous vehicles. This contest has been one of the most competitive and challenging contests in the Rahneshan tournaments with more than 15 teams competing from top universities in Iran.
![Pandas6](https://user-images.githubusercontent.com/44018277/113591619-5e12c700-9649-11eb-805d-dd504081456e.jpg)

### Contact us
Feel free to contact us via email or connect with us on linkedin.

- Milad Soltany --- [Linkedin](https://www.linkedin.com/in/milad-soltany/), [Github](https://github.com/miladsoltany) , [Email](mailto:soltany.m.99@gmail.com)
- Abbas Omidi --- [Linkedin](https://www.linkedin.com/in/abbasomidi77/), [Github](https://github.com/abbasomidi77), [Email](mailto:abbasomidi77@gmail.com)
- Amirhossein Kazerouni ---  [Linkedin](https://www.linkedin.com/in/amirhossein477/), [Github](https://github.com/amirhossein-kz), [Email](mailto:Amirhossein477@gmail.com )
- Amirhossein Heydarian ---  [Linkedin](https://www.linkedin.com/in/amirhosseinh77/), [Github](https://github.com/amirhosseinh77), [Email](mailto:amirhossein4633@gmail.com )
- Aida Mohammadshahi ---  [Linkedin](https://www.linkedin.com/in/aida-mohammadshahi-9845861b3/), [Github](https://github.com/aidamohammadshahi), [Email](mailto:aidamoshahi@gmail.com)
