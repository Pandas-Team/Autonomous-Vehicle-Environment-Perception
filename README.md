# Rahneshan-Environment-Perception-Group
In this repo, the environment perception task of Rahneshan 2020 is done.

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

There you have it
