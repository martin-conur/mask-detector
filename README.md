# FASK: A face mask detector implementation
a project with tensorflow and opencv. 

This is the 3rd of three individual projects

1) [First Project: A face dataset maker, using a state of the art SSD trained on face detection we created a dataset of faces with mask and without mask.](https://github.com/martincontrerasu/face_ROI_extractor)
2) Second Project: Creating a classifier of mask or nomask classes. We use transfer learning on a trained mobilenet.
3) Third Project (this one): Using the classifier of the 2nd project and the SSD of the first project to detect masks on real time.

## Examples:
Many faces:
![](https://raw.githubusercontent.com/martincontrerasu/mask-detector/master/examples/01.gif)

or one...
![](https://raw.githubusercontent.com/martincontrerasu/mask-detector/master/examples/02.gif)

## Uses:
In a command line type 
```console
python fask.py
```
by default uses the first webcam, but you can give it a video by typing  
```console
python fask.py -i "path_to_my_video/my_video.mp4"
```
By default uses a confident value of 0.5, you can set to other desired value between 0 and 1 by typing
```console
python fask.py -c X
```
X=desired confidence value

