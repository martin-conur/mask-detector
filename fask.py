"""Fask is a face mask detector app in python with cv2 and tensorflow. Its a simple idea, for every frame in
the video (webcam livestream) it detects faces and crops its detections, then it passes this cropped detections
to another Neural Net that classify if the face is using a mask or not.

    This two step aproach is simple to use, because is relatively easy to train the classifier (2nd NNet), but relies on
    the detection of the first Neural Net, so if the first NN doesnt predict a face (and is actually a face) it will never
    pass to the second.

    The best solution is to build one FCNN that predict the faces and bounding box of faces with mask and without mask in one pass,
    but this strategy requires a lot more data than a classifier.
"""
#libraries needed
import cv2
import numpy as np
import os
import time

#Deep learning libraries
import tensorflow as tf

#hyperparams
CONFIDENCE = 0.5 #for face detection

#setting the current directory (to avoid issues)
os.chdir(os.path.dirname(__file__))

def face_detection_and_mask_classifier(frame, faceDetectionNet, maskClassifierNet):
    """this function it receives every frame from a video straimng (frame var) and returns
       the locations and predictions of the faces """


    #getting the height and width of the frame
    (h, w) = frame.shape[:2]
    #using cv2's blobFromImage to perform image transfomations required
    #to make a prediction by the face detection net (res10_300x300_ssd_iter_140000)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))

    #pass the transformed image (blob) to the ssd net and obtain the detections
    faceDetectionNet.setInput(blob)
    detections = faceDetectionNet.forward()

    #empy lists to store the predictions (faces, locations and predictions)
    faces, locations, classifications = [],[],[]

    #for every detection do the classification
    for i in range(0, detections.shape[2]):
        #get the confidence of the prediction
        confidence = detections[0, 0, i, 2] #depends on the Nets outputs
        #if the confidence overpass the threshold value (CONFIDENCE hyperparameter)
        if confidence >CONFIDENCE:
            #get the bounding box coordinates (4 corners) and scales it to the img size
            (x0, y0, x1, y1) = (detections[0, 0, i, 3:7]*np.array([w, h, w, h])).astype(int)

            #avoid to get a point outside the frame
            #bottom left points sshould be greater than 0
            (x0, y0) = (max(0, x0), max(0, y0))
            #top right points should be less than 1
            (x1, y1) = (min(w-1, x1), min(h-1, y1))

            #with the bounding box coordinates extract the ROI
            face = frame[y0:y1, x0:x1]
            #from BGR to RGB (cv2 uses BGR and SSD uses RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            #resize to 224x224 (SSDs input)
            face = cv2.resize(face, (224, 224))
            #convert it to an array
            face = tf.keras.preprocessing.image.img_to_array(face)
            #preprocessing for mobilenet
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            #appending the face and boundig box to the lists
            faces.append(face)
            locations.append((x0, y0, x1, y1))

    #if detect at least one face do the classification
    if len(faces)>0:
        faces = np.array(faces, dtype="float32")
        classifications = maskClassifierNet.predict(faces, batch_size=16)

    #return the face locations and the classification (with or without mask) as a tuple
    return (locations, classifications)

print("[INFO] Cargando modelos deep learning ...")
#face detection model
prototxtPath = os.path.join(os.getcwd(), "face_detector_model/deploy.prototxt")
weightsPath = os.path.join(os.getcwd(), "face_detector_model/res10_300x300_ssd_iter_140000.caffemodel")
faceDetectionNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#mask classifier model
maskClassifierNet = tf.keras.models.load_model("mask_detector.model")

#start camera livestream
print("[INFO] Comenzando lectura de video ...")
video = cv2.VideoCapture(0)
#check if there's Video
if not video.isOpened():
    print("(!)Error al abrir la cámara")
    exit(0)

time.sleep(2)

#loop over every frame
while True:
    ret, frame = video.read()
    #in this part it could be resized to assure a certain max shape
    #i won't do that now
    #frame = cv2.resize(frame, (400,400))

    #call the function for detect faces and classify
    (locations, classifications) = face_detection_and_mask_classifier(frame, faceDetectionNet, maskClassifierNet)

    #for every face detected draw it bboxes and its class
    for (bbox, classification) in zip(locations, classifications):
        (x0, y0, x1, y1) = bbox
        (mask, nomask) = classification

        #define a label for mask and nomask, and colors
        label = "Con Mascarilla" if mask>nomask else "Sin Mascarilla"
        #red if nomask, green if mask
        color = (0, 255, 0) if label == "Con Mascarilla" else (0, 0, 255)

        #add in the label the probability that the class corresponds to mask or nomask
        label = f"{label}: {np.round(max(mask, nomask)*100, 2)}"

        #diplay bounding boxes and labels with an offset (10 px)
        cv2.putText(frame, label, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x0, y0), (x1,y1), color, 2)

    #plot the frame
    cv2.imshow("Detector de mascarilla", frame)
    key = cv2.waitKey(1) & 0xFF

    #break the videostream if "q" is pressed
    if key == ord("q"):
        break
#finish up
cv2.destroyAllWindows()
video.stop()
