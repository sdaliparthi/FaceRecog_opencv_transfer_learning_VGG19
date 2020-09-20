#########---------------------------------------------#########
'''
Script     = recogFaces.py
Author     = "Shashi Kanth Daliparthi"
License    = "GPL"
Version    = "1.0.0"
Maintainer = "Shashi Kanth Daliparthi"
E-mail     = "shashi.daliparthi@gmail.com"
Status     = "Production"
'''
#########---------------------------------------------#########


#########---------------------------------------------#########
# Import required modules
import cv2, os
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, load_model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from learnFaces import *
from collectFaces import *
from varConfig import *
#########---------------------------------------------#########


#########---------------------------------------------#########
def predictFaceClass(model, class_indices):
    while True:
        # Initialize primary cam
        cv2.startWindowThread()
        camInst = cv2.VideoCapture(camId)

        # Read image from cam
        flag, camImgData = camInst.read()
        reqCamImgData = camImgData.copy()

        # Predict the face using the VGG19 model
        camImgFaceCrop = markFace(reqCamImgData)
        if camImgFaceCrop is not None:
            # Convert the omage to 128x128 size to make it compatable with our custom VGG19 model
            imgData = cv2.resize(reqCamImgData, (224, 224))
            faceImg = Image.fromarray(imgData, 'RGB')
            faceImgData = np.array(faceImg)
            # Expand the image diamentions to 4D as expected by Keras
            faceImgData = np.expand_dims(faceImgData, axis=0)

            # Predict the class based on the came image
            facePred = model.predict(faceImgData)
            print(f" ##> Custom VGG19 model predicted {facePred}. Actual Class labels : {class_indices}")
            predFaceLabel = class_indices[facePred.argmax()]

            cv2.putText(reqCamImgData, str(predFaceLabel), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('--LOCAL CAM FEED--', reqCamImgData)
        else:
            print(" ##> Unable to detect a face.\n")
            cv2.putText(reqCamImgData, 'NO FACE', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('--LOCAL CAM FEED--', reqCamImgData)

        # Get out of loop when 'q' key is pressed or when required no.of images have been collected.
        if cv2.waitKey(1) == ord('q'):
            print(" ##> Aborting the process of face recognition.\n")
            break
#########---------------------------------------------#########


if __name__ == "__main__":
    # Collect Faces
    if collectFacesDatSetUsingCam:
        while True:
            faceLabel = input("Enter face label (** Press ENTER to SKIP **) : ")
            if faceLabel == '': break
            colectFacesFrmCam(faceLabel = faceLabel)
            print(" ##> Successfully collected the faces from cam.\n")

    # Learn the faces
    testTrainDataset, class_indices = buildFaceImageGenerator()
    if learnFacesUsingVGG19:
        (model, modelFit) = buildVGG19Model(dataset = testTrainDataset)
        plotLossAcc(modelFit = modelFit)
        if modelNameToSave:
            print(f"Attempting to save the model as {modelNameToSave}.\n")
            saveModel(model=model, modelName=modelNameToSave)

    # Load the previously saved custom VGG19 model
    if modelNameToSave:
        if os.path.exists(modelNameToSave):
            vgg19Model = load_model(modelNameToSave)
            predictFaceClass(model=vgg19Model, class_indices=class_indices)
        else:
            print(f" ##> Unable to find the model file {modelNameToSave}. Load/Learn the model.\n");
