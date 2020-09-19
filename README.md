# FaceRecog_opencv_transfer_learning_VGG19

## A tool to collect, learn and recognise faces based on user requirement.

### How to use the tool
1. User can set the paramter in the file 'varConfig.py' which will decide the behaviour of the tool.
2. Script 'collectFaces.py' will collect the faces from the camera, using OpenCV, and stores then as training and testing data for as many labels as user's requirement.
3. Script 'learnFaces.py' will use the training and testing dataset from 'imageStore' location to build a model, using VGG19 transfer learning, and save it with user given name.
4. Script 'recogFaces.py' will used the model saved by 'learnFaces.py' and recognise the face read from the camera and write the label 
anme on the image.
5. Script 'recogFaces.py' will take care of collecting, learning and recognising the faces by calling the functions from 'collectFaces.py' & 'learnFaces.py'. User need not execute 'collectFaces.py' & 'learnFaces.py' scripts.
