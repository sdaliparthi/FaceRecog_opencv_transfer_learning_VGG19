# collectFace variables
haarFrontalFace    = 'haar_cascade_files/haarcascade_frontalface_default.xml'
haarEyes           = 'haar_cascade_files/haarcascade_eye_tree_eyeglasses.xml'
scaleFactor        = 1.3
minNeighbors       = 5
camId              = 0
camImagesCnt       = 100
imageStore         = './CamImagesStore/'
faceLabel          = None # 'Shashi'

# learnFace variables
imgSize            = [224, 224, 3] # A color image
imgTargetSize      = (224, 224)
epochCount         = 5
imageGenBatchSize  = 32
trainDataSetpath   = imageStore + '/train/'
testDataSetpath    = imageStore + '/test/'
modelNameToSave    = 'faceRecogModel_vgg19.h5'

# recogFace variables
collectFacesDatSetUsingCam = False
learnFacesUsingVGG19       = False
