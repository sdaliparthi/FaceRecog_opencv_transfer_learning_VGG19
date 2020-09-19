#########---------------------------------------------#########
'''
Script     = learnFaces.py
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
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, load_model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from varConfig import *
#########---------------------------------------------#########


#########---------------------------------------------#########
def buildFaceImageGenerator():
    global trainDataSetpath, testDataSetpath

    # Build an image data generator objects
    imgGenTrain = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    imgGenTest = ImageDataGenerator(rescale = 1./255)
    # Build training and test image generator
    trainImgDataset = imgGenTrain.flow_from_directory(trainDataSetpath, target_size = imgTargetSize, batch_size = imageGenBatchSize, class_mode = 'categorical')
    testImgDataset = imgGenTest.flow_from_directory(testDataSetpath, target_size = imgTargetSize, batch_size = imageGenBatchSize, class_mode = 'categorical')
    class_indices = dict([(val, key) for key,val in trainImgDataset.class_indices.items()])
    return [{'TRAINING_DATA_GENERATOR':trainImgDataset, 'TESTING_DATA_GENERATOR':testImgDataset}, class_indices]


def buildVGG19Model(dataset):
    classCnt = len(glob(trainDataSetpath+'*'))

    # Load the VGG19 Model and modifiy as per the requirement
    ## Load the VGG19 model
    print(f" ##> Loading VGG19 model without the top layer.\n");
    vggModel = VGG19(input_shape=imgSize, weights='imagenet', include_top=False)
    ## Mark all the layers are non-trainable as we don't want to disturb the layers with our training data
    for layer in vggModel.layers: layer.trainable = False

    # Add final output layers to the model as per our requirement
    ## Attach FC Layer
    print(f" ##> Adding FC layer to the VGG19 model.\n");
    fcLayer = Flatten()(vggModel.output)
    ## Attach softmax
    print(f" ##> Adding SOFTMAX layer to the VGG19 model with {classCnt} nodes.\n");
    outputLayer = Dense(classCnt, activation='softmax')(fcLayer)

    # Create the model
    print(f" ##> Creating the custom VGG19 model.\n");
    vggModel = Model(inputs=vggModel.input, outputs=outputLayer)
    vggModel.summary()

    # Compile the model
    print(f" ##> Compiling the custom VGG19 model.\n");
    vggModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Get the dataset
    trainImgDataset = dataset['TRAINING_DATA_GENERATOR']
    testImgDataset = dataset['TESTING_DATA_GENERATOR']

    # Fit the model to the data
    print(f" ##> Fitting the custom VGG19 model to the dataset.\n");
    vggModelFit = vggModel.fit_generator(trainImgDataset, validation_data=testImgDataset, epochs=epochCount, steps_per_epoch=len(trainImgDataset), validation_steps=len(testImgDataset))
    return (vggModel, vggModelFit)

def plotLossAcc(modelFit):
    # loss
    print(f" ##> Plotting the LOSS for the model.\n");
    plt.plot(modelFit.history['loss'], label='traing loss')
    plt.plot(modelFit.history['val_loss'], label='actual loss')
    plt.legend()
    #plt.show()
    plt.savefig('LossPlot')

    # accuracies
    print(f" ##> Plotting the ACCURACY for the model.\n");
    plt.plot(modelFit.history['accuracy'], label='train accuracy')
    plt.plot(modelFit.history['val_accuracy'], label='value accuracy')
    plt.legend()
    #plt.show()
    plt.savefig('AccuracyPlot')

def saveModel(model, modelName='learnFacesModel_vgg19.h5'):
    print(f" ##> Saving the model {model} as {modelName}.\n");
    model.save(modelName)
#########---------------------------------------------#########


if __name__ == "__main__":
    testTrainDataset, class_indices = buildFaceImageGenerator()
    (model, modelFit) = buildVGG19Model(dataset = testTrainDataset)
    plotLossAcc(modelFit = modelFit)
    saveModel(model=model, modelName=modelNameToSave)
