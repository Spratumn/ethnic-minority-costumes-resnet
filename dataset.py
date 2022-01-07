import os
import random
import tensorflow as tf
from tensorflow.keras import layers


def create_dataset(datasetDir, inputSize=(224, 224), batchSize=8, normaliz=False, preprocess=None):
    trainset = tf.keras.preprocessing.image_dataset_from_directory(os.path.join(datasetDir, 'trainset'),
                                                            labels='inferred',
                                                            label_mode='int',
                                                            shuffle=True,
                                                            image_size=(inputSize[1], inputSize[0]),
                                                            batch_size=batchSize)

    valset = tf.keras.preprocessing.image_dataset_from_directory(os.path.join(datasetDir, 'testset'),
                                                            labels='inferred',
                                                            label_mode='int',
                                                            shuffle=False,
                                                            image_size=(inputSize[1], inputSize[0]),
                                                            batch_size=1)
    
    if preprocess:
        trainset = trainset.map(lambda x, y: (transformer(x), y))
    
    if normaliz:
        normalLayer = layers.experimental.preprocessing.Rescaling(1./255)
        trainset = trainset.map(lambda x, y: (normalLayer(x), y))
        valset = valset.map(lambda x, y: (normalLayer(x), y))
    
    
    return trainset, valset


def transformer(image):
    selectedId = random.randint(0, 2)

    if selectedId == 1:
        return tf.image.adjust_gamma(image, min(0.3, random.random()))
    elif selectedId == 2:
        delta = random.random() 
        if random.randint(0, 1): delta *= -1
        return tf.image.adjust_hue(image, delta)
    else:
        return image









    
    
    

