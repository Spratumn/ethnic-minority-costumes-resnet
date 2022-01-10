import os
import random
import tensorflow as tf
from tensorflow.keras import layers

# 创建数据集用于模型的训练与评估
def create_dataset(datasetDir, inputSize=(224, 224), batchSize=8, normaliz=False, preprocess=None):
    """
    datasetDir: 数据集的路径，默认为'./datasets'
    inputSize: 模型输入尺寸（w，h）
    batchSize: 模型的训练中一个batch的大小
    normaliz: 是否对图像进行归一化处理，训练中默认开启
    preprocess: 是否对图像进行预处理操作，训练中默认开启
    """
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


# 使用tensorflow中的图像预处理功能进行进一步数据增强操作
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









    
    
    

