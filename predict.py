import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt



class Predictor:
    def __init__(self, checkpoint, inputSize, layernames=None):
        self.model = tf.keras.models.load_model(checkpoint)
        self.inputSize = inputSize
        self.catemap = {0: 'manzu', 1: 'mengguzu', 2: 'miaozu', 3: 'yaozu', 4: 'zhuangzu'}
        
        self.layernames = layernames
        if layernames is not None:
            
            outputs = [self.model.get_layer(layername).output for layername in layernames]
            self.featureModel = keras.models.Model(inputs=self.model.input,outputs=outputs)
    
    def predict(self, image, toLable=True):
        h, w = image.shape[:2]
        if h != self.inputSize[1] or w != self.inputSize[0]:
            image = cv2.resize(image, self.inputSize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = self.model.predict(np.array([image], dtype=np.float) / 255.0)
        return self.catemap[np.argmax(predictions[0])] if toLable else np.argmax(predictions[0])

    def draw_features(self, image, saveDir, nw=8, nh=8):
        assert self.layernames is not None
        h, w = image.shape[:2]
        if h != self.inputSize[1] or w != self.inputSize[0]:
            image = cv2.resize(image, self.inputSize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = self.featureModel.predict(np.array([image], dtype=np.float) / 255.0)
        for layername,feature in zip(self.layernames, features):
            sizeh, sizew = feature.shape[1:3]
            displayGrid = np.zeros((sizeh * nw, nh * sizew))
            channelIdxes = random.choices(range(feature.shape[-1]), k=nw*nh)
            for col in range(nw):
                for row in range(nh):
                    channel_image = feature[0,:,:,channelIdxes[col * nh + row]]
                    displayGrid[col * sizeh : (col + 1) * sizeh,row * sizew: (row + 1) * sizew] = channel_image
                
            scaleh = 1. / sizeh
            scalew = 1. / sizew
            plt.figure(figsize=(scaleh * displayGrid.shape[1],scalew * displayGrid.shape[0]))
            plt.title(layername)
            plt.grid(False)
            plt.imshow(displayGrid,aspect='auto',cmap='viridis')
            savePath = os.path.join(saveDir, f'{layername}.png')
            plt.savefig(savePath)
            cv2.imwrite(os.path.join(saveDir, 'input_image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    catemap = {'manzu': 0, 'mengguzu': 1, 'miaozu': 2, 'yaozu': 3, 'zhuangzu': 4}
    predictor = Predictor(checkpoint='log/resnet-origin-0.001-batch-32/checkpoint', 
                          inputSize=(224, 224), 
                          layernames=['conv1_conv2', 'conv2_conv2', 'conv4_block2_conv3'])
    
    datasetDir = './datasets/trainset'
    folders = os.listdir(datasetDir)
    for folder in folders:
        gt = catemap[folder]
        folderDir = os.path.join(datasetDir, folder)
        imagenames = os.listdir(folderDir)
        for imagename in imagenames:
            image = cv2.imread(os.path.join(folderDir, imagename))
            
            pred = predictor.predict(image, toLable=False)
            print(f'{imagename} gt: {gt}, pred: {pred}')
    
    

    
    


