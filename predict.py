import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# 加载模型权重创建预测器
class Predictor:
    def __init__(self, checkpoint, inputSize, layernames=None):
        """
        checkpoint: 模型权重保存路径
        inputSize: 模型输入尺寸
        layernames: 用于绘制卷积层特征的卷积层名称列表
        """
        self.model = tf.keras.models.load_model(checkpoint)
        self.inputSize = inputSize
        self.catemap = {0: 'manzu', 1: 'mengguzu', 2: 'miaozu', 3: 'yaozu', 4: 'zhuangzu'}
        
        self.layernames = layernames
        if layernames is not None:
            outputs = [self.model.get_layer(layername).output for layername in layernames]
            self.featureModel = keras.models.Model(inputs=self.model.input,outputs=outputs)
    
    # 对给定的图像进行预测，输出预测结果（当toLable为True时输出字符串，否则输出0-4的id）
    def predict(self, image, toLable=True):
        h, w = image.shape[:2]
        if h != self.inputSize[1] or w != self.inputSize[0]:
            image = cv2.resize(image, self.inputSize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = self.model.predict(np.array([image], dtype=np.float) / 255.0)
        return self.catemap[np.argmax(predictions[0])] if toLable else np.argmax(predictions[0])
    
    # 绘制卷积特征图
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
