import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf


from model import create_model
from dataset import create_dataset
from resnet_config import *

# 训练函数接口
# logName：训练名称用来对不同的训练进行区分
# pretrain: 相同的训练名称，相同的配置参数， 加载已有的训练权重继续训练
def train(logName, savefig=True, pretrain=False):
    # 1.根据训练名称以及配置文件中的学习率等参数，设置并创建训练记录保存路径
    logName += '-' + str(LEARNING_RATE) + '-batch-' + str(BATCHSIZE)
    logDir = os.path.join('./log', logName)
    checkpointDir = os.path.join(logDir, 'checkpoint')
    
    # 2.加载已有的训练模型或创建新模型
    if not pretrain: 
        if os.path.exists(logDir):shutil.rmtree(logDir)
        os.mkdir(logDir)
        os.mkdir(checkpointDir)
        model = create_model(classNums=CLASSNUMS, flag=MODEL_FLAG)
    else:
        model = tf.keras.models.load_model(checkpointDir)
    
    # 3.设置优化器，使用Adam优化器，并加载配置文件中的学习率参数
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # 4.配置模型训练
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 5. 配置权重保存方式（保存验证集分数最高的模型）
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    mode='max',
                                                    period=1)
    # 6.设置并加载训练数据集与验证数据集
    inputSize = (224, 224) if MODEL_FLAG == 0 else INPUTSIZE
    trainset, valset = create_dataset('./datasets', inputSize=inputSize, batchSize=BATCHSIZE, normaliz=True, preprocess=True)
    
    # 7.执行训练
    history = model.fit(trainset,
                        validation_data=valset,
                        epochs=EPOCHES,
                        shuffle=True,
                        callbacks=[checkpoint]
                        )
    # 8.保存训练过程中的损失和精度信息
    save_history(history, logName, savefig=savefig, pretrain=pretrain)
    

def save_history(history, logName, savefig=True, pretrain=False):
    logDir = os.path.join('./log', logName)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if pretrain:
        with open(os.path.join(logDir, 'history.csv'), 'r') as f:
            oldLines = f.readlines()[1:]
        old_acc, old_val_acc, old_loss, old_val_loss = [], [], [], []
        for line in oldLines:
            _, tloss, tacc, vloss, vacc = line.rstrip('\n').split(',')
            old_acc.append(float(tacc))
            old_val_acc.append(float(vacc))
            old_loss.append(float(tloss))
            old_val_loss.append(float(vloss))
        acc = old_acc + acc
        val_acc = old_val_acc + val_acc
        loss = old_loss + loss
        val_loss = old_val_loss + val_loss
    # 保存训练中的精度数据和损失数据到本地文件
    epoch = 1
    with open(os.path.join(logDir, 'history.csv'), 'w') as f:
        f.write('epoch,loss,accuracy,val_loss,val_accuracy\n')
        for tloss, tacc, vloss, vacc in zip(loss, acc, val_loss, val_acc):
            f.write(f'{epoch},{tloss},{tacc},{vloss},{vacc}\n')
            epoch += 1
    # 绘制训练过程中的精度图与损失图
    if savefig:
        epochs_range = range(len(loss))
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(logDir, 'history.png'))


if __name__ == '__main__':
    pass
    # 对原版resnet进行训练
    # train('resnet-origin')

    # 对原版resnet使用dropout进行训练
    # train('resnet-origin-dropout')

    # 对调整后的resnet进行训练
    # train('resnet-alter')