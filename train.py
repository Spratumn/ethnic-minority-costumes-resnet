import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf


from model import create_model
from dataset import create_dataset
from resnet_config import *

# 训练函数接口
def train(logName, savefig=True, pretrain=False):
    logName += '-' + str(LEARNING_RATE) + '-batch-' + str(BATCHSIZE)
    logDir = os.path.join('./log', logName)
    checkpointDir = os.path.join(logDir, 'checkpoint')

    if not pretrain: 
        if os.path.exists(logDir):shutil.rmtree(logDir)
        os.mkdir(logDir)
        os.mkdir(checkpointDir)
        model = create_model(classNums=CLASSNUMS, flag=MODEL_FLAG)
    else:
        model = tf.keras.models.load_model(checkpointDir)

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    mode='max',
                                                    period=1)

    inputSize = (224, 224) if MODEL_FLAG == 0 else INPUTSIZE
    trainset, valset = create_dataset('./datasets', inputSize=inputSize, batchSize=BATCHSIZE, normaliz=True, preprocess=True)
    
    history = model.fit(trainset,
                        validation_data=valset,
                        epochs=EPOCHES,
                        shuffle=True,
                        callbacks=[checkpoint]
                        )
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
    
    epoch = 1
    with open(os.path.join(logDir, 'history.csv'), 'w') as f:
        f.write('epoch,loss,accuracy,val_loss,val_accuracy\n')
        for tloss, tacc, vloss, vacc in zip(loss, acc, val_loss, val_acc):
            f.write(f'{epoch},{tloss},{tacc},{vloss},{vacc}\n')
            epoch += 1
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
    # train('resnet-origin')
    # train('resnet-origin-dropout')
    # train('resnet-alter')