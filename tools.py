import os
import cv2
import time
import matplotlib.pyplot as plt

from eval import plot_confusion_matrix, plot_prediction
from resnet_config import *
from predict import Predictor

# 运行测试评价，获得混淆矩阵和预测图
def run_eval(datasetDir, checkpoint):
    
    inputSize = (224, 224) if MODEL_FLAG == 0 else INPUTSIZE
    predictor = Predictor(checkpoint=checkpoint, inputSize=inputSize)
    
    catemap = {'manzu': 0, 'mengguzu': 1, 'miaozu': 2, 'yaozu': 3, 'zhuangzu': 4}
    classLabels = ['manzu', 'mengguzu', 'miaozu', 'yaozu', 'zhuangzu']

    trainPredDict = {0: ['manzu', 0, 0], 1: ['mengguzu', 0, 0], 2: ['miaozu', 0, 0], 3: ['yaozu', 0, 0], 4: ['zhuangzu', 0, 0]}
    trainPreds, trainGts = [], []
    trainsetDir = os.path.join(datasetDir, 'trainset')
    folders = os.listdir(trainsetDir)
    start = time.time()
    count = 0
    for folder in folders:
        gt = catemap[folder]
        folderDir = os.path.join(trainsetDir, folder)
        imagenames = os.listdir(folderDir)
        for imagename in imagenames:
            if imagename.split('.')[0].endswith(('_0', '_1', '_2', '_3')):continue
            image = cv2.imread(os.path.join(folderDir, imagename))
            count += 1
            pred = predictor.predict(image, toLable=False)
            trainPreds.append(pred)
            trainGts.append(gt)
            trainPredDict[gt][1] += 1
            if pred == gt:trainPredDict[gt][2] += 1
    print(f'mean pred time cost: {(time.time() - start) / count}')
    plot_prediction(trainPredDict, savePath='./results/train-prediction.png')
    plot_confusion_matrix(trainPreds, trainGts, labels=classLabels, savePath='./results/train-cm.png')

    testPredDict = {0: ['manzu', 0, 0], 1: ['mengguzu', 0, 0], 2: ['miaozu', 0, 0], 3: ['yaozu', 0, 0], 4: ['zhuangzu', 0, 0]}
    testPreds, testGts = [], []
    testsetDir = os.path.join(datasetDir, 'testset')
    folders = os.listdir(testsetDir)
    for folder in folders:
        gt = catemap[folder]
        folderDir = os.path.join(testsetDir, folder)
        imagenames = os.listdir(folderDir)
        for imagename in imagenames:
            image = cv2.imread(os.path.join(folderDir, imagename))
            pred = predictor.predict(image, toLable=False)
            testPreds.append(pred)
            testGts.append(gt)
            testPredDict[gt][1] += 1
            if pred == gt:testPredDict[gt][2] += 1

    plot_prediction(testPredDict, savePath='./results/test-prediction.png')
    plot_confusion_matrix(testPreds, testGts, labels=classLabels, savePath='./results/test-cm.png')


# 绘制卷积特征图
def draw_convlayer_features(imagepath, checkpoint):
    inputSize = (224, 224) if MODEL_FLAG == 0 else INPUTSIZE
    predictor = Predictor(checkpoint=checkpoint, 
                          inputSize=inputSize, 
                          layernames=['conv1_conv2', 'conv2_conv2', 'conv4_block2_conv3'])
    image = cv2.imread(imagepath)
    predictor.draw_features(image, './features', nw=6, nh=5)


# 对比不同的训练过程，并绘制损失和准确率曲线
def draw_diff_loss_acc(logs, savepath):
    train_losses = {}
    train_accs = {}
    test_losses = {}
    test_accs = {}

    for logDir in logs:
        logname = os.path.basename(logDir)
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        with open(os.path.join(logDir, 'history.csv'), 'r') as logfile:
            lines = logfile.readlines()[1:]
        for line in lines:
            _, tloss, tacc, vloss, vacc = line.rstrip('\n').split(',')
            train_acc.append(float(tacc))
            test_acc.append(float(vacc))
            train_loss.append(float(tloss))
            test_loss.append(float(vloss))
        
        train_losses[logname] = train_loss
        train_accs[logname] = train_acc
        test_losses[logname] = test_loss
        test_accs[logname] = test_acc

    plt.figure(figsize=(16, 16))
    # train loss
    plt.subplot(2, 2, 1)
    for logname in train_losses.keys():
        train_loss_ = train_losses[logname]
        plt.plot(range(len(train_loss_)), train_loss_, label=logname)
    plt.legend(loc='upper right')
    plt.title('Training loss')

    # train acc
    plt.subplot(2, 2, 2)
    for logname in train_accs.keys():
        train_acc_ = train_accs[logname]
        plt.plot(range(len(train_acc_)), train_acc_, label=logname)
    plt.legend(loc='lower right')
    plt.title('Training acc')

    # test loss
    plt.subplot(2, 2, 3)
    for logname in test_losses.keys():
        test_loss_ = test_losses[logname]
        plt.plot(range(len(test_loss_)), test_loss_, label=logname)
    plt.legend(loc='upper right')
    plt.title('test loss')

    # test acc
    plt.subplot(2, 2, 4)
    for logname in test_accs.keys():
        test_acc_ = test_accs[logname]
        plt.plot(range(len(test_acc_)), test_acc_, label=logname)
    plt.legend(loc='lower right')
    plt.title('test acc')
    
    plt.savefig(savepath)

# 绘制损失和准确率曲线
def draw_diff_train_test(logDir, savepath):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    with open(os.path.join(logDir, 'history.csv'), 'r') as logfile:
        lines = logfile.readlines()[1:]
    for line in lines:
        _, tloss, tacc, vloss, vacc = line.rstrip('\n').split(',')
        train_acc.append(float(tacc))
        test_acc.append(float(vacc))
        train_loss.append(float(tloss))
        test_loss.append(float(vloss))
    
    plt.figure(figsize=(16, 16))
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss)), train_loss, label='train')
    plt.plot(range(len(test_loss)), test_loss, label='test')
    plt.legend(loc='upper right')
    plt.title('loss')

    # acc
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_acc)), train_acc, label='train')
    plt.plot(range(len(test_acc)), test_acc, label='test')
    plt.legend(loc='lower right')
    plt.title('acc')

    plt.savefig(savepath)

if __name__ == '__main__':
    pass

    # 运行测试评价，获得混淆矩阵和预测图
    # run_eval(datasetDir='./datasets', checkpoint='log/resnet-alter-0.0005-batch-32/checkpoint')
    
    
    # 绘制卷积特征图
    draw_convlayer_features(imagepath='datasets/trainset/manzu/0000015.jpg', 
                            checkpoint='log/resnet-alter-0.0005-batch-32/checkpoint')
    
    # 对比不同的训练过程，并绘制损失和准确率曲线
    # draw_diff_loss_acc(logs=[
    #                          'log/resnet-origin-0.0005-batch-32',
    #                         #  'log/resnet-origin-0.001-batch-32',
    #                         #  'log/resnet-origin-0.0015-batch-32',
    #                         'log/resnet-alter-0.0005-batch-32',
    #                          ], 
    #                    savepath='./results/resnet-alter-diff.png')
    
    # 绘制损失和准确率曲线
    # draw_diff_train_test(logDir='log/resnet-alter-0.0005-batch-32', 
    #                      savepath='./results/resnet-alter-plot.png')