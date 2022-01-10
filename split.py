import os
import cv2
import random
import shutil

# 从svm部分的数据集中创建resnet使用的数据集（确保两个部分训练集和测试集完全相同）
def split_dataset(originDatasetDir, datasetRootDir):
    # originDatasetDir：svm部分的数据集路径
    
    catemap = {0: 'manzu',
                1: 'mengguzu',
                2: 'miaozu',
                3: 'yaozu',
                4: 'zhuangzu',}

    
    with open(os.path.join(originDatasetDir, 'annotations.csv'), 'r') as f:
        lines = f.readlines()

    annoList = []
    for line in lines:
        imagePath, id = line.rstrip('\n').split(',')
        annoList.append((imagePath, int(id)))

    split = int(len(annoList) * 0.7)

    random.seed(0)
    random.shuffle(annoList)

    trainSet = annoList[:split]
    testSet = annoList[split:]
    
    datasetDir = os.path.join(datasetRootDir, 'trainset') 
    if not os.path.exists(datasetDir): os.mkdir(datasetDir)  
    for imagename, cateid in trainSet:
        oriPath = originDatasetDir + imagename
        filename = imagename.split('/')[1]
        catename = catemap[int(cateid)]
        dstDir = os.path.join(datasetDir, catename)
        if not os.path.exists(dstDir):os.mkdir(dstDir)
        
        try:
            cv2.imread(oriPath).shape
            shutil.copyfile(oriPath, os.path.join(dstDir, filename))
        except InterruptedError:
            break
        except AttributeError:
            pass
    datasetDir = os.path.join(datasetRootDir, 'testset')
    if not os.path.exists(datasetDir): os.mkdir(datasetDir)        
    for imagename, cateid in testSet:
        oriPath = originDatasetDir + imagename
        filename = imagename.split('/')[1]
        catename = catemap[int(cateid)]
        dstDir = os.path.join(datasetDir, catename)
        if not os.path.exists(dstDir):os.mkdir(dstDir)
        
        try:
            cv2.imread(oriPath).shape
            shutil.copyfile(oriPath, os.path.join(dstDir, filename))
        except InterruptedError:
            break
        except AttributeError:
            pass

if __name__ == '__main__':
    pass
    # split_dataset(originDatasetDir='./ori/datasets/', datasetRootDir='./datasets')