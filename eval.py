from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# 本文件与svm部分的eval.py基本相同

def plot_confusion_matrix(preds, gts, savePath, labels=None):
    cm = confusion_matrix(gts, preds)
    plt.matshow(cm, cmap=plt.cm.Reds)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xticks(range(len(labels)), labels)
    plt.xlabel('Predicted label')
    plt.yticks(range(len(labels)), labels)
    plt.savefig(savePath)


def plot_prediction(predDict, savePath):
    print(predDict)
    ids = []
    labels = []
    true = []
    false = []
    for id in predDict.keys():
        if id not in ids:
            ids.append(id)
            labels.append(predDict[id][0])
            false.append(predDict[id][1] - predDict[id][2])
            true.append(predDict[id][2])
    
    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(labels, true, width, label='true')
    ax.bar(labels, false, width, bottom=true, label='false')

    ax.set_ylabel('prediction')
    ax.set_title('prediction of resnet')
    ax.legend()
    plt.savefig(savePath)


