CLASSNUMS = 5              # 分类个数
MODEL_FLAG = 1             # 0: 使用原始resnet， 1:使用调整后的resnet
INPUTSIZE = (224, 320)     # 调整后的resnet输入尺寸
BATCHSIZE = 32             # 训练batch参数
EPOCHES = 100              # 训练迭代次数
LEARNING_RATE = 0.0005     # 学习率参数
DROPOUT = 0.0              # dropout参数 大于零时启用dropout，等于零时关闭dropout