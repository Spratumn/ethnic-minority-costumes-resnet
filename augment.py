import os
import cv2
import random
import numpy as np
import tqdm


class Augmenter(object):
    def __init__(self):
        self.augmenters = [random_crop, random_flip, random_rotate, random_shelter]

    def __call__(self, image):
        n = random.randint(1, 4)
        augs = []
        for _ in range(n):
            k = random.randint(0, len(self.augmenters)-1)
            if k in augs: continue
            augs.append(k)
            image = self.augmenters[k](image)
        return image



def random_crop(image, randomScale=True):
    scale = random.randint(5, 9) / 10 if randomScale else 0.6
    H, W = image.shape[:2]
    h, w = int(H * scale), int(W * scale)
    hStart = random.randint(0, int(H*(1-scale)))
    wStart = random.randint(0, int(W*(1-scale)))

    out = image[hStart:hStart+h, wStart:wStart+w, :]
    return out


def random_flip(image):
    image = image[:, ::-1, :]
    return image

def random_rotate(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.randint(30, 50)
    if random.randint(0, 1): angle += -1
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def random_noise(image):
    def clamp(pv):
        if pv > 255:
            return 255
        elif pv < 0:
            return 0
        else:
            return pv
    #给图片增加高斯噪声，计算花费很长时间
    h, w = image.shape[:2]
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            #获取每个像素点的bgr值
            b = image[row, col, 0]  #blue
            g = image[row, col, 1]  #green
            r = image[row, col, 2]  #red\
            #给每个像素值设置新的bgr值
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    return image

def random_shelter(image):
    h, w = image.shape[:2]
    vmin, vmax = int(min(h, w) * 0.1), int(min(h, w) * 0.2)
    if h - vmax < 10 or w - vmax < 10:return image
    nx, ny = random.randint(1, 7), random.randint(1, 7)
    xs, ys = [], []
    for _ in range(nx):
        x1 = random.randint(0, w-vmax-2)
        x2 = x1+random.randint(vmin, vmax)
        xs.append([x1, x2])
    for _ in range(ny):
        y1 = random.randint(0, h-vmax-2)
        y2 = y1+random.randint(vmin, vmax)
        ys.append([y1, y2])
    random.shuffle(xs)
    random.shuffle(ys)
    n = min(len(xs), len(ys))
    for i in range(n):
        x1, x2 = xs[i]
        y1, y2 = ys[i]
        image[y1:y2, x1:x2, :] = [0, 0, 0]
    return image


def generate_augment_images(trainsetDir, n=5):
    augmenter = Augmenter()
    folders = os.listdir(trainsetDir)
    for folder in folders:
        folderDir = os.path.join(trainsetDir, folder)
        imagenames = os.listdir(folderDir)
        for imagename in tqdm.tqdm(imagenames):
            imagepath = os.path.join(folderDir, imagename)
            filename, suffix = imagename.split('.')
            image = cv2.imread(imagepath)
            for i in range(n):
                augmentImage = augmenter(image)
                cv2.imwrite(os.path.join(folderDir, f'{filename}_{i}.{suffix}'), augmentImage)


def clean_augment_images(trainsetDir):
    folders = os.listdir(trainsetDir)
    for folder in folders:
        folderDir = os.path.join(trainsetDir, folder)
        imagenames = os.listdir(folderDir)
        for imagename in imagenames:
            if imagename.split('.')[0].endswith(('_0', '_1', '_2', '_3', '_4')):
                os.remove(os.path.join(folderDir, imagename))


if __name__ == '__main__':
    pass
    # generate_augment_images(trainsetDir='datasets/trainset', n=4)
    # clean_augment_images(trainsetDir='datasets/trainset')

