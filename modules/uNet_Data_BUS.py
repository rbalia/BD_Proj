from __future__ import print_function

import os
import numpy as np
import cv2 as cv

from skimage.io import imsave, imread
from skimage.transform import resize

from skimage import io
from skimage import img_as_ubyte, img_as_float
from skimage.transform import resize
from modules.utils import utils_configs as conf
import matplotlib.pyplot as plt


def create_train_data():
    #train_data_path = os.path.join("../"+ data_path, 'train')
    images = os.listdir("../"+ conf.setDir+conf.trnDir)
    total = len(images) - 1 #subtract the mask dir

    imgs = np.ndarray((total, conf.img_rows, conf.img_cols,conf.channels), dtype=np.float)
    imgs_mask = np.ndarray((total, conf.img_rows, conf.img_cols,conf.channels), dtype=np.float)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for fileNameFull in os.listdir("../"+ conf.setDir + conf.trnDir):
        if fileNameFull == "mask":
            continue

        fileName = fileNameFull.split(".")[0]
        fileExtension = fileNameFull.split(".")[1]

        if conf.channels == 3:
            img = imread(os.path.join("../"+ conf.setDir + conf.trnDir, fileName + "." + fileExtension))
            img_mask = imread(os.path.join("../"+ conf.setDir + conf.trnDir + conf.mskDir, fileName + "_mask." + fileExtension))
        else:
            img = imread(os.path.join("../"+ conf.setDir + conf.trnDir, fileName + "." + fileExtension), as_gray=True)
            img_mask = imread(os.path.join("../"+ conf.setDir + conf.trnDir + conf.mskDir, fileName + "_mask." + fileExtension), as_gray=True)

        #img = img_as_float(img / 255.)
        img = img_as_float(img / np.max(img))
        img_mask = img_as_float(img_mask / np.max(img_mask))

        img = resize(img, (conf.img_rows, conf.img_cols,conf.channels), preserve_range=True)
        img_mask = resize(img_mask, (conf.img_rows, conf.img_cols,conf.channels), preserve_range=True)

        img_mask[img_mask > 0.5] = 1.0
        img_mask[img_mask <= 0.5] = 0.0

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        print('Done: {0}/{1} images'.format(i+1, total))
        i += 1
    """
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(setDir + trnDir, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
    
    
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    """
    print('Loading done.')

    np.save("../" + conf.pkgDir + 'imgs_train.npy', imgs)
    np.save("../" + conf.pkgDir + 'imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')

def load_censored_dataset():
    imgs_train = np.load("binaryPkg_censored/" + 'imgs_train.npy')
    imgs_mask_train = np.load("binaryPkg_censored/" + 'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_train_data():
    import os
    #os.getcwd()
    #print(os.getcwd())
    imgs_train = np.load(conf.pkgDir + 'imgs_train.npy')
    imgs_mask_train = np.load(conf.pkgDir + 'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_train_data_large():
    imgs_train = np.load(conf.pkgDir + 'imgs_train_large.npy')
    imgs_mask_train = np.load(conf.pkgDir + 'imgs_mask_train_large.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    data_path = conf.setDir+conf.tstDir

    images = os.listdir("../"+ data_path)
    total = len(images)#-1  # subtract the mask dir

    imgs = np.ndarray((total, conf.img_rows, conf.img_cols, conf.channels), dtype=np.float)
    imgs_mask = np.ndarray((total, conf.img_rows, conf.img_cols, conf.channels), dtype=np.float)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for fileNameFull in os.listdir("../"+ data_path):
        if fileNameFull == "mask":
            continue

        img_id = i  # int(image_name.split('.')[0])
        fileName = fileNameFull.split(".")[0]
        fileExtension = fileNameFull.split(".")[1]

        if conf.channels == 3:
            img = imread(os.path.join("../"+ data_path, fileName + "." + fileExtension))
            img_mask = imread(os.path.join("../"+ data_path + conf.mskDir, fileName + "_mask." + fileExtension))
        else:
            img = imread(os.path.join("../"+ data_path, fileName + "." + fileExtension), as_gray=True)
            img_mask = imread(os.path.join("../"+ data_path + conf.mskDir, fileName + "_mask." + fileExtension),as_gray=True)

        img = img_as_float(img / np.max(img))
        img_mask = img_as_float(img_mask / np.max(img_mask))

        img = resize(img, (conf.img_rows, conf.img_cols,conf.channels), preserve_range=True)
        img_mask = resize(img_mask, (conf.img_rows, conf.img_cols, conf.channels), preserve_range=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        print('Done: {0}/{1} images'.format(i+1, total))
        i += 1
    print('Loading done.')

    np.save("../" + conf.pkgDir + 'imgs_test.npy', imgs)
    np.save("../" + conf.pkgDir + 'imgs_mask_test.npy', imgs_mask)
    np.save("../" + conf.pkgDir + 'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load(conf.pkgDir + 'imgs_test.npy')
    imgs_id = np.load(conf.pkgDir + 'imgs_id_test.npy')
    return imgs_test, imgs_id

def load_test_data_large():
    imgs_test = np.load(conf.pkgDir + 'imgs_test_large.npy')
    imgs_id = np.load(conf.pkgDir + 'imgs_id_test_large.npy')
    return imgs_test, imgs_id

def load_test_mask_data():
    imgs_test = np.load(conf.pkgDir + 'imgs_mask_test.npy')
    return imgs_test

def load_test_mask_data_large():
    imgs_test = np.load(conf.pkgDir + 'imgs_mask_test_large.npy')
    return imgs_test

def load_pred_data():
    imgs_test = np.load(conf.pkgDir + 'imgs_prediction.npy')
    return imgs_test

if __name__ == '__main__':
    create_train_data()
    create_test_data()


    """
    imgs_train, b = load_train_data()
    imgs_test, _ = load_test_data()
    imgs_true = load_test_mask_data()
    imgs_pred = load_pred_data()

    for img in imgs_test:

        #img = uElab.preprocess(img)
        #img = img_as_ubyte(img) * 255.
        img = img_as_ubyte(img)
        io.imshow(img)
        plt.show()
    """
