import os
import sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import shutil
import time
import pandas as pd
import csv
import cv2 as cv
import tensorflow as tf
from elephas.hyperparam import HyperParamModel

from skimage import img_as_ubyte, img_as_float
from sklearn.model_selection import train_test_split

#from modules.utils import utils_segmentation as uSgm
from modules.utils import utils_configs as conf
from modules.utils import utils_processing as uElab
from modules.utils import utils_print
from modules.uNet_Data_BUS import load_train_data, load_test_mask_data, load_test_data, load_censored_dataset, \
    load_train_data_large, load_test_data_large, load_test_mask_data_large
import modules.uNet_Model as uNet_Model
#from uNet_Predict import predict
#from uNet_Train_GPU_Spark import train
#import uNet_Model_SE

#from main_2 import handcraftedSegmentation
#from classification_single import classification
#from uNet_Data_Aug import load_augmented_train_data
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Convolution2D, \
    UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, UpSampling2D
from tensorflow.python.keras import Model
# from keras.layers.merge import concatenate
# from tensorflow import concat as concatenate
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

import findspark

findspark.init()
from pyspark import SparkContext, SparkConf

import elephas
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from elephas.spark_model import load_spark_model


def splitData(images, masks):
    list = []
    for i, j in zip(images, masks):
        list.append((i, j))

    train_set, test_set = train_test_split(list, test_size=0.1, shuffle=True)
    train_set = np.array(train_set)
    test_set = np.array(test_set)

    train_img = np.ndarray((train_set.shape[0], train_set.shape[2], train_set.shape[3], train_set.shape[4]),
                           dtype=np.float)
    train_msk = np.ndarray((train_set.shape[0], train_set.shape[2], train_set.shape[3], train_set.shape[4]),
                           dtype=np.float)
    test_img = np.ndarray((test_set.shape[0], test_set.shape[2], test_set.shape[3], test_set.shape[4]), dtype=np.float)
    test_msk = np.ndarray((test_set.shape[0], test_set.shape[2], test_set.shape[3], test_set.shape[4]), dtype=np.float)
    test_ids = np.ndarray((test_set.shape[0],), dtype=np.int32)

    i = 0
    for pair in train_set:
        train_img[i] = pair[0]
        train_msk[i] = pair[1]
        i += 1

    i = 0
    for pair in test_set:
        test_img[i] = pair[0]
        test_msk[i] = pair[1]
        test_ids[i] = i
        i += 1

    return train_img, train_msk, test_img, test_msk, test_ids


def getStandardizationParams(set):
    mean = np.mean(set)  # mean for data centering
    std = np.std(set)
    return mean, std


def generateStandardizatedSet(set, mean, std):
    set -= mean
    set /= std
    return set

def uNet_SparkTrain(sparkContext, X_train, Y_train, num_workers=1, epochs=50, saveFlag=False, saveDir="model/spark"):
    # Convert train-set to RDD
    print("Converting numpy arrays to rdd...")
    rdd_train = to_simple_rdd(sparkContext, X_train, Y_train)

    # Generate and compile the uNet model
    print("Generating and compiling model...")
    model = uNet_Model.getModel(conf.img_rows, conf.img_cols)
    model.compile(metrics=[uNet_Model.dice_coef], loss="binary_crossentropy", optimizer="adam")
    model.compiled_metrics = [uNet_Model.dice_coef]
    model.compiled_metrics._metrics = [uNet_Model.dice_coef]

    # Train model
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous', num_workers=num_workers)
    spark_model.fit(rdd_train, epochs=epochs, batch_size=8, verbose=1, validation_split=0.1)

    if saveFlag:
        spark_model.save(saveDir)

    return spark_model


def uNet_SparkPredict(spark_model, X_test, printPlot=True):
    print('-' * 30)
    print('Generate Predictions')
    print('-' * 30)
    predictions = spark_model.predict(X_test)

    if printPlot:
        for pred in predictions:
            io.imshow(pred)
            plt.show()

    return predictions


def uNet_LoadSparkModel(modelDir):
    # load uncompiled model
    model = tf.keras.models.load_model(modelDir, compile=False)

    # compile with custom metrics
    model.compile(metrics=[uNet_Model.dice_coef], loss="binary_crossentropy", optimizer="adam")
    model.compiled_metrics = [uNet_Model.dice_coef]
    model.compiled_metrics._metrics = [uNet_Model.dice_coef]

    # convert to SparkModel
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')#, num_workers=1)

    return spark_model

def getAverageIOU(truth_set, preds_set):

    scoresIOU = []
    for truth, pred in zip(truth_set, preds_set):
        pred255 = img_as_ubyte(pred)
        _, predThres = cv.threshold(pred255, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        predNorm = predThres / np.max(predThres)

        score = uElab.evaluateIOU(predNorm, truth)
        scoresIOU.append(score)
        print(score)

    return np.average(scoresIOU)




if __name__ == '__main__':
    # PARAMS INITIALIZATION ============================================================================================
    modelName = "uNet_BD"
    csvFileName = "scores_resize_monitor_3"

    print("-" * 60)
    print("Spark + Tensorflow : Distributed Prediction : Script Starting...")
    print("-" * 60)

    # PREPROCESSING ====================================================================================================
    # Create the Spark Context - more infos at https://stackoverflow.com/questions/32356143/what-does-setmaster-local-mean-in-spark
    sConf = SparkConf().setAppName('BigData_DistributedPredict')\
        .setMaster('spark://ip-172-31-24-208.us-east-2.compute.internal:7077')\
        .set("spark.network.timeout",           "10000001")\
        .set("spark.executor.heartbeatInterval","10000000")
        #.set("spark.default.parallelism", sys.argv[2])
    #sConf = SparkConf().setAppName('BigData_DistributedTraining').setMaster('spark://riccardo-VirtualBox:7077')
    sc = SparkContext(conf=sConf)
    sc.addPyFile("modules.zip")

    # Load data : BUS DATASET
    print("Loading dataset...")
    train_img, train_msk = load_train_data()
    test_img, test_ids = load_test_data()
    test_msk = load_test_mask_data()
    print("TrainSet Shape...: " + str(train_img.shape))
    print("TestSet Shape....: " + str(test_img.shape))

    # Standardization
    print("Image standardization...")
    mean, std = getStandardizationParams(np.concatenate((train_img, test_img)))
    train_img_stdz = generateStandardizatedSet(train_img, mean, std)
    test_img_stdz = generateStandardizatedSet(test_img, mean, std)



    # LOAD / PREDICT ===================================================================================================
    # Load SparkModel
    spark_model2 = uNet_LoadSparkModel("models/" + sys.argv[1]) # <path>/<path>

    # Predict
    start_pred = time.time()
    predictions2 = uNet_SparkPredict(spark_model2, test_img_stdz, printPlot=False)
    end_pred = time.time()
    print("Time Elapsed : Predictions : " + str(round(end_pred - start_pred, 3)) + " seconds")

    # Evaluate IOU Score
    avgIOU = getAverageIOU(test_msk, predictions2)
    print("Average IOU: " + str(avgIOU))

    #for img, mask, pred in zip(test_img, test_msk, predictions2):
    #    utils_print.printBrief3Cells("title", ["1","2","3"], [img, mask, pred])


    # HYPER-PARAMETER OPTIMIZATION =====================================================================================

    # Define hyper-parameter model and run optimization.
    #hyperparam_model = HyperParamModel(sc)
    #hyperparam_model.minimize(model=model, data=data, max_evals=5)

    # https://github.com/maxpumperla/elephas/blob/master/examples/hyperparam_optimization.py

    # TODO
    # X - Esplorare le opzioni "Estimate"
    # Y - Implementare HyperParameter Optimization

    # TODO Future
    # Creare Cluster / Verificare Cluster Educate

