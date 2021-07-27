import os
import sys

import numpy as np
import shutil
import time
import cv2 as cv
from skimage import img_as_ubyte
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt

from modules.utils import utils_configs as conf
from modules.utils import utils_processing as uElab
from modules.uNet_Data_BUS import load_train_data, load_test_mask_data, load_test_data
import modules.uNet_Model as uNet_Model

import tensorflow as tf
import findspark
findspark.init()

from pyspark import SparkContext, SparkConf

from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel


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
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')#, num_workers=num_workers)
    spark_model.fit(rdd_train, epochs=epochs, batch_size=8, verbose=1, validation_split=0.2)

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

    return np.average(scoresIOU)




if __name__ == '__main__':
    # PARAMS INITIALIZATION ============================================================================================
    modelName = "uNet_BD"
    csvFileName = "scores_resize_monitor_3"

    print("-" * 60)
    print("Spark + Tensorflow : Distributed Training : Script Starting...")
    print("-" * 60)

    # PREPROCESSING ====================================================================================================
    # Create the Spark Context - more infos at https://stackoverflow.com/questions/32356143/what-does-setmaster-local-mean-in-spark
    sConf = SparkConf().setAppName('BigData_DistributedTraining').setMaster('local[3]')
    """sConf = SparkConf().setAppName('BigData_DistributedTraining') \
        .setMaster('spark://ip-172-31-24-208.us-east-2.compute.internal:7077') \
        .set("spark.default.parallelism", sys.argv[2])\
        .set("spark.network.timeout", "10000000")\
        .set("spark.executor.heartbeatInterval", "1000000")\
        .set("spark.worker.cleanup.enabled", "true") \
        .set("spark.worker.cleanup.interval", "300") \
        .set("spark.worker.cleanup.appDataTtl", "300")"""


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


    # DISTRIBUTE TRAINING / PREDICT / SAVE =============================================================================
    # Training : Get model, compile, generate SparkModel, fit
    start_train = time.time()
    spark_model = uNet_SparkTrain(sc, train_img_stdz, train_msk, epochs=int(sys.argv[3]),saveFlag=False)
    end_train = time.time()
    TE_Training = round(end_train - start_train,3)
    print("Time Elapsed : Training : " + str(TE_Training) + " seconds")

    # Prediction
    predictions = uNet_SparkPredict(spark_model, test_img_stdz, printPlot=False)
    avgIOU = getAverageIOU(test_msk, predictions)

    print("-" * 60)
    print("Spark + Tensorflow : Distributed Training : Results")
    print("Time Elapsed during Training.............: " + str(TE_Training) + " seconds")
    print("Average IOU Score Achieved on TestSet....: " + str(avgIOU))
    print("-" * 60)

    # Save SparkModel (Delete if already exist)
    modelFolder = "models/" + sys.argv[1]
    if os.path.exists(modelFolder):
        shutil.rmtree(modelFolder, ignore_errors=True)
    spark_model.save(modelFolder)
