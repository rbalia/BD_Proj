import sys

import numpy as np
import time
import cv2 as cv

from skimage import img_as_ubyte
from sklearn.model_selection import train_test_split

from modules.utils import utils_configs as conf
from modules.utils import utils_processing as uElab
from modules.uNet_Data_BUS import load_train_data, load_test_mask_data, load_test_data
import modules.uNet_Model as uNet_Model

from skimage import io
import matplotlib.pyplot as plt

import tensorflow as tf

import findspark

findspark.init()
from pyspark import SparkContext, SparkConf
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel

def getStandardizationParams(set):
    mean = np.mean(set)  # mean for data centering
    std = np.std(set)
    return mean, std


def generateStandardizatedSet(set, mean, std):
    set -= mean
    set /= std
    return set

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
        .setMaster('spark://ip-172-31-24-208.us-east-2.compute.internal:7077')

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