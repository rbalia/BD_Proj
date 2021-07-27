import matplotlib.pyplot as plt
import cv2 as cv
import os
from skimage.transform import resize
import numpy as np


def printSlicBrief(original, boundaries_openCV, boundaries_Skimage, AVG_openCV, AVG_Skimage, ):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Superpixel Segmentantation")
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(original)
    ax1.set_title('Original Image')

    ax2.imshow(boundaries_openCV)
    ax2.set_title('SLIC Segmentation (OpenCV)')

    ax3.imshow(AVG_openCV)
    ax3.set_title('AverageColor - (OpenCV)')

    ax4.imshow(original)
    ax4.set_title('Original Image')

    ax5.imshow(boundaries_Skimage)
    ax5.set_title('SLIC Segmentation (Skimage)')

    ax6.imshow(AVG_Skimage)
    ax6.set_title('AverageColor - (Skimage)')

    plt.show()


def printProcessingBrief(slicSegm, EdgeMap, newSlicSegm, labels, edgelessLabels, bordersEnhance):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Processing Phase")
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(slicSegm)
    ax1.set_title('Slic Segmentation')

    ax2.imshow(EdgeMap)
    ax2.set_title('Edge Map')

    ax3.imshow(newSlicSegm)
    ax3.set_title('Edgeless Segmentation')

    ax4.imshow(labels, cmap="gray")
    ax4.set_title('Labels')

    ax5.imshow(edgelessLabels, cmap="gray")
    ax5.set_title('Edgeless Labels')

    ax6.imshow(bordersEnhance)
    ax6.set_title('New Segments Borders')

    plt.show()


def printBrief3Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    plt.show()


def printBrief4Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    plt.show()


def printBrief6Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(images[0], cmap="gray")
    ax1.set_title(names[0])

    ax2.imshow(images[1], cmap="gray")
    ax2.set_title(names[1])

    ax3.imshow(images[2], cmap="gray")
    ax3.set_title(names[2])

    ax4.imshow(images[3], cmap="gray")
    ax4.set_title(names[3])

    ax5.imshow(images[4], cmap="gray")
    ax5.set_title(names[4])

    ax6.imshow(images[5], cmap="gray")
    ax6.set_title(names[5])

    plt.show()


def printBrief8Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    ax5.imshow(images[4])
    ax5.set_title(names[4])

    ax6.imshow(images[5])
    ax6.set_title(names[5])

    ax7.imshow(images[6])
    ax7.set_title(names[6])

    ax8.imshow(images[7])
    ax8.set_title(names[7])

    plt.show()


def printBrief9Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    ax5.imshow(images[4])
    ax5.set_title(names[4])

    ax6.imshow(images[5])
    ax6.set_title(names[5])

    ax7.imshow(images[6])
    ax7.set_title(names[6])

    ax8.imshow(images[7])
    ax8.set_title(names[7])

    ax9.imshow(images[8])
    ax9.set_title(names[8])

    plt.show()


def printRoiBrief(roi, clinicalMask, clinicalRoi, highlightedRegions):
    # Display result
    fig2, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Processing Phase")
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(roi)
    ax1.set_title('Detected ROI')

    ax2.imshow(clinicalMask)
    ax2.set_title('Clinical Mask')

    ax3.imshow(clinicalRoi)
    ax3.set_title('Selected ROI from Clinical Mask')

    ax4.imshow(highlightedRegions)
    ax4.set_title('Highlighted Regions')

    plt.show()


def printClassificationBrief(img, segmentation, roi, clinicalRoi, clinicalMask, prediction):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Classification Phase")
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(img, cmap="gray")
    ax1.set_title('Original Image')

    ax2.imshow(segmentation, cmap="gray")
    ax2.set_title('Segmentation')

    ax3.imshow(roi, cmap="gray")
    ax3.set_title('ROI')

    ax4.imshow(clinicalRoi, cmap="gray")
    ax4.set_title('Clinical ROI')

    ax5.imshow(clinicalMask, cmap="gray")
    ax5.set_title('Clinical Mask')

    ax6.imshow(prediction, cmap="gray")
    ax6.set_title('Prediction')

    plt.show()


def printClassificationCompareBrief(image, predictionMask_LR, predictionMask_LDA, predictionMask_KNN,
                                    clinicalMask, predictionMask_CART, predictionMask_NB, predictionMask_SVM):
    # Display result
    fig2, ax_arr = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Classification Phase")
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax_arr.ravel()

    ax1.imshow(image, cmap="gray")
    ax1.set_title('Original Image')

    ax2.imshow(predictionMask_LR, cmap="gray")
    ax2.set_title('LR')

    ax3.imshow(predictionMask_LDA, cmap="gray")
    ax3.set_title('LDA')

    ax4.imshow(predictionMask_KNN, cmap="gray")
    ax4.set_title('KNN')

    ax5.imshow(clinicalMask, cmap="gray")
    ax5.set_title('Clinical Mask')

    ax6.imshow(predictionMask_CART, cmap="gray")
    ax6.set_title('CART')

    ax7.imshow(predictionMask_NB, cmap="gray")
    ax7.set_title('NB')

    ax8.imshow(predictionMask_SVM, cmap="gray")
    ax8.set_title('SVM')

    plt.show()


def printHistogram(image):
    if len(image.shape) == 3:
        plt.subplot(121), plt.imshow(image)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv.calcHist([image], [i], None, [256], [0, 256])
            plt.subplot(122), plt.plot(histr, color=col)
    else:
        plt.subplot(121), plt.imshow(image, 'gray')
        plt.subplot(122), plt.hist(image.ravel(), 256, [0, 256])

    plt.xlim([0, 256])
    plt.show()


def plotUNetPrediction():
    masks_list = []
    image_list = []
    prediction_list = []

    for fileName in os.listdir("dataset/test/"):
        if fileName == "mask":
            continue
        image = cv.imread("dataset/test/" + fileName)
        image_list.append(image)

    for fileName in os.listdir("dataset/test/mask/"):
        mask = cv.imread("dataset/test/mask/" + fileName)
        masks_list.append(mask)

    for fileName, originalImage in zip(os.listdir("preds/"), image_list):
        prediction = cv.imread("preds/" + fileName)
        prediction = cv.normalize(prediction, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        prediction = prediction.astype(np.uint8)
        prediction = resize(prediction, (originalImage.shape[0], originalImage.shape[1], 3))
        prediction_list.append(prediction)

    for img, msk, prd in zip(image_list, masks_list, prediction_list):
        fig, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 10))
        fig.suptitle("uNet Segmentation")
        ax1, ax2, ax3 = ax_arr.ravel()

        ax1.imshow(img)
        ax1.set_title('Test Image')

        ax2.imshow(msk)
        ax2.set_title('Clinical Mask')

        ax3.imshow(prd)
        ax3.set_title('Prediction')

        plt.show()
