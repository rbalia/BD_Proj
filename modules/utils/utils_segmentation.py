import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, img_as_float
from modules.utils import utils_print as uPrint


def handCraftedSegmentation(img):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    imgMod = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)
    imgMod = cv.morphologyEx(imgMod, cv.MORPH_CLOSE, kernel, iterations=30)
    imgMod = cv.morphologyEx(imgMod, cv.MORPH_OPEN, kernel2, iterations=15)

    imgBlur = cv.blur(imgMod, (20, 20))
    imgBlur = cv.blur(imgBlur, (25, 25))

    imgAdd = cv.add(img, cv.bitwise_not(imgBlur))
    #uPrint.printBrief4Cells("title", ["1","2","3","4"], [img, imgBlur, imgMod, imgAdd])
    #imgAdd[cv.blur(img, (10, 10)) == 0] = 255
    imgAdd = cv.blur(imgAdd, (10, 10))

    imgThres = np.copy(imgAdd)
    imgThres[imgAdd > 230] = 255

    min = np.min(imgThres)
    imgThres[imgThres < (min + 255) / 1.8] = 0
    imgThres[imgThres > 0] = 255

    return imgThres, imgAdd

def handCraftedSegmentationBackup(img):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    imgMod = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)
    imgMod = cv.morphologyEx(imgMod, cv.MORPH_CLOSE, kernel, iterations=30)
    imgMod = cv.morphologyEx(imgMod, cv.MORPH_OPEN, kernel2, iterations=15)

    imgBlur = cv.blur(imgMod, (20, 20))
    imgBlur = cv.blur(imgBlur, (25, 25))

    imgAdd = cv.add(img, cv.bitwise_not(imgBlur))
    #imgAdd[cv.blur(img, (10, 10)) == 0] = 255
    imgAdd = cv.blur(imgAdd, (10, 10))

    imgThres = np.copy(imgAdd)
    imgThres[imgAdd > 230] = 255

    min = np.min(imgThres)
    imgThres[imgThres < (min + 255) / 1.8] = 0
    imgThres[imgThres > 0] = 255

    return imgThres, imgAdd

def segmentation3(img, mask):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    imgMod = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)
    imgMod = cv.morphologyEx(imgMod, cv.MORPH_CLOSE, kernel, iterations=20)
    # imgMod = cv.morphologyEx(imgMod, cv.MORPH_OPEN, kernel2, iterations=15)
    imgMod = cv.morphologyEx(imgMod, cv.MORPH_OPEN, kernel, iterations=15)

    imgBlur = cv.blur(imgMod, (20, 20))
    imgBlur = cv.blur(imgBlur, (25, 25))

    imgAdd = cv.add(img, cv.bitwise_not(imgBlur))
    # imgAdd[img > 210] = 255
    imgAdd[cv.blur(img, (10, 10)) == 0] = 255

    # imgAdd = cv.erode(imgAdd, (10,10))
    imgAdd = cv.blur(imgAdd, (10, 10))

    # _, img2 = cv.threshold(imgAdd, np.median(imgAdd), 255, cv.THRESH_BINARY)

    imgAdd = img_as_float(imgAdd)
    imgAdd = imgAdd - np.min(imgAdd)
    imgAdd = imgAdd / np.max(imgAdd)
    imgAdd = img_as_ubyte(imgAdd)
    imgAdd = cv.erode(imgAdd, (5,5))

    imgBlur2 = cv.blur(imgAdd, (5,5))
    imgAdd = cv.add(img, cv.bitwise_not(imgBlur2))
    imgAdd = cv.bitwise_not(imgAdd)

    img2 = np.copy(imgAdd)
    img2[imgAdd > 230] = 255

    # img2 = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel, iterations=1)
    # img2 = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel, iterations=1)

    # img2[cv.blur(img,(20,20)) == 0] = 255
    # _, img2 = cv.threshold(imgAdd, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    min = np.min(img2)
    mean = np.mean(img2)
    img2[img2 < (min + 255) / 1.8] = 0
    img2[img2>0] = 255

    fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig.suptitle("Title")
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(img, cmap="gray"), ax1.set_title("Original")
    ax2.imshow(mask, cmap="gray"), ax2.set_title("Mask")
    ax3.imshow(imgAdd, cmap="gray"), ax3.set_title("Processing")
    ax4.imshow(img2, cmap="gray"), ax4.set_title("Segmentation")

    plt.show()

    return img2, imgAdd