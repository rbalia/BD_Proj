import cv2 as cv
import numpy as np
import csv
from skimage import io
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage import segmentation as segmentationskimage
from skimage import morphology
from skimage import color

from modules.utils import utils_statistic as uStat
from modules.utils import utils_print as uPrint
from modules.utils import utils_slic as uSeg

def color2gray(img):
    if len(img.shape) == 3:    # if the image is Grayscale
        if img.shape[2] == 3:
            img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)  # then convert color

    return img

def gray2color(img):
    if len(img.shape) == 2:    # if the image is Grayscale
        img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)   # then convert color
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)   # then convert color
    return img

def preprocess(image):
    image = img_as_ubyte(image)  # Set range [0,255]
    if image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert to Gray Scale
    return image


def dilateLoc(loc, w, pxCount):
    maskRight = [(s + 2) % pxCount for s in loc]
    maskLeft = [(s - 2) % pxCount for s in loc]
    maskTop = [(s - w * 2) % pxCount for s in loc]
    maskBottom = [(s + w * 2) % pxCount for s in loc]
    augLoc = list(set(maskRight) | set(maskLeft) | set(maskTop) | set(maskBottom) | set(loc))
    return augLoc


def genSegmentationAndLabels(image, genMask=False, library="CV", printPlots=False, printLogs=False):

    ### GENERATE THE SUPERPIXEL SEGMENTATIONS

    if library == "CV":
        label_CV, boundaries_CV = uSeg.SegmentationSLIC_OpenCV(image, region_size=20, ratio=0.1)
        avg_CV = uSeg.SuperpixelAverageColor_Numpy(image, label_CV)
        segmentation = preprocess(avg_CV)
        label = np.copy(label_CV)
    elif library == "SK":
        label_Sk, boundaries_Sk = uSeg.SegmentationSLIC_Skimage(image, n_segments=250, compactness=25)
        avg_Sk = uSeg.SuperpixelAverageColor_Numpy(image, label_Sk)
        segmentation = preprocess(avg_Sk)
        label = np.copy(label_Sk)
    elif library == "SK2":
        label_Sk, boundaries_Sk = uSeg.SegmentationSLIC_Skimage(image, n_segments=400, compactness=15)
        avg_Sk = uSeg.SuperpixelAverageColor_Numpy(image, label_Sk)
        segmentation = preprocess(avg_Sk)
        label = np.copy(label_Sk)
    if library == "CV2":
        label_CV, boundaries_CV = uSeg.SegmentationSLIC_OpenCV(image, region_size=15, ratio=0.05)
        avg_CV = uSeg.SuperpixelAverageColor_Numpy(image, label_CV)
        segmentation = preprocess(avg_CV)
        label = np.copy(label_CV)

    ### REMOVE THE LABELS AT THE EDGE OF THE IMAGE #####################################################################
    image = preprocess(image)
    image1D = np.reshape(image, -1)  # image.reshape((image.shape[0]*image.shape[1],image.shape[2]))
    label1D = np.reshape(label, -1)

    labelIdList = np.unique(label1D)
    edgeLabels1D = np.zeros(image1D.shape)

    for i in labelIdList:
        loc = np.where(label1D == i)[0]
        h, w = image.shape[:2]

        # Detect all the labels touching the edges
        if all((i > w and i % w != 0 and i % w != w - 1 and i < (w * (h - 1))) for i in loc):
            edgeLabels1D[loc] = 255
        else:
            label1D[loc] = 0  # assign label reserved to background

    edgeLabelsMap = np.reshape(edgeLabels1D, [image.shape[0], image.shape[1]]).astype('uint8')
    _, edgeLabelsMap = cv.threshold(edgeLabelsMap, 10, 255, cv.THRESH_BINARY)  # convert to binary

    labelEdgeless = np.reshape(label1D, [label.shape[0], label.shape[1]]).astype('uint8')
    segmEdgeless = cv.bitwise_and(segmentation, segmentation, mask=edgeLabelsMap)

    # Print a Summary of the Processing Phase
    if printPlots:
        newBordersEnhance = segmentationskimage.mark_boundaries(image, labelEdgeless)
        uPrint.printProcessingBrief(segmentation, edgeLabelsMap, segmEdgeless, label, labelEdgeless, newBordersEnhance)

    return image, segmEdgeless, label, edgeLabelsMap


def genRoiMasks(image, segmented, labels, clinicalMask=None, printPlots=False):
    segmented = img_as_ubyte(segmented)

    image1D = image.reshape((image.shape[0] * image.shape[1]))
    clinicalMask1D = clinicalMask.reshape((clinicalMask.shape[0] * clinicalMask.shape[1]))
    segmentation1D = segmented.reshape((segmented.shape[0] * segmented.shape[1]))
    label1D = np.reshape(labels, -1)

    labelIdList = np.unique(label1D)
    roi1D = np.zeros(image1D.shape)
    clinicalRoi1D = np.zeros(image1D.shape)

    h, w = segmented.shape[:2]
    pxCount = w * h

    # CVS Data Generation
    for i in labelIdList:
        loc = np.where(label1D == i)[0]

        # Compute the Augmented Coordinates (Used to find adjacent labels)
        augLoc = dilateLoc(loc, w, pxCount)

        # Compute other statistical features
        meanValue = int(segmentation1D[loc[0]])  # faster mean: (all pixels are equal to the mean)(Int)

        adjacentLabels = np.unique(label1D[augLoc])
        maxVal = np.max(segmentation1D[augLoc])
        # Check if the region is surrounded by brighter regions only
        if all((j >= meanValue and j != 0) for j in segmentation1D[augLoc]):
            roi1D[loc] = 255
            """
            for adjLabel in adjacentLabels:
                if adjLabel == i:
                    continue
                adjLoc = np.where(label1D == adjLabel)[0]
                meanAdjValue = int(segmentation1D[adjLoc[0]])
                roi1D[adjLoc] = 20
                if meanAdjValue < (maxVal + meanValue)/1.5:
                    roi1D[adjLoc] = 80
                if meanAdjValue < meanValue*1.5:
                    roi1D[adjLoc] = 160
            """

        # If more than 1/3 of the pixel are labelled as true in the clinical mask set the region as clinicalROI
        if np.count_nonzero(clinicalMask1D[loc]) > len(loc) / 3:
            clinicalRoi1D[loc] = 255

        roi = np.reshape(roi1D, [segmented.shape[0], segmented.shape[1]]).astype('uint8')
        clinicalRoi = np.reshape(clinicalRoi1D, [segmented.shape[0], segmented.shape[1]]).astype('uint8')

    if printPlots:
        highlighted = segmentationskimage.mark_boundaries(segmented, roi, color=(1, 0, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, clinicalMask, color=(0, 1, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, clinicalRoi, color=(0, 0, 1))
        uPrint.printRoiBrief(roi, clinicalMask, clinicalRoi, highlighted)

    return roi, clinicalRoi


def genExtendedRoiFull(image, segmented, labels, roi, clinicalRoi=None, printPlots=False):
    segmented = img_as_ubyte(segmented)

    image1D = image.reshape((image.shape[0] * image.shape[1]))
    segmentation1D = segmented.reshape((segmented.shape[0] * segmented.shape[1]))
    roi1D = roi.reshape((roi.shape[0] * roi.shape[1]))
    label1D = np.reshape(labels, -1)

    labelIdList = np.unique(label1D)
    extendedRoi1D = np.copy(roi1D)

    h, w = segmented.shape[:2]
    pxCount = w * h

    roiLocs = np.where(roi1D == 255)[0]
    roiLbls = np.unique(label1D[roiLocs])
    roiAvgs = np.unique(segmentation1D[roiLocs])
    minAvg = np.min(roiAvgs)
    maxAvg = np.max(roiAvgs)
    # midAvg = (maxAvg+minAvg)/2

    print()

    # Filter or Extend each selected ROI
    for i in roiLbls:
        loc = np.where(label1D == i)[0]
        augLoc = dilateLoc(loc, w, pxCount)
        meanValue = int(segmentation1D[loc[0]])

        # Filter black areas (probably belongs to background)
        if meanValue < 5 or meanValue > 150:
            extendedRoi1D[loc] = 10
            continue

        adjacentLabels = np.unique(label1D[augLoc])
        maxVal = np.max(segmentation1D[augLoc])
        minVal = np.min(segmentation1D[augLoc])
        OldRange = (maxVal - minVal)

        # Check if the region is surrounded by brighter regions only
        if all((j >= meanValue and j > 10) for j in segmentation1D[augLoc]):
            extendedRoi1D[loc] = 255

            for adjLabel in adjacentLabels:
                if adjLabel == i:
                    continue
                adjLoc = np.where(label1D == adjLabel)[0]
                meanAdjValue = int(segmentation1D[adjLoc[0]])
                extendedRoi1D[adjLoc] = 20

                # Transform in range 100-0 from max-min
                rngMeanValue = (((meanValue - minVal) * 100) / OldRange) + 0
                rngMeanAdjValue = (((meanAdjValue - minVal) * 100) / OldRange) + 0
                # if meanAdjValue < (maxVal + meanValue)/1.5:
                if rngMeanAdjValue - rngMeanValue < 60:
                    extendedRoi1D[adjLoc] = 80
                # if meanAdjValue < meanValue*1.5:
                if rngMeanAdjValue - rngMeanValue < 30:
                    extendedRoi1D[adjLoc] = 160

                    loc2 = np.where(label1D == adjLabel)[0]
                    augLoc2 = dilateLoc(loc2, w, pxCount)
                    adjacentLabels2 = np.unique(label1D[augLoc2])

                    for adjLabel2 in adjacentLabels2:
                        if adjLabel2 in roiLbls or adjLabel2 in adjacentLabels:
                            continue
                        adjLoc2 = np.where(label1D == adjLabel2)[0]
                        meanAdjValue2 = int(segmentation1D[adjLoc2[0]])

                        # Transform in range 100-0 from max-min
                        rngMeanAdjValue2 = (((meanAdjValue2 - minVal) * 100) / OldRange) + 0
                        if rngMeanAdjValue2 - rngMeanValue < 60:
                            extendedRoi1D[adjLoc2] = 70
                        if rngMeanAdjValue2 - rngMeanValue < 30:
                            extendedRoi1D[adjLoc2] = 140

    extendedRoi = np.reshape(extendedRoi1D, [segmented.shape[0], segmented.shape[1]]).astype('uint8')

    if printPlots:
        highlighted = segmentationskimage.mark_boundaries(segmented, roi, color=(1, 0, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, extendedRoi, color=(0, 1, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, clinicalRoi, color=(0, 0, 1))
        highlighted = img_as_ubyte(highlighted)
        uPrint.printBrief4Cells(
            "Extended Roi - Processing Phase",
            ["Roi", "Extended Roi", "Clinical Mask from Labels", "Highlight on Segmentation"],
            [roi, extendedRoi, clinicalRoi, highlighted])

    return extendedRoi


def genExtendedRoi(image, segmented, labels, roi, clinicalRoi=None, printPlots=False):
    segmented = img_as_ubyte(segmented)

    image1D = image.reshape((image.shape[0] * image.shape[1]))
    segmentation1D = segmented.reshape((segmented.shape[0] * segmented.shape[1]))
    roi1D = roi.reshape((roi.shape[0] * roi.shape[1]))
    label1D = np.reshape(labels, -1)

    labelIdList = np.unique(label1D)
    extendedRoi1D = np.copy(roi1D)

    h, w = segmented.shape[:2]
    pxCount = w * h

    roiLocs = np.where(roi1D == 255)[0]
    roiLbls = np.unique(label1D[roiLocs])
    roiAvgs = np.unique(segmentation1D[roiLocs])
    # minAvg = np.min(roiAvgs)
    # maxAvg = np.max(roiAvgs)
    # midAvg = (maxAvg+minAvg)/2

    print()

    # Filter or Extend each selected ROI
    for i in roiLbls:
        loc = np.where(label1D == i)[0]
        augLoc = dilateLoc(loc, w, pxCount)
        meanValue = int(segmentation1D[loc[0]])

        # Filter black areas (probably belongs to background)
        if meanValue < 5 or meanValue > 150:
            extendedRoi1D[loc] = 0
            continue

        # if len(currRoiLbls) > 5 and meanValue > midAvg:
        #    extendedRoi1D[loc] = 0
        #    continue

        adjacentLabels = np.unique(label1D[augLoc])
        maxVal = np.max(segmentation1D[augLoc])
        minVal = np.min(segmentation1D[augLoc])
        OldRange = (maxVal - minVal)

        # Check if the region is surrounded by brighter regions only
        if all((j >= meanValue and j != 0) for j in segmentation1D[augLoc]):
            extendedRoi1D[loc] = 255

            for adjLabel in adjacentLabels:
                if adjLabel == i:
                    continue
                adjLoc = np.where(label1D == adjLabel)[0]
                meanAdjValue = int(segmentation1D[adjLoc[0]])
                # extendedRoi1D[adjLoc] = 20

                # Transform in range 100-0 from max-min
                rngMeanValue = (((meanValue - minVal) * 100) / OldRange) + 0
                rngMeanAdjValue = (((meanAdjValue - minVal) * 100) / OldRange) + 0
                # if meanAdjValue < (maxVal + meanValue)/1.5:
                # if rngMeanAdjValue - rngMeanValue < 60:
                # extendedRoi1D[adjLoc] = 80
                # if meanAdjValue < meanValue*1.5:
                if rngMeanAdjValue - rngMeanValue < 30:
                    extendedRoi1D[adjLoc] = 160

                    loc2 = np.where(label1D == adjLabel)[0]
                    augLoc2 = dilateLoc(loc2, w, pxCount)
                    adjacentLabels2 = np.unique(label1D[augLoc2])

                    for adjLabel2 in adjacentLabels2:
                        if adjLabel2 in roiLbls or adjLabel2 in adjacentLabels:
                            continue
                        adjLoc2 = np.where(label1D == adjLabel2)[0]
                        meanAdjValue2 = int(segmentation1D[adjLoc2[0]])

                        # Transform in range 100-0 from max-min
                        rngMeanAdjValue2 = (((meanAdjValue2 - minVal) * 100) / OldRange) + 0
                        # if rngMeanAdjValue2 - rngMeanValue < 60:
                        # extendedRoi1D[adjLoc2] = 70
                        if rngMeanAdjValue2 - rngMeanValue < 30:
                            extendedRoi1D[adjLoc2] = 140

    extendedRoi = np.reshape(extendedRoi1D, [segmented.shape[0], segmented.shape[1]]).astype('uint8')

    if printPlots:
        highlighted = segmentationskimage.mark_boundaries(segmented, roi, color=(1, 0, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, extendedRoi, color=(0, 1, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, clinicalRoi, color=(0, 0, 1))
        highlighted = img_as_ubyte(highlighted)
        uPrint.printBrief4Cells(
            "Extended Roi - Processing Phase",
            ["Roi", "Extended Roi", "Clinical Mask from Labels", "Highlight on Segmentation"],
            [roi, extendedRoi, clinicalRoi, highlighted])

    return extendedRoi


def genExtendedRoi2(segmented, labels, roi, clinicalRoi=None, printPlots=False):
    segmented = img_as_ubyte(segmented)

    segmentation1D = segmented.reshape((segmented.shape[0] * segmented.shape[1]))
    roi1D = roi.reshape((roi.shape[0] * roi.shape[1]))
    label1D = np.reshape(labels, -1)

    labelIdList = np.unique(label1D)
    extendedRoi1D = np.copy(roi1D)

    h, w = segmented.shape[:2]
    pxCount = w * h

    roiLocs = np.where(roi1D == 255)[0]
    roiLbls = np.unique(label1D[roiLocs])
    roiAvgs = np.unique(segmentation1D[roiLocs])

    # Filter or Extend each selected ROI
    for i in roiLbls:
        loc = np.where(label1D == i)[0]
        augLoc = dilateLoc(loc, w, pxCount)
        meanValue = int(segmentation1D[loc[0]])

        # Filter black areas (probably belongs to background)
        if meanValue < 5 or meanValue > 100:
            extendedRoi1D[loc] = 0
            continue

        adjacentLabels = np.unique(label1D[augLoc])
        maxVal = np.max(segmentation1D[augLoc])
        minVal = np.min(segmentation1D[augLoc])
        OldRange = (maxVal - minVal)

        # Check if the region is surrounded by brighter regions only
        if all((j >= meanValue and j != 0) for j in segmentation1D[augLoc]):
            extendedRoi1D[loc] = 255

            for adjLabel in adjacentLabels:
                if adjLabel == i:
                    continue
                adjLoc = np.where(label1D == adjLabel)[0]
                meanAdjValue = int(segmentation1D[adjLoc[0]])
                # extendedRoi1D[adjLoc] = 20

                # Transform in range 100-0 from max-min
                rngMeanValue = (((meanValue - minVal) * 100) / OldRange) + 0
                rngMeanAdjValue = (((meanAdjValue - minVal) * 100) / OldRange) + 0

                if rngMeanAdjValue - rngMeanValue < 20 and meanAdjValue < 80:
                    extendedRoi1D[adjLoc] = 160

                    loc2 = np.where(label1D == adjLabel)[0]
                    augLoc2 = dilateLoc(loc2, w, pxCount)
                    adjacentLabels2 = np.unique(label1D[augLoc2])

                    for adjLabel2 in adjacentLabels2:
                        if adjLabel2 in roiLbls or adjLabel2 in adjacentLabels:
                            continue
                        adjLoc2 = np.where(label1D == adjLabel2)[0]
                        meanAdjValue2 = int(segmentation1D[adjLoc2[0]])

                        # Transform in range 100-0 from max-min
                        rngMeanAdjValue2 = (((meanAdjValue2 - minVal) * 100) / OldRange) + 0

                        if rngMeanAdjValue2 - rngMeanValue < 25 and meanAdjValue2 < 80:
                            extendedRoi1D[adjLoc2] = 120


        # Rileva ed Evidenzia regioni di minimo non evidenziate dall'intersezione con la prediction

        if any((j < meanValue and j != 0 and j != meanValue) for j in segmentation1D[augLoc]):
            minMean = int(np.min(segmentation1D[augLoc]))

            for adjLabel in adjacentLabels:
                if adjLabel == i:
                    continue

                adjLoc = np.where(label1D == adjLabel)[0]
                if minMean == (segmentation1D[adjLoc[0]]) and extendedRoi1D[adjLoc[0]] < 1:
                    extendedRoi1D[adjLoc] = 101


    extendedRoi = np.reshape(extendedRoi1D, [segmented.shape[0], segmented.shape[1]]).astype('uint8')

    if printPlots:
        highlighted = segmentationskimage.mark_boundaries(segmented, roi, color=(1, 0, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, extendedRoi, color=(0, 1, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, clinicalRoi, color=(0, 0, 1))
        highlighted = img_as_ubyte(highlighted)
        uPrint.printBrief4Cells(
            "Extended Roi - Processing Phase",
            ["Roi", "Extended Roi", "Clinical Mask from Labels", "Highlight on Segmentation"],
            [roi, extendedRoi, clinicalRoi, highlighted])

    return extendedRoi

def recursive_region_validation():
    print("nulla")

def genExtendedRoi3(segmented, labels, roi, clinicalRoi=None, printPlots=False):
    segmented = img_as_ubyte(segmented)

    segmentation1D = segmented.reshape((segmented.shape[0] * segmented.shape[1]))
    roi1D = roi.reshape((roi.shape[0] * roi.shape[1]))
    label1D = np.reshape(labels, -1)

    extendedRoi1D = np.copy(roi1D)

    h, w = segmented.shape[:2]
    pxCount = w * h

    roiLocs = np.where(roi1D == 255)[0]
    roiLbls = np.unique(label1D[roiLocs]).tolist()

    blacklist = []
    visitedLabels = []
    toVisitLabels = roiLbls

    # Filter or Extend each selected ROI
    while toVisitLabels:
        currLbl = toVisitLabels.pop(0)
        visitedLabels = visitedLabels + [currLbl]
        #print(toVisitLabels)
        #print(visitedLabels)

        loc = np.where(label1D == currLbl)[0]
        augLoc = dilateLoc(loc, w, pxCount)
        meanValue = int(segmentation1D[loc[0]])


        # Filter black, or too bright areas (probably belongs to background)
        if meanValue < 5 or meanValue > 100 or np.min(segmentation1D[augLoc]) == 0 or currLbl in blacklist:
            if currLbl not in blacklist:
                blacklist = blacklist + [currLbl]
            extendedRoi1D[loc] = 0
            continue


        roisLocs = np.where(extendedRoi1D > 0)[0]
        roisMean = np.mean(segmentation1D[roisLocs])

        maxVal = np.max(segmentation1D[roisLocs])
        minVal = np.min(segmentation1D[roisLocs])
        ulMarg = (255 - maxVal) / 10
        llMarg = minVal / 10

        #print(str(minVal) + " - " + str(meanValue))
        #if (minVal - llMarg) < meanValue < (maxVal + ulMarg):
        #print(roisMean - meanValue)
        isValid = False
        if all((j >= meanValue and j != 0) for j in segmentation1D[augLoc]):
            if extendedRoi1D[loc[0]] == 0:
                extendedRoi1D[loc] = 200
            isValid = True

        #elif meanValue < minVal or abs(roisMean - meanValue) < 20:
        elif meanValue - minVal < 30:
            if extendedRoi1D[loc[0]] == 0:
                extendedRoi1D[loc] = 160
            isValid = True

        if isValid:
            adjacentLabels = np.unique(label1D[augLoc])
            for i in adjacentLabels:
                if (i not in toVisitLabels and i not in visitedLabels) and i != 0:

                    toVisitLabels = toVisitLabels + [i]





        """maxVal = np.max(segmentation1D[augLoc])
        minVal = np.min(segmentation1D[augLoc])
        OldRange = (maxVal - minVal)

        roisLoc = np.where(extendedRoi1D > 0)[0]
        meanRois = int(np.mean((segmentation1D[roisLoc[0]])))
        # extendedRoi1D[adjLoc] = 20

        # Transform in range 100-0 from max-min
        rngMeanValue = (((meanValue - minVal) * 100) / OldRange) + 0
        rngMeanAdjValue = (((meanRois - minVal) * 100) / OldRange) + 0

        if rngMeanAdjValue - rngMeanValue < 20 and meanRois < 80:
            extendedRoi1D[loc] = 160

            adjacentLabels = np.unique(label1D[augLoc])
            for i in adjacentLabels:
                if i not in toVisitLabels or i not in visitedLabels:
                    toVisitLabels = toVisitLabels + [i]
        """


        # Rileva ed Evidenzia regioni di minimo non evidenziate dall'intersezione con la prediction
        """if any((j < meanValue and j != 0 and j != meanValue) for j in segmentation1D[augLoc]):
            minMean = int(np.min(segmentation1D[augLoc]))

            for adjLabel in adjacentLabels:
                if adjLabel == i:
                    continue

                adjLoc = np.where(label1D == adjLabel)[0]
                if minMean == (segmentation1D[adjLoc[0]]) and extendedRoi1D[adjLoc[0]] < 1:
                    extendedRoi1D[adjLoc] = 101"""


    extendedRoi = np.reshape(extendedRoi1D, [segmented.shape[0], segmented.shape[1]]).astype('uint8')

    if printPlots:
        highlighted = segmentationskimage.mark_boundaries(segmented, roi, color=(1, 0, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, extendedRoi, color=(0, 1, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, clinicalRoi, color=(0, 0, 1))
        highlighted = img_as_ubyte(highlighted)
        uPrint.printBrief4Cells(
            "Extended Roi - Processing Phase",
            ["Roi", "Extended Roi", "Clinical Mask from Labels", "Highlight on Segmentation"],
            [roi, extendedRoi, clinicalRoi, highlighted])

    return extendedRoi



def genPredictionRoiIntesection(image, segmentation, labels, prediction, roi, printPlots=False):
    h, w = image.shape[:2]
    image1D = image.reshape(h * w)
    prediction1D = prediction.reshape(h * w)
    roi1D = roi.reshape(h * w)
    label1D = np.reshape(labels, -1)

    labelIdList = np.unique(label1D)

    intersectionCanvas1D = np.zeros(image1D.shape)

    # CVS Data Generation
    for i in labelIdList:
        loc = np.where(label1D == i)[0]
        regionPxCount = len(loc)
        if np.count_nonzero(prediction1D[loc]) > regionPxCount / 4 and np.count_nonzero(roi1D[loc]) > 0 and roi1D[
            loc[0]] > 30:
            intersectionCanvas1D[loc] = roi1D[loc[0]]

    prediction = prediction.astype("uint8")
    intersection = np.reshape(intersectionCanvas1D, [h, w]).astype('uint8')

    if printPlots:
        highlighted = segmentationskimage.mark_boundaries(image, roi, color=(1, 0, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, prediction, color=(0, 1, 0))
        highlighted = segmentationskimage.mark_boundaries(highlighted, intersection, color=(0, 0, 1))
        uPrint.printBrief4Cells("uNet / Handcrafted Results",
                                ["Handrafted detected ROIs", "uNet Predictions (Thresholded)",
                                 "Roi + uNet Intersection", "Original Image (Annotated)"],
                                [roi, prediction, intersection, highlighted])

    return intersection


def roiFromPrediction(labl, pred):
    h, w = labl.shape[:2]
    labl1D = labl.reshape(h * w)
    pred1D = pred.reshape(h * w)

    labelList = np.unique(labl1D)

    interesectionCanvas1D = np.zeros(labl1D.shape)

    for i in labelList:
        loc = np.where(labl1D == i)[0]
        regionPxCount = len(loc)

        if np.count_nonzero(pred1D[loc]) > regionPxCount / 4:
            interesectionCanvas1D[loc] = 255

    intersection = np.reshape(interesectionCanvas1D, [h, w]).astype('uint8')
    return intersection


def imReconstruction(img, seed):

    if np.count_nonzero(seed) > 0:
        reconstruct = morphology.reconstruction(seed, img, method="dilation")
        reconstruct = np.array(reconstruct, dtype=np.uint8)
    else:
        reconstruct = img
    return reconstruct


def getClinicalMask(fileName, setDir, mskDir):
    clinicalMask = cv.imread(setDir + mskDir + fileName + ".png")
    clinicalMask = cv.cvtColor(clinicalMask, cv.COLOR_BGR2GRAY)
    return clinicalMask


def evaluateIOU(prediction, label):
    prediction = prediction.reshape((prediction.shape[0], prediction.shape[1],1))
    label = label.reshape((label.shape[0], label.shape[1], 1))
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    iou_score = np.sum(intersection) / np.sum(union)
    return round(iou_score, 2)


def evaluateDice(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.sum(prediction) + np.sum(label)
    dice_score = (2 * np.sum(intersection) + 1.) / (union + 1.)
    return round((dice_score), 2)
