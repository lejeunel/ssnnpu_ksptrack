import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io
from skimage import color
import sys
import os
import gazeCsv as gaze
import my_utils as utils
from skimage.feature import BRIEF
import progressbar


def maskKeypoints(kp,mask):

    kpOut = []
    kpOutIdx = []
    for n in range(len(kp)):
        j = int(kp[n].pt[0])
        i = int(kp[n].pt[1])
        if mask[i,j]:
            kpOut.append(kp[n])
            kpOutIdx.append(n)

    return kpOut,kpOutIdx

def getMatchedKeypoints(matches,kp,excludeIdx):

    kpOut = []
    for i in range(len(matches)):
        if i != excludeIdx:
            for j in range(len(matches[i])):
                #print kp[matches[i][j].queryIdx]
                kpOut.append(kp[matches[i][j].trainIdx])

    return kpOut

def getValuesInCircle(A,center,radius):

    valOut = list()

    y,x = np.ogrid[0:A.shape[0], 0:A.shape[1]]
    mask = (x-center[0])**2 + (y-center[1])**2 <= radius*radius

    return A[mask], mask

def getSeenSPsIn(labels,myGaze):
    """
    Finds coordinate of centroid and label index of region with gaze-point in
    """

    print("Getting label index of seen parts...")
    nFrames = np.min((labels.shape[2],myGaze.shape[0]))
    seenLabel = np.zeros(nFrames)
    mask = np.zeros((labels.shape[0],labels.shape[1]))
    thisCentroid = []
    for i in range(nFrames): #iterate over frames
        #for j in range(nLabels+1): #iterate over labels
        labelNums = np.unique(labels[:,:,i])
        ci,cj = gaze.gazeCoord2Pixel(myGaze[i,3],myGaze[i,4],mask.shape[1],mask.shape[0])
        for j in range(labelNums.shape[0]): #iterate over labels
            #print i,j,nFrames
            mask = (labels[:,:,i] == labelNums[j])
            #thisCentroid = ndimage.measurements.center_of_mass(mask)
            if (mask[ci,cj]):
                break
        seenLabel[i] = labelNums[j]

    return(seenLabel)

def getSeenSPs(labels,myGaze,method=None):
    """
    Finds coordinate of centroid and label index of region whose center of mass is closest to gaze-point:
    """

    print("Getting label index of seen parts...")
    nLabels = np.max(labels[0])
    nFrames = len(labels)
    seenCentroids = np.zeros((nLabels,nFrames))
    seenLabel = np.zeros(nFrames)
    mask = np.zeros(labels[0].shape)
    thisCentroid = []
    for i in range(nFrames): #iterate over frames
        distsToCentroid = []
        #for j in range(nLabels+1): #iterate over labels
        for j in range(nLabels): #iterate over labels
            #print i,j,nLabels,nFrames
            mask = (labels[i] == j)
            thisCentroid = ndimage.measurements.center_of_mass(mask)
            ci,cj = gaze.gazeCoord2Pixel(myGaze[i,3],myGaze[i,4],mask.shape[1],mask.shape[0])
            distsToCentroid.append(np.linalg.norm(np.array([ci,cj])-thisCentroid))
        seenCentroids[np.argmin(distsToCentroid),i] = 1
        #seenLabel[i] = labels[i][np.argmin(distsToCentroid)]
        seenLabel[i] = np.argmin(distsToCentroid)
    return (seenCentroids,seenLabel)

def getSeenSPsInRadius(labels,myGaze,gazeRadius):

    nLabels = np.max(labels[0])
    nFrames = myGaze.shape[0]
    seenCentroids = np.zeros((nLabels,nFrames))
    mask = np.zeros(labels[0].shape)
    thisCentroid = []
    for i in range(nFrames): #iterate over labels
        for j in range(nLabels+1): #iterate over frames
            mask = (labels[i] == j)
            thisCentroid = ndimage.measurements.center_of_mass(mask)
            ci,cj = gaze.gazeCoord2Pixel(myGaze[i,3],myGaze[i,4],mask.shape[1],mask.shape[0])
            if np.linalg.norm(np.array([ci,cj])-thisCentroid) <= gazeRadius:
                seenCentroids[j,i] = 1
    return seenCentroids

def getSeenSPsInRect(labels,myGaze,w_norm,h_norm,center,frameIndSource,frameIndTarget):

    seenParts = []
    nLabels = np.max(labels[0])
    w = w_norm*labels.shape[1]
    h = h_norm*labels.shape[0]
    mask = np.zeros(labels[0].shape)
    ci,cj = gaze.gazeCoord2Pixel(myGaze[frameIndSource,3],myGaze[frameIndSource,4],mask.shape[1],mask.shape[0])
    mask[int(ci-h/2):int(ci+h/2),int(cj-w/2):int(cj+w/2)] = 1

    for j in range(nLabels+1): #iterate over labels
        thisLabelMask = (labels[frameIndTarget] == j)
        thisCentroid = ndimage.measurements.center_of_mass(thisLabelMask)
        if mask[thisCentroid[0],thisCentroid[1]]:
            seenParts.append(j)

    return seenParts

def getColorDesc(frameFileNames,labels,dense_step=1,normCst = 1):
    """
    normCst : pixel values are normalized w.r.t that. Default = 1
    """

    keyFramesIdx = np.arange(0,len(frameFileNames))
    desc = [] # average histogram (descriptor) of keypoint

    for keyFrame in keyFramesIdx:
        desc.append([])
        img = utils.imread(frameFileNames[keyFrame])
        img = color.rgb2hsv(img)
        for i in np.unique(labels[:,:,keyFrame]):
            mask = (labels[:,:,keyFrame] == i)
            this_region = np.zeros((img.shape[2],np.sum(mask)))
            for j in range(img.shape[2]):
                this_channel = img[:,:,j]
                this_region[j,:] = np.ravel(this_channel[mask])

            this_mean = np.mean(this_region,axis=1)/np.tile(normCst,3)

            desc[keyFrame].append((i,this_mean))

    return(desc)

def normalizeDesc(desc):

    for i in range(desc.shape[1]):
        this_min = np.min(desc[:,i])
        this_max = np.max(desc[:,i])
        desc[:,i] = (desc[:,i] - this_min)/(this_max-this_min)

    return desc

def getMeanSPs(sp,feature2d,frameFileNames,denseStep,sp_num_iterations):
    """
    Computes mean descriptors inside superpixel segmentation for a set of images

    Parameters
    ----------
    sp : Superpixel object
    feature2d : OpenCV object to compute keypoints and descriptors
    frameFileNames : list of strings
        Complete path to set of images
    denseStep : Step size of grid in which keypoints are generated
    sp_num_iterations : Number of iterations for superpixel method

    Returns
    -------
    avgDesc : list of list
        Average descriptor of each superpixel
    kps : list of list
        Keypoints (all of them)
    desc : list of list
        Descriptors (all of them)
    labels : list of arrays
        Superpixel labels of each image
    spLabelContourMask : list of arrays
        Superpixel contour mask of each image
    """

    keyFramesIdx = np.arange(0,len(frameFileNames))

    labels = []
    spLabelContourMask = []
    kpKeyFrame = []
    avgDesc = []
    kps = []
    desc = []

    for keyFrame in keyFramesIdx:
        sys.stdout.write("Processing frame %i/%i \n" % (keyFrame,len(keyFramesIdx)-1))

        img = utils.imread(frameFileNames[keyFrame])
        sp.iterate(cv2.cvtColor(img,cv2.COLOR_RGB2HSV),sp_num_iterations)
        spLabelToCheck = sp.getLabels()

        while(True):
            detectedEmpty = False

            kpDense = [cv2.KeyPoint(x, y, denseStep) for y in range(0, img.shape[0], denseStep)  for x in range(0, img.shape[1], denseStep)]

            mask = np.zeros(img.shape)
            for i in range(sp.getNumberOfSuperpixels()):
                mask = (spLabelToCheck == i)
                thisSPkps, thisSPkpsIdx = maskKeypoints(kpDense,mask)

                if len(thisSPkps) == 0:
                    detectedEmpty = True
                    denseStep -= 1
                    #print denseStep
                    break
                    #print "A region without keypoints was detected, reducing keypoint step size by 1 on this image"

            if detectedEmpty == False:
                break

        labels.append(spLabelToCheck)
        spLabelContourMask.append(sp.getLabelContourMask())
        kptmp,destmp = feature2d.compute(img, kpDense)
        kps.append(kptmp)
        desc.append(destmp)

        #Average descriptors contained in superpixels
        thisFrameAvgDesc = []
        for i in range(sp.getNumberOfSuperpixels()):
            mask = (labels[-1] == i)
            thisSPkps, thisSPkpsIdx = maskKeypoints(kps[-1],mask)
            thisSPDesc = desc[-1][thisSPkpsIdx]
            #print thisSPDesc
            #print "thisSPDesc.shape: ", thisSPDesc.shape
            #thisFrameAvgDesc.append(np.mean(thisSPDesc,axis=0))
            thisFrameAvgDesc.append(np.mean(thisSPDesc,axis=0))

        avgDesc.append(thisFrameAvgDesc)

    avgDesc = np.asarray(avgDesc)

    #return(avgDesc,kps,desc,labels,spLabelContourMask)
    return(np.asarray(avgDesc),kps,desc,labels,spLabelContourMask)

def getNearestSP(avgDesc,refLabelsInd,refSPInd,matchLabelsInd):
    """
    Find superpixel of frame matchLabelsInd whose average descriptor is nearest to reference frame refLabelsInd/refSPInd
    """

    dist = np.zeros(len(avgDesc[refLabelsInd]))

    for j in range(len(avgDesc[refLabelsInd])):

        thisDist = np.linalg.norm(avgDesc[refLabelsInd][refSPInd] - avgDesc[matchLabelsInd][j])
    #   print "j=",j,',thisDist=',thisDist
        dist[j] = thisDist

    return(np.nanmin(dist),np.nanargmin(dist))

def getSPinMask(avgDesc,refLabelsInd,refSPInd,matchLabelsInd,labelMask,centroids):
    """
    Returns Euclidean distance (w.r.t. refLabelsInd/refSPInd) and index of superpixels of frame matchLabelsInd whose centroids are contained in labelMask
    """

    idx = []
    dist = []

    for j in range(len(avgDesc[refLabelsInd])):
        thisDist = np.linalg.norm(avgDesc[refLabelsInd][refSPInd] - avgDesc[matchLabelsInd][j])
    #   print "j=",j,',thisDist=',thisDist
        dist[j] = thisDist

    return(np.nanmin(dist),np.nanargmin(dist))

def getLabelCentroids(labels):
    """
    labels is a list of positive-integer matrix whose values refer to a label. Values must be contiguous.
    Returns: List of same length as labels, containing arrays with first element containing value of label, and second and third element containing the x and y coordinates of centroid.
    """

    nFrames = labels.shape[2]
    centroids = []

    #normFactor = np.linalg.norm((labels.shape[0],labels.shape[1]))

    centroid_list = []
    with progressbar.ProgressBar(maxval=nFrames) as bar:
        for i in range(nFrames):
            bar.update(i)
            idxLabels = np.unique(labels[:,:,i])
            #print "frame: ", str(i+1), "/", str(nFrames)
            for j in range(len(idxLabels)):
                thisMask = (labels[:,:,i] == idxLabels[j])
                pos = np.asarray(ndimage.measurements.center_of_mass(thisMask))
                pos_norm = gaze.gazePix2Norm(pos[1],pos[0],labels.shape[1],labels.shape[0])
                centroid_list.append([i, int(idxLabels[j]),pos_norm[0],pos_norm[1]])
    centroids = pd.DataFrame(centroid_list,columns=['frame', 'sp_label', 'pos_norm_x','pos_norm_y'])


    return(centroids)

def drawLabelContourMask(img,label):

    red_img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    mask_inv = np.invert(label)
    i_mask,j_mask = np.where(label)
    i_mask_inv,j_mask_inv = np.where(mask_inv)
    result = img
    result[i_mask,j_mask,:] = 0
    result[i_mask,j_mask,0] = 255

    return result

def getCircleIdx(gazeCoord,shape,radius):

    ci,cj = gaze.gazeCoord2Pixel(gazeCoord[1],gazeCoord[0],shape[1],shape[0])
    center = (ci,cj)

    y,x = np.ogrid[0:shape[0], 0:shape[1]]
    mask = (x-center[1])**2 + (y-center[0])**2 <= radius*radius

    return np.where(mask)

def getMeanColorInCircle(img,gazeCoord,gaze_radius,normCst=1):

    ci,cj = gaze.gazeCoord2Pixel(gazeCoord[0],gazeCoord[1],img.shape[1],img.shape[0])
    this_pix,_ = getValuesInCircle(img,(cj,ci),gaze_radius)
    this_mean = np.mean(this_pix,axis=0)/np.tile(normCst,3)

    return this_mean,this_pix

def getSeenColorsInCircle(img,gazeCoord,gaze_radius,standardization=None):

    ci,cj = gaze.gazeCoord2Pixel(gazeCoord[i,3],gazeCoord[i,4],imgShape[1],imgShape[0])
    this_= getValuesInCircle(img,(cj,ci),gaze_radius)

def extractDaisyDescriptors(img,patch_size):

    extractor = BRIEF(patch_size=patch_size)
    i_idx = np.arange(0,img.shape[0])
    j_idx = np.arange(0,img.shape[1])
    kps_i,kps_j = np.meshgrid(i_idx,j_idx)
    kps_i = kps_i.ravel()
    kps_i.shape = (kps_i.shape[0],1)
    kps_j = kps_j.ravel()
    kps_j.shape = (kps_j.shape[0],1)
    for i in range(img.shape[2]):
        extractor.extract(img[:,:,i],np.concatenate((kps_i,kps_j),axis=1))
        dsc = extractor.descriptors

    return dsc,kps_i,kps_j
