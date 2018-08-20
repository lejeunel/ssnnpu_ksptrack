#!/usr/bin/python
#import graph_tool.all as gt
import cv2
import numpy as np
from superPixels import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def makeFrameFileNames(prefix,nDigits,idx,path='',extension='.png'):

    frameFileNames = []
    formatStr =  path + prefix + '%0' + str(nDigits) + 'd'

    for i in range(idx.shape[0]):
        frameFileNames.append(str(formatStr%idx[i])+extension)

    return frameFileNames

def drawKPandSP(img,kp,spLabels,spLabelsContourMask):

    mask = spLabelsContourMask

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)

    # Draw matches.
    img = cv2.drawKeypoints(img, kp, None)
    result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    red_img = np.zeros((img.shape[0], img.shape[1],3), np.uint8)
    red_img[:] = (0, 0, 255)
    result_fg = cv2.bitwise_and(red_img, red_img, mask=mask)
    img = cv2.add(result_bg, result_fg)

    return img

def makeGraphRoots(seenCentroidsIdx):

    g = gt.Graph()
    g.vp["partInd"] = g.new_vp("int")
    g.vp["frameInd"] = g.new_vp("int")
    g.vp["partDist"] = g.new_vp("double")
    g.vp["partDistToRoot"] = g.new_vp("double")
    g.vp["partDistToParent"] = g.new_vp("double")
    g.vp["isRoot"] = g.new_vp("bool")

    for i in range(len(seenCentroidsIdx[1])): #Iterate on frames
        #print('In image: ', seenCentroidsIdx[1][i])
        #print('SP index: ', seenCentroidsIdx[0][i])

        v = g.add_vertex(1)
        #v_roots.append(v)
        g.vp["isRoot"][v] = True
        #print v_roots[i]
        #vprop_dist[v_roots[i]] = 0
        g.vp["partDistToRoot"][v] = 0
        g.vp["partDistToParent"][v] = 0
        #vprop_frame[v_roots[i]] = seenCentroidsIdx[1][i]
        g.vp["frameInd"][v] = seenCentroidsIdx[1][i]
        #vprop_sp[v_roots[i]] = seenCentroidsIdx[0][i]
        g.vp["partInd"][v] = seenCentroidsIdx[0][i]

    return(g)

def buildTreesNeighborhood(g,winWidth,avgDesc,spLabels,w_norm,h_norm,myGaze):
    """
    Build trees with superpixels within (w_norm, h_norm) neighborhood centered at gaze point
    """

    centroids = getLabelCentroids(spLabels)

    #Create neighborhood's dimensions
    w = w_norm*spLabels[0].shape[1]
    h = h_norm*spLabels[0].shape[0]

    # Iterate over roots (seen superpix)
    for v in g.vertices():
        #Forward matching
        #print("Forward matching with root at frame:", g.vp["frameInd"][v])
        thisRoot = v
        idx = np.arange(g.vp["frameInd"][thisRoot],(winWidth)) + 1
        thisV = v
        #print idx
        for i in range(len(idx)):
            #Find closest SP
            minDist,indMinDist = getNearestSP(avgDesc,g.vp["frameInd"][thisRoot],g.vp["partInd"][thisRoot],idx[i])
            thisMatchV = g.add_vertex()
            e = g.add_edge(thisV,thisMatchV)
            g.vp["frameInd"][thisMatchV] = idx[i]
            g.vp["partInd"][thisMatchV] = indMinDist
            g.vp["partDistToRoot"][thisMatchV] = minDist
            thisV = thisMatchV
        #Backward matching
        #print "Backward matching with root at frame:", g.vp["frameInd"][v]
        idx = np.arange(0,g.vp["frameInd"][thisRoot]) 
        idx = idx[::-1]
        thisV = v
        #print idx
        for i in range(len(idx)):
            #Find closest SP
            minDist,indMinDist = getNearestSP(avgDesc,g.vp["frameInd"][thisRoot],g.vp["partInd"][thisRoot],idx[i])
            thisMatchV = g.add_vertex()
            e = g.add_edge(thisV,thisMatchV)
            g.vp["frameInd"][thisMatchV] = idx[i]
            g.vp["partInd"][thisMatchV] = indMinDist
            g.vp["partDistToRoot"][thisMatchV] = minDist
            thisV = thisMatchV

    return(g)

def buildTrees(g,winWidth,avgDesc):

    """
    Calculates distances w.r.t root
    """
    # Iterate over roots (seen superpix)
    for v in g.vertices():
        #Forward matching
        #print "Forward matching with root at frame:", g.vp["frameInd"][v]
        thisRoot = v
        idx = np.arange(g.vp["frameInd"][thisRoot],(winWidth)) + 1
        thisV = v
        #print idx
        for i in range(len(idx)):
            #Find closest SP
            minDist,indMinDist = getNearestSP(avgDesc,g.vp["frameInd"][thisRoot],g.vp["partInd"][thisRoot],idx[i])
            thisMatchV = g.add_vertex()
            e = g.add_edge(thisV,thisMatchV)
            g.vp["frameInd"][thisMatchV] = idx[i]
            g.vp["partInd"][thisMatchV] = indMinDist
            g.vp["partDist"][thisMatchV] = minDist
            thisV = thisMatchV
        #Backward matching
        #print "Backward matching with root at frame:", g.vp["frameInd"][v]
        idx = np.arange(0,g.vp["frameInd"][thisRoot]) 
        idx = idx[::-1]
        thisV = v
        #print idx
        for i in range(len(idx)):
            #Find closest SP
            minDist,indMinDist = getNearestSP(avgDesc,g.vp["frameInd"][thisRoot],g.vp["partInd"][thisRoot],idx[i])
            thisMatchV = g.add_vertex()
            e = g.add_edge(thisV,thisMatchV)
            g.vp["frameInd"][thisMatchV] = idx[i]
            g.vp["partInd"][thisMatchV] = indMinDist
            g.vp["partDist"][thisMatchV] = minDist
            thisV = thisMatchV

    return(g)
def buildTrees_rootParent(g,winWidth,avgDesc):

    """
    Calculates distances w.r.t root and nearest parent (takes min value)
    Exp. 2
    """
    # Iterate over roots (seen superpix)
    for v in g.vertices():
        #Forward matching
        #print "Forward matching with root at frame:", g.vp["frameInd"][v]
        thisRoot = v
        idx = np.arange(g.vp["frameInd"][thisRoot],(winWidth)) + 1
        thisV = v
        #print idx
        for i in range(len(idx)):
            #Find closest SP
            minDist,indMinDist = getNearestSP(avgDesc,g.vp["frameInd"][thisRoot],g.vp["partInd"][thisRoot],idx[i])
            thisMatchV = g.add_vertex()
            e = g.add_edge(thisV,thisMatchV)
            g.vp["frameInd"][thisMatchV] = idx[i]
            g.vp["partInd"][thisMatchV] = indMinDist
            g.vp["partDistToRoot"][thisMatchV] = minDist
            if (e.source() == thisRoot):
                g.vp["partDistToParent"][thisMatchV] = 1
            else:
                thisParentFrame = g.vp["frameInd"][e.source()]
                thisParentPartInd = g.vp["partInd"][e.source()]
                g.vp["partDistToParent"][thisMatchV] = np.linalg.norm(avgDesc[thisParentFrame][thisParentPartInd] - avgDesc[idx[i]][indMinDist])
            g.vp["partDist"][thisMatchV] = g.vp["partDistToParent"][thisMatchV]*g.vp["partDistToRoot"][thisMatchV]
            thisV = thisMatchV
        #Backward matching
        #print "Backward matching with root at frame:", g.vp["frameInd"][v]
        idx = np.arange(0,g.vp["frameInd"][thisRoot])
        idx = idx[::-1]
        thisV = v
        #print idx
        for i in range(len(idx)):
            #Find closest SP
            minDist,indMinDist = getNearestSP(avgDesc,g.vp["frameInd"][thisRoot],g.vp["partInd"][thisRoot],idx[i])
            thisMatchV = g.add_vertex()
            e = g.add_edge(thisV,thisMatchV)
            g.vp["frameInd"][thisMatchV] = idx[i]
            g.vp["partInd"][thisMatchV] = indMinDist
            g.vp["partDistToRoot"][thisMatchV] = minDist
            if (e.source() == thisRoot):
                g.vp["partDistToParent"][thisMatchV] = 1
            else:
                thisParentFrame = g.vp["frameInd"][e.source()]
                thisParentPartInd = g.vp["partInd"][e.source()]
                g.vp["partDistToParent"][thisMatchV] = np.linalg.norm(avgDesc[thisParentFrame][thisParentPartInd] - avgDesc[idx[i]][indMinDist])
            g.vp["partDist"][thisMatchV] = g.vp["partDistToParent"][thisMatchV]*g.vp["partDistToRoot"][thisMatchV]
            thisV = thisMatchV

    return(g)

def makeSPscores(g,spLabels,eps):
    spDists = np.inf*np.ones((len(spLabels),spLabels[0].shape[0],spLabels[0].shape[1]),np.double)

    #print "Adding distances from parent and/or root"
    count = 1
    for v in g.vertices():
            #print count, "/", g.num_vertices()
            count += 1
            mask = (spLabels[g.vp["frameInd"][v]] == g.vp["partInd"][v])
            whereMaskIsTrueI = np.where(mask==True)[0]
            whereMaskIsTrueJ = np.where(mask==True)[1]
            previousDist = spDists[g.vp["frameInd"][v],whereMaskIsTrueI[0],whereMaskIsTrueJ[0]]
            #spDists[g.vp["frameInd"][v],:,:] += (1.*mask)*g.vp["partDist"][v]
            if g.vp["partDist"][v] < previousDist:
                spDists[g.vp["frameInd"][v],whereMaskIsTrueI,whereMaskIsTrueJ] = g.vp["partDist"][v]

    #print "Threshold distances"
    thr = np.ones((len(spLabels),spLabels[0].shape[0],spLabels[0].shape[1]),np.double)*eps
    return(1./np.maximum(thr,spDists),spDists)

def normalizeDists(g,minDist,maxDist):
    for v in g.vertices():
        g.vp["partDistToRoot"][v] = (g.vp["partDistToRoot"][v]-minDist)/(maxDist-minDist)

    return g

def normalizeDists_root_parent(g,minDist,maxDist):
    for v in g.vertices():
        g.vp["partDist"][v] = (g.vp["partDist"][v]-minDist)/(maxDist-minDist)

    return g

def normalizeDists_rootParent(g,minDist,maxDist):
    for v in g.vertices():
        g.vp["partDistToRoot"][v] = (g.vp["partDistToRoot"][v]-minDist)/(maxDist-minDist)

    return g

def thresholdGraph(g_arg,thr):

    numVertBefore = g_arg.num_vertices()
    g_thr = g_arg
    v_toremove = []
    if thr < 1:
        for v in g_thr.vertices():
            #if vprop_dist[v] > thr:
            if g_thr.vp["partDistToRoot"][v] > thr:
                #print "Found vertex to delete with distance: ", vprop_dist[v]
                v_toremove.append(v)

        for v in reversed(sorted(v_toremove)):
                #print "Deleting vertex: ", v
                g_thr.remove_vertex(v)

    #print "Vertices before/after: ", numVertBefore, "/", g_thr.num_vertices()

    return g_thr


def makeSegmentedImage(img,maskSegm,maskContour):

    white_img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    white_img[:] = (255, 255, 255)
    mask_inv = cv2.bitwise_not(maskContour)
    img = cv2.bitwise_and(img,img,mask=mask_inv)
    result_fg = cv2.bitwise_and(white_img, white_img, mask=maskSegm)
    imgMaskedSegm = cv2.add(img,result_fg)

    return imgMaskedSegm

