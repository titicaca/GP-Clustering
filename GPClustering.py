# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Algorithm Implementation

# <codecell>

import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import scipy
import math
from numpy import matrix

#plot the scatter plot
def plotScatter(X,Y,title,c):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X,Y,c+'.')
    ax.set_title(title)
    plt.show()

def plotCluster(X,Y,clusters,title):
#    fig = plt.figure()
 #   ax = fig.add_subplot(111)
    
    colors = np.arange(int(max(clusters)))
    t = [colors[(int)(clusters[i]) - 1]  for i in range (len(clusters))]
    plt.scatter(X, Y, c=t)
    plt.title(title)
    plt.show()
    
def plotCluster2(X,Y,clusters,title):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ['r','g','b','c','m','y']   
    c = [colors[((int)(clusters[i])-1) % len(colors)]  + '.' for i in range (len(clusters))]
    
    for i in range (len(X)):
        ax.plot(X[i],Y[i],c[i])
    ax.set_title(title)
    plt.show() 

def plotHeatMap(heatmap, extent):
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    plt.colorbar()
    plt.show()
    
#init parameters for the model
# need to be determined for different dataset
# alpha = 1
dimension = 2
v0 = 1
l =  [1. for i in range(dimension)]
v1 = 0
v2 = 1

# <markdowncell>

# ##Step 1. Construct a variance function and compute the level value $r^* = max_{x_i} \sigma ^2 (x_i) $

# <codecell>

#Calculate Covariance Matrix C
def kernelFunction(x1, x2):
    c = 0.;
    for i in range(dimension):
        c += l[i] * ((x1[i] - x2[i]) **2 )

    c = v0 * math.exp(- 0.5 * c)+ v1
    return c

#calculate the whole covariance matrix 
def getCovarianceMatrix(data):
    size = len(data[0,:])
    C = [[ 0. for i in range (size)] for j in range (size)]
    for i in range (size):
        for j in range (size):
            C[i][j] = kernelFunction(data[:,i], data[:,j])
    return matrix(C)  


#TODO check how to calculate the variance
#calculate variance of dataset
def getVariance(data):
    variance = 0
    for i in range (dimension):
        variance += np.var(data[i,:])
    return variance

#calculate inversion of Matrix A
def getInversionMatrix(data, C):
    size = len(data[0,:])
    var = getVariance(data)
    return  (C + v2 *var * matrix(np.eye(size)) ).I 

#calcalate the variance for a given point x
def varianceFunction(x, data, inv):
    size = len(data[0,:])
    k = matrix( [kernelFunction(x, data[:,i]) for i in range (size)]).T
    result = kernelFunction(x,x) - k.T * inv * k
    return result[0,0]



def getMaxVariance(data, invA):
    maxVars = 0.
    maxIndex = 0
    size = len(data[0,:])
    for i in range (size):
        if(i == 0):
            maxVar = varianceFunction(data[:,i], data, invA)
            maxIndex = 0
        else:
            tmp = varianceFunction(data[:,i], data, invA)
            if(tmp > maxVar):
                maxVar = tmp
                maxIndex = i
    return maxVar, data[:,maxIndex]

def getMaxVariances(data, invA, num):
    size = len(data[0,:])
    maxVars = [ 0. for i in range (num)]
    variances = array([varianceFunction(data[:,i], data, invA) for i in range(size)])
    sorted_vars = sort(variances)
    for i in range (num):
        maxVars[i] = sorted_vars[-1-i]
    
    return maxVars

# <markdowncell>

# ##Step 2. Compute Stable Equilibrium Points

# <codecell>

#nabla_variance_function
def nablaVarianceFunction(x, data, inv):
    size = len(data[0,:])
    k = matrix( [kernelFunction(x, data[:,i]) for i in range (size)]).T
    
    nablaK_arr = [ matrix([ - kernelFunction(x, data[:,i]) * (x[d] - data[d,i]) for i in range (size)]).T for d in range (dimension) ] 
    deltaX_arr = [ float (nablaK_arr[d].T * inv * k + k.T * inv * nablaK_arr[d]) for d in range (dimension) ]  
    
#     nablaK_x1 = matrix([ - kernelFunction(x, data[:,i]) * (x[0] - data[0,i]) for i in range (size)]).T
#     nablaK_x2 = matrix([ - kernelFunction(x, data[:,i]) * (x[1] - data[1,i]) for i in range (size)]).T
#     delta_x1 =  nablaK_x1.T * inv * k + k.T * inv * nablaK_x1 
#     delta_x2 =  nablaK_x2.T * inv * k + k.T * inv * nablaK_x2

    return deltaX_arr

#gradient descent iteration
def gradientDescentIteration(x, data, inv, ita):
    delta_x = nablaVarianceFunction(x,data,inv)
    #print "delta_x:" , delta_x
    return [ x[i] + ita * delta_x[i] for i in range (dimension)]

def getEquilibriumPoint(x,data,inv,ita,maxIteration):
    x_old = x
    iteration = 1
    for i in range(maxIteration):
        x_new = gradientDescentIteration(x_old,data,inv,ita)
        
        stopFlags = [( x_new[d] - x_old[d] ) / x_old[d] < 0.00001 for d in range (dimension)]
        
        if( all(stopFlags) ):
            break
        else:
            x_old = x_new
            iteration += 1
    print "iteration: " , iteration, "x0: ", x, "xt: ", x_new
    return x_new

def isExistInList(sepList, point, min_accepted_covariance):
    for i in range (len(sepList)):
        covariance = kernelFunction(sepList[i], point)
        if(covariance > min_accepted_covariance):
            return i
    return -1

#remove duplicate SEPs and return a reduced SEPs list
#min_accepted_covariance is set to judge if two SEPs are the same one
#if the covariance of two points are larger than the min_accepted_covariance then we determine that they are one SEP point
def reduceSEPs(seps, min_accepted_covariance):
    sepList = []
    sepIndexMap = {}
    for i in range (len(seps)):
        index = isExistInList(sepList, seps[i], min_accepted_covariance)
        if index == -1 :
            index = len(sepList)
            sepList.append(seps[i])
        sepIndexMap[i] = index
    return array(sepList), sepIndexMap

# <markdowncell>

# ##Step 3. Construct Adjacency Matrix A

# <codecell>

def getGeometricDistance(x1, x2):
    d = 0.
    for i in range (dimension):
        d += (x1[i] - x2[i])**2
    return math.sqrt(d)

def getAdjacencyMatrix(sepList, maxVar, pointsNumPerDistanceUnit, data, invA):
    A = [[ -1 for i in range (len(sepList))] for j in range (len(sepList)) ]
    for i in range (len(sepList)):
        for j in range (len(sepList)):
            if(i == j ):
                A[i][j] = 1
            elif( i < j ):
                isConnected = True
                delta = sepList[i] - sepList[j]
                distance = getGeometricDistance(sepList[i], sepList[j])
                pointsNum = pointsNumPerDistanceUnit * distance
                for m in range ((int)(pointsNum)):
                    testPoint = sepList[j] + (m+1) * delta/pointsNum
                    testVar = varianceFunction(testPoint, data, invA)
                    if(testVar > maxVar):
                        isConnected = False
                        break
                if isConnected is True:
                    A[i][j] = 1
                else:
                    A[i][j] = 0
            elif (i > j ):
                A[i][j] = A[j][i]
    return array(A)

# <markdowncell>

# ##Step 4. Assign cluster Labels to training data Points

# <codecell>

def getSEPsClusters(adjacencyMatrix, sepList):
    clusters = [ -1 for i in range (len(sepList))]
    clusterIndex = 1
    for i in range (len(sepList)):
        isNewCluster = True;
        clusterNo = clusterIndex;
        for j in range ( len(sepList)):
            if adjacencyMatrix[i][j] == 1 and clusters[j] != -1:
                isNewCluster = False
                clusterNo = clusters[j]
                break
        for j in range ( i , len(sepList)):
            if adjacencyMatrix[i][j] == 1:
                clusters[j] = clusterNo
        if isNewCluster:
            clusterIndex += 1
        
    return clusters    

def getPointClusters(sepsClusters, sepIndexMap):
    clusters = [-1 for i in range (len(sepIndexMap))]
    for i in range (len(sepIndexMap)):
        clusters[i] = sepsClusters[sepIndexMap[i]]
    return clusters

# <markdowncell>

# ##Alogorithm Evaluation

# <markdowncell>

# $n_{r,c}$ -- the number of data points that belong to reference cluster r and are assigned to cluster c by a clustering algorithm
# 
# $n$ -- the number of all the data points
# 
# $n_r$ -- the number of the data points in the reference cluster r
# 
# $n_c$ -- the number of the data points in the cluser c obtained by a clustering algorithm

# <codecell>

# count n_{r,c}

def getCountNrc(references, clusters, referencesNum, clustersNum):
    size = len(clusters)
    Nrc = [[0 for c in range(clustersNum)] for r in range(referencesNum) ]
    for r in range (referencesNum):
        for c in range (clustersNum):
            for i in range (size):
                if (references[i] -1 == r and clusters[i]-1 == c):
                    Nrc[r][c] += 1
    return Nrc

# <markdowncell>

# ###1. Reference error rate RE

# <markdowncell>

# $ RE = 1 - \frac{\sum_r {max_c n_{r,c}}}{n}$

# <codecell>

def getRE(references, clusters, referencesNum, clustersNum, Nrc):
    size = len(clusters)
    sumTmp = 0.
    for r in range (referencesNum):
        maxTmp = 0.
        for c in range (clustersNum):
            if(maxTmp < Nrc[r][c]):
                maxTmp = Nrc[r][c]
        sumTmp += maxTmp
    return 1 - sumTmp / size

# <markdowncell>

# ###2. Cluster error

# <markdowncell>

# $CE = 1 - \frac{\sum_c {max_r {n_{r,c}}}} {n}$

# <codecell>

def getCE(references, clusters, referencesNum, clustersNum, Nrc):
    size = len(clusters)
    sumTmp = 0.
    for c in range (clustersNum):
        maxTmp = 0.
        for r in range (referencesNum):
            if(maxTmp < Nrc[r][c]):
                maxTmp = Nrc[r][c]
        sumTmp += maxTmp
    return 1 - sumTmp / size

# <markdowncell>

# ###3. F Score

# <markdowncell>

# The FScore of the reference cluster r and cluster c is defined as:
# 
# $F_{r,c} = \frac {2 R_{r,c} P_{r,c}} {R_{r,c} + p_{r,c}}$ 
# 
# Where $R_{r,c} = \frac {n_{r,c}}{n_c}$ represents Recall and $P_{r,c} = \frac{n_{r,c}}{n_r}$ represents Precision

# <markdowncell>

# The FScore of the reference cluster r is the maximum FScore value over all the clusters as
# 
# $F_r = max_c F_{r,c}$

# <markdowncell>

# The overall FScore is defined as
# 
# 
# $FScore = \sum_r \frac {n_r}{n} F_r$
# 
# In general, the higher the FScore, the better the clustering result

# <codecell>

#Reference ID and ClusterID start from 1
def getFScore(references, clusters, referencesNum, clustersNum, Nrc):  
    size = len(clusters)
    
    Nr = [0. for r in range (referencesNum)]
    Nc = [0. for c in range (clustersNum)]
   
    for i in range (size):
        Nr[int(references[i]) -1 ] += 1.
        Nc[int(clusters[i]) - 1 ] += 1.

    print 'Nr', Nr
    print 'Nc', Nc
    R = [[ float(Nrc[r][c]) / Nr[r] for c in range (clustersNum)] for r in range (referencesNum)]
    P = [[ float(Nrc[r][c]) /Nc[c] for c in range (clustersNum)] for r in range (referencesNum)]
    F = [[ 2*R[r][c] * P[r][c] / (R[r][c] + P[r][c] + 0.000000000000001) for c in range (clustersNum) ] for r in range (referencesNum)]
#     print "R", array(R)
#     print "P", array(P)
#     print "F", array(F)
    
    Fr = [ 0. for r in range (referencesNum)]
    
    for r in range (referencesNum):
        maxTmp = 0.
        for c in range (clustersNum):
            if(maxTmp < F[r][c]):
                maxTmp = F[r][c]
        Fr[r] = maxTmp

#     print "Fr", array(Fr)
    
    sumTmp = 0.
    

    
    for r in range (referencesNum):
        sumTmp += Nr[r] / size * Fr[r]
        
    return sumTmp

# <markdowncell>

# ##High Dimension Virtualization

# <codecell>

#-*- coding:utf-8 -*-
from pylab import *
from numpy import *
def pca(data,nRedDim=0,normalise=1):
   
    # normalization
    m = mean(data,axis=0)
    data -= m
    # covariance matrix
    C = cov(transpose(data))
    # compute eigen vectors, decreasing ordered 
    evals,evecs = linalg.eig(C)
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    if nRedDim>0:
        evecs = evecs[:,:nRedDim]
   
    if normalise:
        for i in range(shape(evecs)[1]):
            evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
    # new data matrix
    x = dot(transpose(evecs),transpose(data))
    # recompute reduction data
    y=transpose(dot(evecs,x))+m
    return x,y,evals,evecs,indices

# <markdowncell>

# ##Test

# <codecell>

# #init
# dataset = loadtxt("dataset\R15.txt")
# data = array([dataset[:,0], dataset[:,1]])
# clusters = dataset[:,2]
# size = len(clusters)

# X = data[0,:]
# Y = data[1,:]
# dimension = 2
# l = [2,2]
# plotScatter(X,Y,"Scatter Plot", 'g')

# #calculate Covariance Matrix
# C = getCovarianceMatrix(data)
# invA = getInversionMatrix(data, C)

# grid_x = [0 + 0.25 * i for i in range (80) ] 
# grid_y = [20 - 0.25 * i for i in range (80) ] 
# variances = [[0 for i in range (len(grid_x)) ] for j in range (len(grid_y))]

# for i in range (len(grid_y)):
#     for j in range (len(grid_x)):
#         variances[i][j] = varianceFunction([grid_x[j], grid_y[i]], data, invA)
        
# heatmap = variances
# extent = [grid_x[0], grid_x[-1], grid_y[-1], grid_y[0]]

# plotHeatMap(heatmap, extent)

# <codecell>

# #get SPEs
# SEPs = array([getEquilibriumPoint(data[:,i], data, invA, 0.5, 1000) for i in range (size) ])
# sepList, sepIndexMap = reduceSEPs(SEPs, 0.99)
# print "size of reduced SEPs: " , len(sepList)
# plotScatter(sepList[:,0],sepList[:,1],"reduced SEPs Scatter Plot" ,"r")

# <codecell>

# #get adjacency matrix
# A = getAdjacencyMatrix(sepList, 0.50, 5, data, invA)
# sepsClusters = getSEPsClusters(A, sepList)
# print sepsClusters
# plotCluster(sepList[:,0],sepList[:,1],sepsClusters,"Scatter Plot")

# <codecell>

# #get clusters assignment
# results = getPointClusters(sepsClusters, sepIndexMap)
# plotCluster2(X,Y,results,"Scatter Plot")

# <codecell>

# nrc =getCountNrc(clusters,results,int(max(clusters)),int(max(results)))

# <codecell>

# getRE(clusters,results,int(max(clusters)),int(max(results)),nrc)

# <codecell>

# getCE(clusters,results,int(max(clusters)),int(max(results)),nrc)

# <codecell>

# getFScore(clusters,results,int(max(clusters)),int(max(results)),nrc)

# <codecell>

# #get adjacency matrix
# A = getAdjacencyMatrix(sepList, 0.56, 5, data, invA)
# sepsClusters = getSEPsClusters(A, sepList)
# print sepsClusters
# plotCluster(sepList[:,0],sepList[:,1],sepsClusters,"Scatter Plot")

# <codecell>

# #get clusters assignment
# results = getPointClusters(sepsClusters, sepIndexMap)
# plotCluster2(X,Y,results,"Scatter Plot")

# <codecell>

# nrc =getCountNrc(clusters,results,int(max(clusters)),int(max(results)))

# <codecell>

# getRE(clusters,results,int(max(clusters)),int(max(results)),nrc)

# <codecell>

# getCE(clusters,results,int(max(clusters)),int(max(results)),nrc)

# <codecell>

# getFScore(clusters,results,int(max(clusters)),int(max(results)),nrc)

# <codecell>


