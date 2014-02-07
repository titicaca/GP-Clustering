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
    
    colors = np.arange(len(clusters))
    t = [colors[(int)(clusters[i])]  for i in range (len(clusters))]
    plt.scatter(X, Y, c=t)
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
alpha = 1

# <markdowncell>

# ##Step 1. Construct a variance function and compute the level value $r^* = max_{x_i} \sigma ^2 (x_i) $

# <codecell>

#Calculate Covariance Matrix C
def kernelFunction(x1, x2):
    c = 0.;
    for i in range(len(x1)):
        c += alpha * (x1[i] - x2[i]) **2

    c = math.exp(- 0.5 * c)
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
    return np.var(data)

#calculate inversion of Matrix A
def getInversionMatrix(data, C):
    size = len(data[0,:])
    var = getVariance(data)
    return  (C + var * matrix(np.eye(size)) ).I 

#calcalate the variance for a given point x
def varianceFunction(x, data, inv):
    size = len(data[0,:])
    k = matrix( [kernelFunction(x, data[:,i]) for i in range (size)]).T
    result = kernelFunction(x,x) - k.T * inv * k
    return result[0,0]



def getMaxVariance():
    maxVars = 0.
    maxIndex = 0
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

def getMaxVariances(num):
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
    nablaK_x1 = matrix([ - kernelFunction(x, data[:,i]) * (x[0] - data[0,i]) for i in range (size)]).T
    nablaK_x2 = matrix([ - kernelFunction(x, data[:,i]) * (x[1] - data[1,i]) for i in range (size)]).T
    delta_x1 =  nablaK_x1.T * inv * k + k.T * inv * nablaK_x1 
    delta_x2 =  nablaK_x2.T * inv * k + k.T * inv * nablaK_x2
    return [delta_x1[0,0], delta_x2[0,0]]

#gradient descent iteration
def gradientDescentIteration(x, data, inv, ita):
    delta_x = nablaVarianceFunction(x,data,inv)
    #print "delta_x:" , delta_x
    return [ x[i] + ita * delta_x[i] for i in range (len(x))]

def getEquilibriumPoint(x,data,inv,ita,maxIteration):
    x_old = x
    iteration = 1
    for i in range(maxIteration):
        x_new = gradientDescentIteration(x_old,data,inv,ita)
        if( (x_new[0] - x_old[0])/x_old[0] < 0.00001 and (x_new[1] - x_old[1])/x_old[1] < 0.00001):
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

def reduceSEPs(seps, min_accepted_covariance):
    sepList = []
    sepIndexMap = {}
    for i in range (len(seps)):
        index = isExistInList(sepList, [seps[i,0],seps[i,1]], min_accepted_covariance)
        if index == -1 :
            index = len(sepList)
            sepList.append([seps[i,0],seps[i,1]])
        sepIndexMap[i] = index
    return array(sepList), sepIndexMap

# <markdowncell>

# ##Step 3. Construct Adjacency Matrix A

# <codecell>

def getGeometricDistance(x1, x2):
    d = 0.
    for i in range (len(x1)):
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

# <codecell>


