# GP-Clustering

In this project, we develop an understanding of the idea for clustering with Gaussian process models, according to the work of Hyun-Chul Kim and Jaewook Lee. Based on that, we implement a Gaussian Process Clustering Python Package, and perform some clustering tests with different datasets.


More details about this project can be found in the final report:
http://pan.baidu.com/s/1kTpIdgz
==============

READ ME

Project in Artificial Intelligence and Machine Learning WS 2013/14
	-- Clustering Algorithm based on Gaussian Process
	
1.File Index:
	1.1. Folders:
		Dataset  -- stores dataset files
		Markers -- stores the markers with differernt colors for GMap Clustering Application 
		results -- stores the clustering results of ACT Schools
	1.2. Notebook:
		GPClustering_Impl.ipynb  -- Implementation of GPClustering Package
		GPClustering_Variance.ipynb -- Variance Test with Differernt Covariance Function Parameters
		GPClustering_Test.ipynb -- GP Clustering Test with different datasets 
		GPClustering_Test_HighDimension.ipynb -- GP Clustering Test with high dimensional dataset
		KMeans_Test.ipynb -- KMeans Clustering Test with different datasets
	1.3 Other Files:
		GPClustering.py -- The python package of our implemented GPClustering, which can be exported in notebook GPClustering_Impl.ipynb  
		schools.html -- The Clustering Application of ACT Schools

2. HOW TO:

2.1. Intallation of Ipython notebook
	A installation Guide: http://ipython.org/install.html
	If you are in windows, you can just download Ipython.exe in the following link: http://www.lfd.uci.edu/~gohlke/pythonlibs/#ipython
2.2. Install Python Dependency Packages if some needed packages are missing
2.3. Go to the root path of the project, and start ipython notebook with the command:
	>ipython notebook --pylab inline
2.4. Open notebooks and run the codes (for some big dataset, the clustering test might take some time..)
	The Package GPClustering.py which is used in clustering test can be re-downloaded in the notebook GPClustering_Impl.ipynb, if it is missing
2.5. Open schools.html with IE Explore to see the Application of ACT Schools (Chrome and Firefox cannot work beause of ActiveX) 

3. Dataset Download: 

Shape Sets: R15, path-based2:spiral
UCI datasets: Iris
	http://cs.joensuu.fi/sipu/datasets/
		
ACT Schools:	
	https://www.data.act.gov.au/Education/ACT-School-Locations/q8rt-q8cy
	


OPENSOURCE
Copyright (c) 2014 [Fangzhou Yang]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

CONTACT
Fangzhou Yang: fangzhou.sjtu@gmail.com
