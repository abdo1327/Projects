# Clustering 
## Task 
The main purpose of this project is apply different clustering methods on the two datasets and to find the best model with optimal parameters. 

## The datasets 

1. Yale Face Dataset B: This dataset contains face images under different poses and illumination conditions. In this assignment, we only use 10 classes, named from yaleB01 to yaleB10, in the Cropped Images. 
	More details about the dataset can be found at  [here](http://vision.ucsd.edu/content/extended-yale-face-database-b-b)
2. Multi-Domain Sentiment Dataset: This dataset contains reviews of thousands of Amazon products. In this assignment, we only use the book subset. The label is always at the end of each line and indicates the sentiment.
More details about the dataset can be found at [here](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/)

## References
1. For finding the optimal epsilon value graph for DBSCAN Clustering I used this [article](https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc) and [This one ](https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd)

2. I used sklearn library documentation for how to implant some algorithm namely :

   - davies_bouldin_score
   - TruncatedSVD
   - SpectralClustering
   - adjusted_rand_score
   - homogeneity_completeness_v_measure
