# Poses Detection Algorithm
## Brief description
This project is a Poses Detection application where a user can provide a query pose image and the application returns similar pose frames from a video or from  images in a dataset. You can test the code with this [dataset](https://www.robots.ox.ac.uk/~vgg/data/pose/) .
## Workflow 
The  application allow the user to choose one out of two options, you  give the directory and location of the image  want to find pose similar to it,the application either to find similar images in local dataset of images or in a video form the user selection.For any  choices the code work as the following 
1. Extract poses from the query image and the video
2. Compute similarity between query pose and frame poses for ranking; and
3. Display the retrieval results.

## Note
The pose extraction process done with the help with this open sources [code](https://github.com/quanhua92/human-pose-estimation-opencv/blob/master/openpose.py)  so You need to download it in order to run the application.
