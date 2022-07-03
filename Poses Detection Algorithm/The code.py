#Import the necessary libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import dot
from numpy.linalg import norm
import fnmatch

# Allow the user to choose  one out of three options 
n = int((input('''Please enter  1 or 2 
            \n1 : Find a similar pose for your image in our the dataset  
            \n2:  Find a similar pose for your image in any video you want
            \n enter your selection : ''')))


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
thr = 0.2
#The function used to extract the pose  
def poseDetector(frame):
    #read the img or frame and extract X and Y vector for BODY_PARTS
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert(len(BODY_PARTS) == out.shape[1])
    points = []
    cof_list=[]
    points2=[]
    points3=[]
       
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        cof_list.append(conf)
        points2.append(int(x)if conf > thr else None)
        points3.append(int(y)if conf > thr else None)
        points.append((int(x), int(y)) if conf > thr else None)
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        #Draw on the image the pose 
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (255, 0, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 255,0 ), cv.FILLED)

    #Return the image or the frame with drawing X and Y vector 
    return frame,points2,points3


#Find a similar pose in image from the user  in the dataset

if n == 1:
    your_pose = input("Enter the  address of  the image with the pose you want search for with the Extension  : ")
    # read user image and compute  the similarity  with images in dataset
    img=cv.imread(your_pose)
    imgx,x,y=poseDetector(img)
    x=[0 if v is None else v for v in x]
    y=[0 if v is None else v for v in y]
    i=0
    directory = r"C:/Users/abdoa/Desktop/homework2/POES/GT_frames"
    for path,dirs,files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename,"*.jpg"):
                f=cv.imread(os.path.join(path, filename))
                t=os.path.join(path, filename)
                t=t[38:]
                print(t)
                imgx2,x2,y2=poseDetector(f)
                x2=[0 if v is None else v for v in x2]
                y2=[0 if v is None else v for v in y2]
                # apply cousin similarity
                cos_simx = dot(x, x2)/(norm(x)*norm(x2))
                cos_simy = dot(y, y2)/(norm(y)*norm(y2))
                cos_x=[]
                cos_y=[]
                #plot the image which contain the similar pose with the user image 
                if cos_simx >.4 and cos_simy>.4:
                    fig, ax = plt.subplots(2,1,figsize=(10, 3))
                    ax[0].imshow(cv.cvtColor(imgx,cv.COLOR_BGR2RGB))
                    ax[1].imshow(cv.cvtColor(imgx2,cv.COLOR_BGR2RGB))
                    fig.suptitle(t, fontsize=16)
                    plt.show()
                print("done"+str(i))
                i=i+1
                        
           

if n==2 :
    # ask the user to enter a image and video from his/her chose 
    your_pose = input("Enter the  address of  the image with the pose you want search for with the Extension  : ")
    img=cv.imread(your_pose)
    imgx,x5,y5=poseDetector(img)
    x5=[0 if v is None else v for v in x5]
    y5=[0 if v is None else v for v in y5]
    your_vid=input("Enter the  address of  the video with the pose you want search for with the Extension  : ")
    # read the video frame by frame
    cap = cv.VideoCapture(your_vid)
    count=0
    
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  
    
        assert(len(BODY_PARTS) == out.shape[1])
    
        points = []
        points2=[]
        points3=[]
        for i in range(len(BODY_PARTS)):
            
            heatMap = out[0, i, :, :]
    
            # "Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way." comment from the open source code 
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x6 = (frameWidth * point[0]) / out.shape[3]
            y6 = (frameHeight * point[1]) / out.shape[2]
            points2.append(int(x6)if conf > thr else None)
            points3.append(int(y6)if conf > thr else None)
            points.append((int(x6), int(y6)) if conf > thr else None)
    
        
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)
    
            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
        #Draw the pose on the image 
            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        points2=[0 if v is None else v for v in points2]
        points3=[0 if v is None else v for v in points2]
        
        cos_simx = dot(x5, points2)/(norm(x5)*norm(points2))
        cos_simy = dot(y5, points3)/(norm(y5)*norm(points3))
        # get the time stamp 
        milliseconds = cap.get(cv.CAP_PROP_POS_MSEC)
    
        seconds = milliseconds//1000
        milliseconds = milliseconds%1000
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = seconds//60
            seconds = seconds % 60
    
        if minutes >= 60:
            hours = minutes//60
            minutes = minutes % 60
        countf=count%30
        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        #write the time stamp on the frame
        cv.putText(frame,("frame count "+str(countf)),(50, 50),cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),thickness = 5)
        cv.putText(frame,("min"+str(minutes)+"sec"+str(seconds)),(50, 100),cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),thickness = 5)
       #apply cousin similarity and save the similar image  
        print("the X  cousin similarity is \n")
        print(cos_simx)
        print("the Y cousin similarity is \n")
        print(cos_simy)
        if cos_simx >.8 and cos_simy>.7:
            cv.imwrite("frame%d.jpg" % count, frame)
            imgx2=cv.imshow('OpenPose using OpenCV', frame)
             
        count += 1
        # save frame as JPEG file
        

    
else:
   print("Invalid input, pleas try again ")   
    
    
    
''' the pose extraction process  done with the help with this open sources code
https://github.com/quanhua92/human-pose-estimation-opencv/blob/master/openpose.py   '''   
        
    
        
