# Inertial Based System For Real-Time Identification Of Ramp And Stairs Ascent And Descent
##  Brief description
this was my Bachelor degree graduation project .Gait analysis is a technique used in many different fields and it may be a good way to indicate some disease which effecting the walking pattern.
In addition, gait analysis can be used as an input for walking aid and prosthetic control. 
Of particular interest in gait analysis is identifying the terrain a subject is walking onto. 
This project describes the use of an inertial sensor to identify in real-time whether a subject is walking in two different terrains, namely the ramp and the stairs, in both ascending and descending motions. 
The identification of ramp and stairs ascent or descent pattern is carried using the Arduino and a single Inertial Measurement Unit IMU sensor which outputs the acceleration and angular velocity profile of the walking in three different axes. A signal conditioning circuit is designed to collect the required dataset 
to be processed and this will serve as an input for a machine learning algorithm which will be applied to identify the gait pattern in real-time.
## Attachment  
1. project report :explain the project and show the results 
2. The (The_Arduino_code_for_the_divece.ino)  has the code used to collect the data from the sensor to create the dataset which will be used for creating .Gait analysis algorithms
3. The classifier:code used to create the classifier which will be add to the mobile app
