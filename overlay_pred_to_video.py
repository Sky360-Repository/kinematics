import pickle
import cv2
import numpy as np
import pandas as pd
from pytictoc import TicToc
from matplotlib.font_manager import json_load
from scipy.spatial import distance
from utility import *
import math
t = TicToc() #create instance of class

# type of font to be used later on
font = cv2.FONT_HERSHEY_SIMPLEX

# import trained classification model
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_HIVECOTEV2_bird_plant_plane_30.pkl', 'rb') as f:
    clf = pickle.load(f)

# import video
#Plane
cap = cv2.VideoCapture('E:\\Dokumente\\UFO\\vectoring_guiding\\Processed\\7-done\\7-done\\tracked\\7a313d0f-c06a-465b-9d5a-68e16fd3d378_002507\\annotated_video.mp4')
#cap = cv2.VideoCapture('E:\\Dokumente\\UFO\\vectoring_guiding\\Processed\\7-done\\7-done\\tracked\\87\\annotated_video.mp4')
#cap = cv2.VideoCapture('E:\\Sky360_videos\\5\\54a2d385-88cd-49c2-99ef-76ebb4ae616a_000871\\annotated_video.mp4')

# get annotations for video
#Plane
df = json_load('E:\\Dokumente\\UFO\\vectoring_guiding\\Processed\\7-done\\7-done\\tracked\\7a313d0f-c06a-465b-9d5a-68e16fd3d378_002507\\annotations.json')
#df = json_load('E:\\Dokumente\\UFO\\vectoring_guiding\\Processed\\7-done\\7-done\\tracked\\87\\annotations.json')
#df = json_load('E:\\Sky360_videos\\5\\54a2d385-88cd-49c2-99ef-76ebb4ae616a_000871\\annotations.json')
# track of interest
track = 5

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

bbox_center = []
vector_list = []
dist_list = []
direction_change_list = [0]
img = []
i = 0 # counter for frames
n = 0 # counter for bounding box
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        img.append(frame)


        annotations = df["frames"][i]["annotations"]
        # loop through annotations
        for j in range(0,len(annotations)):
            if annotations[j]["track_id"] == track:
                bbox = df["frames"][i]["annotations"][0]["bbox"]
                # get center of bounding box as tuple (x,y)
                bbox_center.append(center_of_mass(bbox))
                
        if len(bbox_center)>3:
            # calculate distance and direction vector

            _,dist = distance_vector(bbox_center[-1],bbox_center[-2])
            vector,_ = distance_vector(bbox_center[-1],bbox_center[-4])
            vector_list.append(vector)
            dist_list.append(dist)

            if len(vector_list) > 1:
                theta = direction_change(vector_list[-1],vector_list[-2])
                direction_change_list.append(theta)

        if len(dist_list) ==30: # apply model every 30 frames

            # direction_change_list contains NaNs here and there
            # this happens when the magnitude of a direction vector is 0
            # here it makes sense to just replace nans with 0
            direction_change_list_cleaned = []
            for j in range(0,len(direction_change_list)):
                if str(direction_change_list[j]) == 'nan':
                    direction_change_list_cleaned.append(0)
                else:
                    direction_change_list_cleaned.append(direction_change_list[j])

            # prepare dataframe with input data for classification model
            d = {}
            m = 0
            for j in range(0,30):
                if j % 30 == 0:
                  d[m] = pd.Series(direction_change_list_cleaned[j:j+30])
                  m += 1
            df1 = pd.Series(d).to_frame('Direction_changes')

            d = {}
            m = 0
            for j in range(0,30):
                if j % 30 == 0:
                  d[m] = pd.Series(dist_list[j:j+30])
                  m += 1
            df2 = pd.Series(d).to_frame('Distances')

            # input data for classification
            X_test = pd.concat([df1, df2], axis=1)
            
            
            t.tic() #Start timer
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            t.toc() #Time elapsed since t.tic()

            # empty lists to rewrite info for next 30 frames
            bbox_center = []
            vector_list = []
            dist_list = []
            direction_change_list = [0]
        i += 1
        
        # Use putText() method for
        # inserting text on video
        try:
            cv2.putText(frame,y_pred[0],(bbox[0], bbox[1]+40),font, 1,(0, 255, 0),2,cv2.LINE_4)
            if y_pred[0] == 'Plane':
                text = 'Prob:'+str(round(y_pred_proba[0][1],2))
            elif y_pred[0] == 'Bird':
                text = 'Prob:'+str(round(y_pred_proba[0][0],2))
            cv2.putText(frame,text,(bbox[0], bbox[1]+70),font, 1,(0, 255, 0),2,cv2.LINE_4)
        except:
            pass
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

#Closes all the frames
cv2.destroyAllWindows()

# write video
height,width,layers=img[0].shape
video=cv2.VideoWriter('video_plane_5_HIVECOTEV2.mp4',-1,fps=15,frameSize=(width,height))
for j in range(0,len(img)):
    video.write(img[j])
video.release()
