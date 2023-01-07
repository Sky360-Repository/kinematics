import pickle
import cv2
import pandas as pd
from pytictoc import TicToc
from matplotlib.font_manager import json_load
from utility import *

# create instance of timer
t = TicToc() 

# type of font to be used later on
font = cv2.FONT_HERSHEY_SIMPLEX

# import trained classification models
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_Catch22_bird_plant_plane_5.pkl', 'rb') as f:
    clf5 = pickle.load(f)
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_Catch22_bird_plant_plane_10.pkl', 'rb') as f:
    clf10 = pickle.load(f)
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_Catch22_bird_plant_plane_15.pkl', 'rb') as f:
    clf15 = pickle.load(f)
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_Catch22_bird_plant_plane_20.pkl', 'rb') as f:
    clf20 = pickle.load(f)
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_Catch22_bird_plant_plane_25.pkl', 'rb') as f:
    clf25 = pickle.load(f)
with open('E:\Dokumente\Repos\kinematics-main\kinematics-main\model_Catch22_bird_plant_plane_30.pkl', 'rb') as f:
    clf30 = pickle.load(f)

# path
# Bird
path = 'E:\\Dokumente\\UFO\\vectoring_guiding\\Processed\\7-done\\7-done\\tracked\\78629d4c-4534-44d6-a22e-e995aa8b9f43_002147'

# import video
cap = cv2.VideoCapture(path+'\\annotated_video.mp4')

# get annotations for video
df = json_load(path+'\\annotations.json')

# track of interest
track = 34

# number of frames for doing predictions
sequence_length = 10

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# get video resolution
height,width,fps = video_info(path+"\\annotated_video.mp4")

bbox_center = []
vector_list = []
dist_list = []
direction_change_list = [0]
img = []
flag5=1
flag10=1
flag15=1
flag20=1
flag25=1
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

            _,dist = distance_vector(bbox_center[-1],bbox_center[-2],height,width,fps)
            vector,_ = distance_vector(bbox_center[-1],bbox_center[-4],height,width,fps)
            vector_list.append(vector)
            dist_list.append(dist)

            if len(vector_list) > 1:
                theta = direction_change(vector_list[-1],vector_list[-2])
                direction_change_list.append(theta)

        if len(dist_list) == 5 and flag5==1: # apply model every x frames
            flag5=0
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
            for j in range(0,5):
                if j % 5 == 0:
                  d[m] = pd.Series(direction_change_list_cleaned[j:j+5])
                  m += 1
            df1 = pd.Series(d).to_frame('Direction_changes')

            d = {}
            m = 0
            for j in range(0,5):
                if j % 5 == 0:
                  d[m] = pd.Series(dist_list[j:j+5])
                  m += 1
            df2 = pd.Series(d).to_frame('Distances')

            # input data for classification
            X_test = pd.concat([df1, df2], axis=1)
            
            
            t.tic() #Start timer
            y_pred = clf5.predict(X_test)
            y_pred_proba = clf5.predict_proba(X_test)
            t.toc() #Time elapsed since t.tic()

            # empty lists to rewrite info for next 30 frames
            bbox_center = []
            vector_list = []
            dist_list = []
            direction_change_list = [0]
        elif len(dist_list) == 10 and flag10==1: # apply model every x frames
            flag10=0
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
            for j in range(0,10):
                if j % 10 == 0:
                  d[m] = pd.Series(direction_change_list_cleaned[j:j+10])
                  m += 1
            df1 = pd.Series(d).to_frame('Direction_changes')

            d = {}
            m = 0
            for j in range(0,10):
                if j % 10 == 0:
                  d[m] = pd.Series(dist_list[j:j+10])
                  m += 1
            df2 = pd.Series(d).to_frame('Distances')

            # input data for classification
            X_test = pd.concat([df1, df2], axis=1)
            
            
            t.tic() #Start timer
            y_pred = clf10.predict(X_test)
            y_pred_proba = clf10.predict_proba(X_test)
            t.toc() #Time elapsed since t.tic()

        elif len(dist_list) == 15 and flag15==1: # apply model every x frames
            flag15=0
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
            for j in range(0,15):
                if j % 15 == 0:
                  d[m] = pd.Series(direction_change_list_cleaned[j:j+15])
                  m += 1
            df1 = pd.Series(d).to_frame('Direction_changes')

            d = {}
            m = 0
            for j in range(0,15):
                if j % 15 == 0:
                  d[m] = pd.Series(dist_list[j:j+15])
                  m += 1
            df2 = pd.Series(d).to_frame('Distances')

            # input data for classification
            X_test = pd.concat([df1, df2], axis=1)
            
            
            t.tic() #Start timer
            y_pred = clf15.predict(X_test)
            y_pred_proba = clf15.predict_proba(X_test)
            t.toc() #Time elapsed since t.tic()

            # empty lists to rewrite info for next 30 frames
            bbox_center = []
            vector_list = []
            dist_list = []
            direction_change_list = [0]
        elif len(dist_list) == 20 and flag20 ==1 : # apply model every x frames
            flag20=0
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
            for j in range(0,20):
                if j % 20 == 0:
                  d[m] = pd.Series(direction_change_list_cleaned[j:j+20])
                  m += 1
            df1 = pd.Series(d).to_frame('Direction_changes')

            d = {}
            m = 0
            for j in range(0,20):
                if j % 20 == 0:
                  d[m] = pd.Series(dist_list[j:j+20])
                  m += 1
            df2 = pd.Series(d).to_frame('Distances')

            # input data for classification
            X_test = pd.concat([df1, df2], axis=1)
            
            
            t.tic() #Start timer
            y_pred = clf20.predict(X_test)
            y_pred_proba = clf20.predict_proba(X_test)
            t.toc() #Time elapsed since t.tic()

        elif len(dist_list) == 25 and flag25==1: # apply model every x frames
            flag25=0
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
            for j in range(0,25):
                if j % 25 == 0:
                  d[m] = pd.Series(direction_change_list_cleaned[j:j+25])
                  m += 1
            df1 = pd.Series(d).to_frame('Direction_changes')

            d = {}
            m = 0
            for j in range(0,25):
                if j % 25 == 0:
                  d[m] = pd.Series(dist_list[j:j+25])
                  m += 1
            df2 = pd.Series(d).to_frame('Distances')

            # input data for classification
            X_test = pd.concat([df1, df2], axis=1)
            
            
            t.tic() #Start timer
            y_pred = clf25.predict(X_test)
            y_pred_proba = clf25.predict_proba(X_test)
            t.toc() #Time elapsed since t.tic()


        elif len(dist_list) == 30: # apply model every x frames

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
                if j %305 == 0:
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
            y_pred = clf30.predict(X_test)
            y_pred_proba = clf30.predict_proba(X_test)
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
video=cv2.VideoWriter('video_bird_5_Catch22_frame_models.mp4',-1,fps=15,frameSize=(width,height))
for j in range(0,len(img)):
    video.write(img[j])
video.release()
