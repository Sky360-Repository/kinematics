
from matplotlib.font_manager import json_load
from utility import *


def direction_change_distances(video_files,Object,sequence_length):
    
    bbox_center = []
    vector_list = []
    dist_list = []
    direction_change_list = [0]
    k_list = []

    # loop through videos
    for k in range(0,len(video_files)):
        if video_files["Object"][k]==Object:
            track = video_files["Track"][k]
            df = json_load(video_files["File paths"][k]+"\\annotations.json")
            height,width,fps = video_info(video_files["File paths"][k]+"\\annotated_video.mp4")
            # in some cases cv2 can't get the correct fps and returns 0.0
            # for now we'll assume fps=30 in this case, but need a better fix
            if fps == 0.0:
                fps = 30.0
            print("video resolution = "+str(height)+"x"+str(width))
        else:
            continue

        # check if track length is equal or longer than required sequence length
        t = 0
        for l in range(0,len(df["frames"])):
            annotations = df["frames"][l]["annotations"]
            for o in range(0,len(annotations)):
                if annotations[o]["track_id"] == track:
                    t +=1

        # when the track length is too short go on with the next video
        if t < sequence_length:
            print("Track of video with number "+str(k)+" is too short")
            continue

        t2 = 0
        # loop through frames
        for i in range(0,len(df["frames"])):
            annotations = df["frames"][i]["annotations"]
            # loop through annotations
            for j in range(0,len(annotations)):
                
                # check if there are enough images left for the required sequence
                if t-t2 < sequence_length:
                    #print("i= "+str(i))
                    #print("t="+str(t))
                    #print("t2="+str(t2))
                    break

                if annotations[j]["track_id"] == track:
                    bbox = df["frames"][i]["annotations"][j]["bbox"]
                    # get center of bounding box as tuple (x,y)
                    bbox_center.append(center_of_mass(bbox))
                    k_list.append(k)
                    t2+=1
                    #print(len(bbox_center))
                    if len(bbox_center)>3:
                        # calculate distance and direction vector
                        #print(i)
                        #print(len(bbox_center))
                        if k_list[-2] == k_list[-1]:
                            _,dist = distance_vector(bbox_center[-1],bbox_center[-2],height,width,fps)
                            dist_list.append(dist)
                        if k_list[-4] == k_list[-1]:
                            # bounding box is jittery, therefore angles can always be large from frame to frame
                            # more informative if angles are based on frames that are farther apart
                            vector,_ = distance_vector(bbox_center[-1],bbox_center[-4],height,width,fps)
                            vector_list.append(vector)
                            
                        
                    if len(vector_list) > 1 and k_list[-2]==k_list[-1]:
                        theta = direction_change(vector_list[-1],vector_list[-2])
                        direction_change_list.append(theta)

# # outlier detection
# z = np.abs(stats.zscore(dist_list))
# dist_list2 = []
# direction_change_list2 = []
# for i in range(0,len(dist_list)-1):
#     if z[i]<3:
#         dist_list2.append(dist_list[i])
#         direction_change_list2.append(direction_change_list[i])
    

    dist_list2 = dist_list[0:len(dist_list)-len(dist_list)%sequence_length]
    direction_change_list2 = direction_change_list[0:len(direction_change_list)-len(direction_change_list)%sequence_length]


    return dist_list2,direction_change_list2