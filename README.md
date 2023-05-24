# kinematics

**overlay_pred_to_video.py** uses a time series classification model (model_Catch22_bird_plant_plane_30.pkl) to predict the object type (for now only planes, birds and plants) based on the object's movement and speed. The prediction is overlaid on the video. The input are annotated videos and the annotations.json file from simpletracker.

**overlay_pred_to_video_frames.py**: uses Catch22 time series classification models to predict the object type (for now only planes, birds and plants) based on the object's movement and speed for sequences of 5, 10, 15, 20, 25 and 30 frames. The prediction is overlaid on the video. The input are annotated videos and the annotations.json file from simpletracker.

**model_Catch22_bird_plane_cloud_raindrop_x.pkl** and **model_Catch22_bird_plant_plane_x.pkl** contain a trained random forest classifier based on the Catch22 algorithm. The models were trained on sequences of x=5,10,15,20,25,30 frames to make predictions. It can be used in overlay_pred_to_video.py, only the file path has to be adjusted. The accuracies for **model_Catch22_bird_plane_cloud_raindrop_x.pkl** are ~0.74,0.77,0.81,0.83,0.86 and 0.87 (lowest/highest accuracy for lowest/highest number of frames).

The notebook **training_data_model_fit.ipynb** can be used to generate input data for model training, fitting and testing time series classification models.

**Literature**: The article describing the Catch22 algorithm is found here (open access): https://link.springer.com/article/10.1007/s10618-019-00647-x


# Testing visualization of codebase
[Link to visualization](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=Sky360-Repository%2Fkinematics)
