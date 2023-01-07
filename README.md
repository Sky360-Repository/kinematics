# kinematics

**overlay_pred_to_video.py** uses a time series classification model (model_Catch22_bird_plant_plane_30.pkl) to predict the object type (for now only planes, birds and plants) based on the object's movement and speed. The prediction is overlaid on the video. The input are annotated videos and the annotations.json file from simpletracker.

**overlay_pred_to_video_frames.py**: uses Catch22 time series classification models ( to predict the object type (for now only planes, birds and plants) based on the object's movement and speed. The prediction is overlaid on the video. The input are annotated videos and the annotations.json file from simpletracker.

**model_Catch22_bird_plant_plane_x.pkl** contains a trained random forest classifier based on the Catch22 algorithm. The models were trained on sequences of x=5,10,15,20,25,30 frames to make predictions. It can be used in overlay_pred_to_video.py, only the file path has to be adjusted.

The notebook **training_data_model_fit.ipynb** can be used to generate input data for model training, fitting and testing time series classification models.

**Literature**: The article describing the Catch22 algorithm is found here (open access): https://link.springer.com/article/10.1007/s10618-019-00647-x
