# kinematics
overlay_pred_to_video.py uses a time series classification model to predict the object type (for now only planes, birds and plants) based on the object's movement and speed. The prediction is overlaid on the video.
model_Catch22_bird_plant_plane_30.pkl contains a trained random forest classifier based on the Catch22 algorithm. This was trained on sequences of 30 frames to make predictions. It can be used in overlay_pred_to_video.py, only the file path has to be adjusted.
The notebook training_data_model_fit.ipynb can be used to generate input data for model training, fitting and testing time series classification models.
