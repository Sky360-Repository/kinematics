# overlay_pred_to_video.py

import pickle
import cv2
import rclpy
from pytictoc import TicToc
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from utility import *

# create instance of timer
t = TicToc()

# type of font to be used later on
font = cv2.FONT_HERSHEY_SIMPLEX

def load_model(model_path):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

def process_input(msg):
    # Convert the ROS2 image message to OpenCV format and apply necessary preprocessing
    pass

def overlay_prediction(prediction, img):
    # Overlay the prediction text on the input image
    pass

def classify_image(image, clf):
    # Code for classifying the image and overlaying the result
    pass

class OverlayPredictionNode(Node):
    def __init__(self, clf):
        super().__init__('overlay_prediction_node')
        self.clf = clf
        self.bridge = CvBridge()
        self.subscriber = self.create_subscription(
            Image,
            'sky360/frames/masked',
            self.callback,
            10
        )
        self.publisher = self.create_publisher(Image, 'sky360/frames/overlayed', 10)

    def callback(self, msg):
        # Convert the ROS2 image message to an OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Classify the image and overlay the prediction
        classified_image = classify_image(image, self.clf)

        # Convert the modified OpenCV image back to a ROS2 image message
        msg_out = self.bridge.cv2_to_imgmsg(classified_image, encoding='bgr8')
        msg_out.header = msg.header

        # Publish the modified image with the overlayed prediction
        self.publisher.publish(msg_out)

def main(args=None):
    model_path = 'E:\\Dokumente\\Repos\\kinematics-main\\kinematics-main\\model_Catch22_bird_plant_plane_10.pkl'
    clf = load_model(model_path)

    rclpy.init(args=args)
    overlay_prediction_node = OverlayPredictionNode(clf)
    rclpy.spin(overlay_prediction_node)
    overlay_prediction_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
