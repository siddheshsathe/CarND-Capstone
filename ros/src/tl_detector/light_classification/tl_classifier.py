from styx_msgs.msg import TrafficLight
import cv2
import rospy
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """
        Determines traffic signal color using opencv filters
        Arguments:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        try:
            trafficLight = TrafficLight.UNKNOWN
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Here, we're using opencv for classifying the traffic signal lights.
            # From the experiments done on received images, lower and upper threshold
            # values for red color were found as below

            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])

            # Anything between HSV value [0, 50, 50] to [10, 255, 255] is considered as red

            red1 = cv2.inRange(hsv, lower_red , upper_red)

            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            red2 = cv2.inRange(hsv, lower_red , upper_red)

            converted_img = cv2.addWeighted(red1, 1.0, red2, 1.0, 0.0)

            blur_img = cv2.GaussianBlur(converted_img,(15,15),0)

            # Finding circles of red colored signals (actual dump is filtered grayscale)
            circles = cv2.HoughCircles(blur_img,cv2.HOUGH_GRADIENT,0.5,41, param1=70,param2=30,minRadius=5,maxRadius=150)

            # If found shape is a circle, it's a red signal. else it's unknown
            if circles is not None:
                trafficLight = TrafficLight.RED
        except Exception as e:
            # If there's an exception, should return as red considering safety
            trafficLight = TrafficLight.RED
            rospy.logerror(e)
        finally:
            return trafficLight