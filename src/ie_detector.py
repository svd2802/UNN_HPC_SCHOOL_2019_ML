"""

Inference engine detector
 
"""
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class InferenceEngineDetector:
    def __init__(self, weightsPath = None, configPath = None,
                 device = 'CPU', extension = None):
        #
        # Add your code here
        #
        
        return

    def draw_detection(self, detections, img):
    
        #
        # Add your code here
        #
        
        return img

    def _prepare_image(self, image, h, w):
    
        #
        # Add your code here
        #
        
        return image
        
    def detect(self, image):
    
        #
        # Add your code here
        #
        detection = None
        
        return detection