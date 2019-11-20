"""

Inference engine detector
 
"""
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class InferenceEngineDetector:
    def __init__(self, weightsPath = None, configPath = None, device = 'CPU', extension = 
                 'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll'):
        ie = IECore()
        if extension:
            ie.add_extension(extension)
        net = IENetwork(model = configPath, weights = weightsPath)
        exec_net = self.ie.load_network(network = self.net, device_name = device)
        return

    def draw_detection(self, detections, img):
    
        #
        # Add your code here
        #
        
        return img

    def _prepare_image(self, image, h, w):
        image = image.transpose((2, 0, 1))
        image = cv2.resize(image, dsize = (h, w))
        
        return image
        
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs = {input_blob: blob})
        output = output[out_blob]

        detection = None
        
        return detection