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
        classNames = { 0: 'background',
                      1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                      5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                      10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                      14: 'motorbike', 15: 'person', 16: 'pottedplant',
                      17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        return

    def draw_detection(self, detections, img):
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]   
            if confidence > 0.2: 
                class_id = int(detections[0, 0, i, 1]) 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                
                heightFactor = img.shape[0]/300.0  
                widthFactor = img.shape[1]/300.0 
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
  
                cv2.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(img, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", img)
        return img

    def _prepare_image(self, image, h, w):
        image = image.transpose((2, 0, 1))
        image = cv2.resize(image, dsize = (h, w))
        
        return image
        
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        blob = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs = {input_blob: blob})
        output = output[out_blob]

        detection = self.draw_detection(output, image)
        
        return detection