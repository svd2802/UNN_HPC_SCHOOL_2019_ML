"""

Inference engine detector sample
 
"""
import sys
import cv2
import argparse
import logging as log
sys.path.append('../src')
from ie_detector import InferenceEngineDetector

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = 'your cmd \ input ', type = str)
    parser.add_argument('-w', '--weights', help = 'your cmd \ input ', type = str)
    parser.add_argument('-i', '--input', help = 'your cmd \ input ', type = str)

    
    return parser

def main():
    log.basicConfig(format = "[%(levelname)s] %(message)s", level = log.INFO, 
                               stream = sys.stdout)
    log.info("Hello object detection!")
    args = build_argparse().parse_args()
    weights = args.weights
    config = args.config
    IEDetector = InferenceEngineDetector(weights, config, 'CPU')
    
    image_path = args.input
    image = cv2.imread(image_path)
    log.info(image.shape)
    
    #IEDetector.detect(image)

    cv2.imshow("Image", IEDetector.detect(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return 

if __name__ == '__main__':
    sys.exit(main()) 