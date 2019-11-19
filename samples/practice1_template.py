import sys
import cv2
import logging as log
import argparse

sys.path.append('../src')
from imagefilter import ImageFilter

def build_argparse():
    parser=argparse.ArgumentParser()
    
    #
    # Add your code here
    #
    
    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Hello image filtering")
    args = build_argparse().parse_args()
    
    #
    # Add your code here
    #
    
    return
    
if __name__ == '__main__':
    sys.exit(main())