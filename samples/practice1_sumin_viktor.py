import sys
import cv2
import logging as log
import argparse

sys.path.append('../src')
from imagefilter import ImageFilter

def build_argparse():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-in', '--input', help = 'your cmd \ input ', type = str)
    #parser.add_argument('-w', '--width', help = 'your cmd \ width ', type = str, required = False)
    #parser.add_argument('-h', '--high', help = 'your cmd \high ', type = str, required = False)
    #
    # Add your code here
    #
    
    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Hello image filtering")
    args = build_argparse().parse_args()
    
    image_path = args.input
    log.info(image_path)
    print(str(image_path))
    
    image = cv2.imread("image_path")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # Add your code here
    #
    
    return
    
if __name__ == '__main__':
    sys.exit(main())