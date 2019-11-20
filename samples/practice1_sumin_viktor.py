import sys
import cv2
import logging as log
import argparse

sys.path.append('../src')
from imagefilter import ImageFilter

def build_argparse():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', help = 'your cmd \ input ', type = str)
    parser.add_argument('-w', '--width', help = 'your cmd \ width ', type = int, default = 600, required = False)
    parser.add_argument('-l', '--high', help = 'your cmd \high ', type = int, default = 600, required = False)
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
    
    image = cv2.imread(image_path)
    log.info(image.shape)
    
    filter_gray = ImageFilter(gray = True, shape = (image.shape[1], image.shape[0]), crop = False)
    image_gray = filter_gray.process_image(image)
    cv2.imshow("Gray", image_gray)
    log.info(image_gray.shape)
    
    filter_shape = ImageFilter(gray = False, shape = (args.width, args.high), crop = False)
    image_shape = filter_shape.process_image(image)   
    cv2.imshow("Shape", image_shape)
    log.info(image_shape.shape)
    
    filter_crop = ImageFilter(gray = False, shape = (image.shape[1], image.shape[0]), crop = True)
    image_crop = filter_crop.process_image(image)
    cv2.imshow("Crop", image_crop)
    log.info(image_crop.shape)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    return
    
if __name__ == '__main__':
    sys.exit(main())