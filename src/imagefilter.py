import cv2

class ImageFilter():
    def __init__(self, gray = False, shape = None, crop = None):
        self.gray = gray
        if shape:
            self.shape = shape
        if crop:
            self.crop = crop
    
    def process_image(self, image):
        #
        # Put your code here, like example
        #
        #if self.shape:
        #    do something
        # 
        return image