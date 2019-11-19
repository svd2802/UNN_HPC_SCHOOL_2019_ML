import cv2

class ImageFilter():
    def __init__(self, gray = False, shape = None, crop = None):
        
        self.gray = gray
        if shape:
            self.shape = shape
        self.crop = crop
    
    def process_image(self, image):
        if self.shape:
            image = cv2.resize(image, dsize = self.shape)
            
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        if self.crop:
            center = (self.shape[0]//2, self.shape[1]//2)
            half_length = min(center)
            left = center[1]-half_length
            right = center[1]+half_length
            bot = center[0]-half_length
            top  = center[0]+half_length
            image = image[left:right, bot:top]
        
        return image