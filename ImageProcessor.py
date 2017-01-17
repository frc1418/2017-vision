import cv2
import numpy as np

from networktables import NetworkTable, NumberArray
from networktables.util import ntproperty

class ImageProcessor:
    
    # Values for the lifecam-3000
    VFOV = 45.6 # Camera's vertical field of view
    HFOV = 61 # Camera's horizontal field of view
    
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    
    enabled = ntproperty('/camera/enabled', False)
    
    min_width = ntproperty('/camera/min_width', 50)
    min_height = ntproperty('/camera/min_height', 100)
    
    thresh_hue_low = ntproperty('/camera/thresholds/hue_low', 60)
    thresh_hue_high = ntproperty('/camera/thresholds/hue_high', 100)
    thresh_sat_low = ntproperty('/camera/thresholds/sat_low', 150)
    thresh_sat_high = ntproperty('/camera/thresholds/sat_high', 255)
    thresh_val_low = ntproperty('/camera/thresholds/val_low', 140)
    thresh_val_high = ntproperty('/camera/thresholds/val_high', 255)
    
    draw_thresh = ntproperty('/camera/draw_thresh', False)
    draw_contours = ntproperty('/camera/draw_contours', False)
    
    def __init__(self):
        self.size = None
        self.nt = NetworkTable.getTable('/camera')
        
        self.thresh_low = np.array([self.thresh_hue_lower, self.thresh_sat_lower, self.thresh_val_lower], dtype=np.uint8)
        self.thresh_high = np.array([self.thresh_hue_high, self.thresh_sat_high, self.thresh_val_high], dtype=np.uint8)
        
    def preallocate(self, img):
        if self.size is None or self.size[0] != img.shape[0] or self.size[1] != img.shape[1]:
            h, w = img.shape[:2]
            self.size = (h, w)
            
            self.hsv = np.empty((h, w, 3), dtype=np.uint8)
            self.bin = np.empty((h, w, 1), dtype=np.uint8)
            
            self.out = np.empty((h, w, 3), dtype=np.uint8)
            
            # for overlays
            self.zeros = np.zeros((h, w, 1), dtype=np.bool)
            self.black = np.zeros((h, w, 3), dtype=np.uint8)
            
            self.morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(1,1))
            
        cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
        
    def threshhold(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=self.hsv)
        cv2.inRange(self.hsv, self.lower, self.upper, dst=self.bin)
        
        cv2.morphologyEx(self.bin, cv2.MORPH_CLOSE, self.morphKernel, dst=self.bin, iterations=9)
        
        #Remy's Secret Code
        #1234567989ggcvcviyt7etwsrgofdyurwtfdfggdfgcztgg da'hyd;.i9;k,gt5jrmfghkhgjbnl,,bbnnnnnbgbjjihju,hul'lopijt=54wq
        
        if self.draw_thresh:
            b = (self.bin != 0)
            cv2.copyMakeBorder(self.black, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
            self.out[np.dstack((b, b, b))] = 255
            
        return self.bin
    
    def find_contours(self, img):
        
        thresh_img = self.threshold(img)
        
        _, contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            
            if len(approx) > 3 and len(approx) > 15:
                _,_,w,h = cv2.boundingRect(approx)
        
            if h > self.min_height and w > self.min_width:      
                    hull = cv2.convexHull(cnt)
                    approx2 = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
                    
                    if len(approx) in (4,5):
                        result.append(approx)
                        
                        if self.draw_other:
                            cv2.drawContours(self.out, [approx], -1, self.YELLOW, 2, lineType=8)
                    
        return results
    
    def process_image(self, img):
        if not self.enabled:
            self.nt.putBoolean('gear_target_present', False)
            self.nt.putBoolean('shoot_target_present', False)
            return img
        
        result = []
        
        self.preallocate(img)
        cnt = self.find_contours(img)
        
        return self.out
        
        
def init_filter():
    NetworkTable.setIPAddress('127.0.0.1')
    NetworkTable.setClientMode()
    NetworkTable.initialize()
    
    filter = ImageProcessor()
    return filter.proces_image