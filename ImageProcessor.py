import cv2
import numpy as np
import cscore as cs

from networktables import NetworkTable
from networktables.util import ntproperty

class ImageProcessor:
    
    # Values for the lifecam-3000
    VFOV = 45.6 # Camera's vertical field of view
    HFOV = 61 # Camera's horizontal field of view
    
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    
    enabled = ntproperty('/camera/enabled', False)
    
    min_width = ntproperty('/camera/min_width', 5)
    min_height = ntproperty('/camera/min_height', 10)
    
    thresh_hue_lower = ntproperty('/camera/thresholds/hue_low', 60)
    thresh_hue_high = ntproperty('/camera/thresholds/hue_high', 100)
    thresh_sat_lower = ntproperty('/camera/thresholds/sat_low', 150)
    thresh_sat_high = ntproperty('/camera/thresholds/sat_high', 255)
    thresh_val_lower = ntproperty('/camera/thresholds/val_low', 140)
    thresh_val_high = ntproperty('/camera/thresholds/val_high', 255)
    
    draw_thresh = ntproperty('/camera/draw_thresh', False)
    draw_approx = ntproperty('/camera/draw_approx', False)
    draw_approx2 = ntproperty('/camera/draw_approx2', False)
    draw_contours = ntproperty('/camera/draw_contours', False)
    
    def __init__(self):
        NetworkTable.setIPAddress('10.14.18.2')
        NetworkTable.setClientMode()
        NetworkTable.initialize()
        
        self.width = 320
        self.height = 240
        
        self.nt = NetworkTable.getTable('/camera')
        
        self.thresh_low = np.array([self.thresh_hue_lower, self.thresh_sat_lower, self.thresh_val_lower], dtype=np.uint8)
        self.thresh_high = np.array([self.thresh_hue_high, self.thresh_sat_high, self.thresh_val_high], dtype=np.uint8)
        
        #CSCore
        self.camera = cs.UsbCamera("usbcam", 0)
        self.camera.setVideoMode(cs.VideoMode.PixelFormat.kMJPEG, self.width, self.height, 30)
        
        self.mjpegServer = cs.MjpegServer("httpserver", 8081)
        self.mjpegServer.setSource(self.camera)
        
        self.cvsink = cs.CvSink("cvsink")
        self.cvsink.setSource(self.camera)
        
        self.cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, self.width, self.height, 30)
        self.cvMjpegServer = cs.MjpegServer("cvhttpserver", 8082)
        self.cvMjpegServer.setSource(self.cvSource)
        
        self.preallocate()
        
        #Start Image Processing
        self.process_image()
        
    def preallocate(self):
        
        self.img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        
        self.hsv = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.bin = np.empty((self.height, self.width, 1), dtype=np.uint8)
        self.bin2 = np.empty((self.height, self.width, 1), dtype=np.uint8)
        
        self.out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        
        # for overlays
        self.zeros = np.zeros((self.height, self.width, 1), dtype=np.bool)
        self.black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(0,0))
        
    def threshhold(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=self.hsv)
        cv2.inRange(self.hsv, self.thresh_low, self.thresh_high, dst=self.bin)
        
        cv2.morphologyEx(self.bin, cv2.MORPH_CLOSE, self.morphKernel, dst=self.bin2, iterations=1)
        
        #Remy's Secret Code
        #1234567989ggcvcviyt7etwsrgofdyurwtfdfggdfgcztgg da'hyd;.i9;k,gt5jrmfghkhgjbnl,,bbnnnnnbgbjjihju,hul'lopijt=54wq
        
        if self.draw_thresh:
            b = (self.bin2 != 0)
            cv2.copyMakeBorder(self.black, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
            self.out[np.dstack((b, b, b))] = 255
            
        return self.bin2
    
    def find_contours(self, img):
        
        thresh_img = self.threshhold(img)
        
        _, contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            
            if self.draw_approx:
                #print('true')
                cv2.drawContours(self.out, [approx], -1, self.BLUE, 2, lineType=8)
            
            if len(approx) > 3 and len(approx) < 15:
                _,_,w,h = cv2.boundingRect(approx)
                #
                if h > self.min_height and w > self.min_width: 
                        #print('passed H: %s W: %s' % (w,h))   
                        hull = cv2.convexHull(cnt)
                        approx2 = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
                        
                        if self.draw_approx2:
                            cv2.drawContours(self.out, [approx2], -1, self.GREEN, 2, lineType=8)
                        
                        if len(approx2) in (4,5):
                            result.append(approx)
                            
                            if self.draw_contours:
                                cv2.drawContours(self.out, [approx], -1, self.YELLOW, 2, lineType=8)
                    
        return result
    
    def process_image(self):
        while True:
            time, self.img = self.cvsink.grabFrame(self.img)
            
            if time == 0:
                print("error:", self.cvsink.getError())
                continue
            
            cv2.copyMakeBorder(self.img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
            
            if not self.enabled:
                self.nt.putBoolean('gear_target_present', False)
                self.nt.putBoolean('shoot_target_present', False)
                continue
        
            result = []
        
            cnt = self.find_contours(self.img)
        
            self.cvSource.putFrame(self.out)
        
        
if __name__ == '__main__':
    processor = ImageProcessor()