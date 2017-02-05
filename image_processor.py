import cv2
import numpy as np

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
    
    min_width = ntproperty('/camera/min_width', 5)
    min_height = ntproperty('/camera/min_height', 10)
    
    thresh_hue_lower = ntproperty('/camera/thresholds/hue_low', 60)
    thresh_hue_high = ntproperty('/camera/thresholds/hue_high', 100)
    thresh_sat_lower = ntproperty('/camera/thresholds/sat_low', 150)
    thresh_sat_high = ntproperty('/camera/thresholds/sat_high', 255)
    thresh_val_lower = ntproperty('/camera/thresholds/val_low', 140)
    thresh_val_high = ntproperty('/camera/thresholds/val_high', 255)
    
    square_tolerance = ntproperty('/camera/square_tolerance', 10)
    broken_tolerance_x = ntproperty('/camera/broken_tolerance_x', 2)
    broken_tolerance_y = ntproperty('/camera/broken_tolerance_y', 20)
    
    gear_spacing = ntproperty('/camera/gear_spacing', 2)
    
    draw_thresh = ntproperty('/camera/draw_thresh', True)
    draw_approx = ntproperty('/camera/draw_approx', False)
    draw_approx2 = ntproperty('/camera/draw_approx2', False)
    draw_contours = ntproperty('/camera/draw_contours', False)
    draw_gear_patch = ntproperty('/camera/draw_gear_patch', False)
    draw_gear_target = ntproperty('/camera/draw_gear_target', True)
    
    def __init__(self):
        self.size = None
        self.thresh_low = np.array([self.thresh_hue_lower, self.thresh_sat_lower, self.thresh_val_lower], dtype=np.uint8)
        self.thresh_high = np.array([self.thresh_hue_high, self.thresh_sat_high, self.thresh_val_high], dtype=np.uint8)
        
        self.nt = NetworkTable.getTable('/camera')
        
    def preallocate(self, img):
        if self.size is None or self.size[0] != img.shape[0] or self.size[1] != img.shape[1]:
            h, w = img.shape[:2]
            self.size = (h, w)
            
            self.img = np.empty((h, w, 3), dtype=np.uint8)
            
            self.hsv = np.empty((h, w, 3), dtype=np.uint8)
            self.bin = np.empty((h, w, 1), dtype=np.uint8)
            self.bin2 = np.empty((h, w, 1), dtype=np.uint8)
            
            self.out = np.empty((h, w, 3), dtype=np.uint8)
            
            # for overlays
            self.zeros = np.zeros((h, w, 1), dtype=np.bool)
            self.black = np.zeros((h, w, 3), dtype=np.uint8)
        
            self.morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(0,0))
        
        cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
        
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
                            
                        result.append(approx2)
                        '''
                        if len(approx2) in (4,5):
                            result.append(approx)
                            
                            if self.draw_contours:
                                cv2.drawContours(self.out, [approx], -1, self.YELLOW, 2, lineType=8)'''
                    
        return result
    
    def get_contour_info(self, contour):
        contour_info = {}
            
        contour_info['x'], contour_info['y'], contour_info['w'], contour_info['h'] = cv2.boundingRect(contour)
        
        contour_info['cx'] = contour_info['x'] + contour_info['w'] / 2
        contour_info['cy'] = contour_info['y'] + contour_info['h'] / 2
        
        return contour_info
    
    def process_for_gear_target(self, contours):
        # Filter contours for complete gear targets and possible 'broken gear targets'
        self.targets = {'complete':[], 'broken':[]}
        
        for c in contours:
            target_info = self.get_contour_info(c)
            target_info['cnt'] = c
            
            # Check for square-ish blobs (might indecate broken gear target)
            if target_info['w'] >= (target_info['h'] - self.square_tolerance) and target_info['w'] <= (target_info['h'] + self.square_tolerance):
                self.targets['broken'].append(target_info)
            elif target_info['w'] < target_info['h']:
                self.targets['complete'].append(target_info)
        
        # Groups contours together if within a certain tolerance
        for i, b in enumerate(self.targets['broken'][:]):
            
            if len(self.targets['broken']) == 0:
                break
            
            for b2 in self.targets['broken'][i+1:]:
                if b['cx'] >= b2['cx'] - self.broken_tolerance_x and b['cx'] <= b2['cx'] + self.broken_tolerance_x:
                    if b['cy'] >= b2['cy'] - self.broken_tolerance_y and b['cy'] <= b2['cy'] + self.broken_tolerance_y:
                        new_blob = np.concatenate([b['cnt'], b2['cnt']])
                        
                        hull = cv2.convexHull(new_blob)
                        new_blob = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
                        
                        target_info = self.get_contour_info(new_blob)
                        target_info['cnt'] = new_blob
                        
                        self.targets['complete'].append(target_info)
                        
                        self.targets['broken'] = {}
                        
                        break
        
        # Draws gears after `patching` them together 
        if self.draw_gear_patch:
            contours = []
            for g in self.targets['complete']:
                cv2.drawContours(self.out, [g['cnt']], -1, self.YELLOW, 2, lineType=8)
                contours.append(g['cnt'])
        
        # Breaks out of loop if no complete targets
        if len(self.targets['complete']) == 0:
            self.nt.putBoolean('gear_target_present', False)
            return
        
        # Finds the target that is closest to the center
        h = float(self.size[0])
        w = float(self.size[1])
        
        center_most_target = None
        for i, g in enumerate(self.targets['complete'][:]):
            greater_than = True
            
            for g2 in self.targets['complete'][i:]:
                if g['cx'] - (h / 2) < g2['cx'] - (h / 2):
                    greater_than = False
            
            if greater_than:
                center_most_target = g
                self.targets['complete'].remove(g)
                break
        
        
        # Finds the another close gear target if present
        main_target_contour = center_most_target['cnt']
        partial = True
        if len(self.targets['complete']) > 0:
            for i, g in enumerate(self.targets['complete']):
                greater_than = True
                
                if abs(g['cx'] - center_most_target['cx']) < self.gear_spacing * center_most_target['h']:
                    
                    for g2 in self.targets['complete'][i:]:
                        if g['cx'] - (h / 2) < g2['cx'] - (h / 2):
                            greater_than = False
                else:
                    greater_than = False
                
                if greater_than:
                    main_target_contour = np.concatenate([g['cnt'],main_target_contour])
                    partial = False
                    break
                
        # Preforms math on contours to make them useful
        hull = cv2.convexHull(main_target_contour)
        main_target_contour = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
        
        cnt_info = self.get_contour_info(main_target_contour)
        
        angle = self.VFOV * target_info['cy'] / h - self.VFOV/2.0
        height = self.HFOV * target_info['cx'] / w - self.HFOV/2.0
        
        self.nt.putBoolean('gear_target_present', True)
        self.nt.putBoolean('gear_target_partial', partial)
        self.nt.putNumber('gear_target_angle', angle)
        self.nt.putNumber('gear_target_height', height)
        
        if self.draw_gear_target:
            cv2.drawContours(self.out, [main_target_contour], -1, self.RED, 2, lineType=8)
    
    def process_frame(self, frame):
        self.preallocate(frame)
        
        cnt = self.find_contours(frame)
        
        self.process_for_gear_target(cnt)
            
        return self.out
        