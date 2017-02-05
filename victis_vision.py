#!/usr/bin/env python3

try:
    import cscore as cs
    CSCORE=True
except ImportError:
    CSCORE=False
    
import cv2
import numpy as np

from enum import Enum
from image_processor import ImageProcessor
import argparse

from networktables import NetworkTable
from networktables.util import ntproperty


class VisionMode:
    CSCORE_STREAM_ONLY = 1
    CSCORE_WITH_STREAM = 2
    PHOTO_WITH_IMSHOW = 3
    
class VictisVision:
    
    enabled = ntproperty('/camera/enabled', False)
    
    def __init__(self, *args, **kwargs):
        self.mode = kwargs.pop("mode", VisionMode.CSCORE_WITH_STREAM)
        
        # Don't mess with these values for now.
        self.width = 320 #kwargs.pop("width", 320)
        self.height = 240 #kwargs.pop("height", 240)
        
        if self.mode == VisionMode.CSCORE_WITH_STREAM or self.mode == VisionMode.CSCORE_STREAM_ONLY:
            if not CSCORE:
                raise 'Error: cscore option selected but cscore failed to import'
        
        NetworkTable.setIPAddress(kwargs.pop("nt_address", "localhost"))
        NetworkTable.setClientMode()
        NetworkTable.initialize()
        
        self.nt = NetworkTable.getTable('/camera')
        
        self.processor = ImageProcessor()
        
        if self.mode in (1,2):
            self.setup_cscore_stream(kwargs.pop("camera_port", 0), kwargs.pop("stream_port", 8081))
        if self.mode == VisionMode.CSCORE_WITH_STREAM:
            self.setup_cscore_cv(kwargs.pop("stream_cv", True), kwargs.pop("cv_stream_port", 8082))
        if self.mode == VisionMode.PHOTO_WITH_IMSHOW:
            self.process_photo(kwargs.pop("photo_path", None))
            
        self.process()
            
    def setup_cscore_stream(self, camera_port, stream_port):
        self.camera = cs.UsbCamera("usbcam", camera_port)
        self.camera.setVideoMode(cs.VideoMode.PixelFormat.kMJPEG, self.width, self.height, 30)
        
        self.mjpegServer = cs.MjpegServer("httpserver", stream_port)
        self.mjpegServer.setSource(self.camera)
    
    def setup_cscore_cv(self, stream, port):
        if self.camera is None:
            raise 'Camera not intialized'
        
        self.cvsink = cs.CvSink("cvsink")
        self.cvsink.setSource(self.camera)
        
        if stream:
            self.cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, self.width, self.height, 30)
            self.cvMjpegServer = cs.MjpegServer("cvhttpserver", port)
            self.cvMjpegServer.setSource(self.cvSource)
            
    def process(self):
        img = np.zeros(shape=(self.height, self.width, 3), dtype=np.uint8)

        while True:
            if self.mode == VisionMode.CSCORE_WITH_STREAM:
                time, img = self.cvsink.grabFrame(img)
                
                if time == 0:
                    print("error:", self.cvsink.getError())
                    continue
                #print('got frame', self.enabled)
                if not self.enabled:
                    self.nt.putBoolean('gear_target_present', False)
                    self.cvSource.putFrame(img)
                    continue
                
                out = self.processor.process_frame(img)
                
                self.cvSource.putFrame(out)
                
    def process_photo(self, path):
        if path is None:
            raise 'photo path must be provided'
        
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))
        out = self.processor.process_frame(img)
               
        cv2.imshow('Frame', out)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        exit(0)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--nt-address", default='localhost', help="Adress of NetworkTables server")
    
    parser.add_argument("-s","--stream-only", action="store_true", default=False, help="Streams only camera output")
    parser.add_argument("-cv","--cv-stream", action="store_true", default=False, help="Streams camera output and runs it through OpenCv processing")
    parser.add_argument("-i","--image", action="store_true", default=False, help="Processes single photo image")
    
    parser.add_argument("--camera-port", default=0, help="Port of camera if using \'-s\' or \'-cv\'")
    parser.add_argument("--stream-port", default=8081, help="Port of camera stream if using \'-s\' or \'-cv\'")
    
    parser.add_argument("--stream-cv", action="store_true", default=False, help="Stream out after OpenCV processing if using \'-cv\'")
    parser.add_argument("--cvstream-port", default=8082, help="Port of OpenCV stream if using \'-cv\'")
    
    parser.add_argument("--photo-path", default=None, help="Path of photo if using \'-i\'")
    
    args = parser.parse_args()
    
    mode = None
    if args.stream_only:
        mode = VisionMode.CSCORE_STREAM_ONLY
    if args.cv_stream:
        if mode is not None:
            raise 'Multiple modes set please use only \'-i\', \'-s\', or \'-cv\''
        mode = VisionMode.CSCORE_WITH_STREAM
    if args.image:
        if mode is not None:
            raise 'Multiple modes set please use only \'-i\', \'-s\', or \'-cv\''
        
        if args.photo_path is None:
            raise 'Photo path must be passed in image mode'
        
        mode = VisionMode.PHOTO_WITH_IMSHOW
        
    if mode is None:
        raise 'No vision mode set!'
    
    
        
    vision = VictisVision(mode=mode,
                          nt_address=args.nt_address,
                          camera_port=args.camera_port,
                          stream_port=args.stream_port,
                          stream_cv=args.stream_cv,
                          cv_stream_port=args.cvstream_port,
                          photo_path=args.photo_path)
    