# backend/back/video_checker.py
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import os

class RoomVideoChecker:
    def __init__(self, threshold=0.7, frame_skip=30):
        self.threshold = threshold
        self.frame_skip = frame_skip
        # Initialize your model here
    
    def is_room_video(self, video_path):
        # Implement your room detection logic
        return True  # temporary for testing