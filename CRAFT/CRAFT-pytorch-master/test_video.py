"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_ic15_20k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--video_folder', default='/home/songeun/LogoDetection/Video', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


# """ For test images in a folder """
# video_list, _, _ = file_utils.get_files(args.video_folder)
# print(f"Videos found: {len(video_list)} in folder {args.video_folder}")
# if len(video_list) == 0:
#     print("No videos found. Please check the folder path and ensure it contains valid videos.")
#     sys.exit(1)

result_folder = './result'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


if __name__ == '__main__':

    print("현재 작업 디렉토리:", os.getcwd())

    # load net
    net = CRAFT()  # initialize

    # Extract model name from path
    model_name = os.path.splitext(os.path.basename(args.trained_model))[0]
    
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda and torch.cuda.is_available():
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        print("CUDA is not available. Loading model on CPU...")
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=torch.device('cpu'))))

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        try:
            if args.cuda and torch.cuda.is_available():
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location=torch.device('cpu'))))
            refine_net.eval()
            args.poly = True
        except Exception as e:
            print(f"Error loading refiner model: {e}")
            sys.exit(1)

    t = time.time()

    # # load data
    # for k, image_path in enumerate(video_list):
    #     print("Test image {:d}/{:d}: {:s}".format(k+1, len(video_list), image_path), end='\r')
    #     image = imgproc.loadImage(image_path)

      
    """ 영상 처리를 위해 새로 추가 """
    # 비디오 경로 설정0 
    video_path = '/home/songeun/LogoDetection/Video/museum.mp4'
    
    # 텍스트 검출 및 블러 처리
    craft_utils.detect_and_blur_text_in_video(
        video_path, 
        net, 
        text_threshold=0.7, 
        link_threshold=0.4, 
        low_text=0.4, 
        cuda=False, 
        poly=True, 
        refine_net=refine_net
    )

    print("elapsed time : {:.2f}s".format(time.time() - t))

