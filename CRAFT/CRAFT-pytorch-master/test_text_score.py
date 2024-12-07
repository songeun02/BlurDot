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
import score

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

""" 여러 이미지 사진들로 F1 SCORE 추출하기 위해 추가 """
def process_multiple_images(test_image_folder, ground_truth_folder, result_folder):
    # 이미지 리스트 가져오기
    image_list, _, _ = file_utils.get_files(test_image_folder)
    
    # 결과 저장을 위한 리스트
    precisions = []
    recalls = []
    f1_scores = []
    
    for k, image_path in enumerate(image_list):
        # 이미지 파일 이름 추출 (확장자 제외)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 해당 이미지의 탐지된 바운딩 박스 파일 경로
        detected_file = os.path.join(result_folder, f"craft_ic15_20k_res_{image_name}.txt")
        
        # 해당 이미지의 ground truth 파일 경로 
        ground_truth_file = os.path.join(ground_truth_folder, f"gt_{image_name}.txt")
        
        # 데이터 읽기
        detected_data = score.read_model_output(detected_file)
        ground_truth_data = score.read_ground_truth(ground_truth_file)
        
        # F1-Score 계산
        precision, recall, f1 = score.calculate_f1_score(detected_data, ground_truth_data)
        
        # 결과 저장
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"Image {image_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    
    # 평균 계산 외 다른 방법들
    return {
        'mean': {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores)
        },
        'weighted_mean': {
            'precision': np.average(precisions),
            'recall': np.average(recalls),
            'f1_score': np.average(f1_scores)
        },
        'median': {
            'precision': np.median(precisions),
            'recall': np.median(recalls),
            'f1_score': np.median(f1_scores)
        }
    }


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
parser.add_argument('--test_folder', default='../../Image', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)
print(f"Images found: {len(image_list)} in folder {args.test_folder}")
if len(image_list) == 0:
    print("No images found. Please check the folder path and ensure it contains valid images.")
    sys.exit(1)

result_img_folder = './result_img' # 블러된 이미지 저장 
result_folder = './result'
result_jpg = './result_jpg/' # 탐지한 이미지 저장 
result_txt = './result_txt/' # 탐지한 좌표 저장 

if not os.path.isdir(result_img_folder):
    os.mkdir(result_img_folder)

if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

if not os.path.isdir(result_jpg):
    os.mkdir(result_jpg)

if not os.path.isdir(result_txt):
    os.mkdir(result_txt)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    print("Before adjustment, polys:", polys)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None or not isinstance(polys[k], (list, np.ndarray)):
            polys[k] = boxes[k]  # 기본적으로 boxes 데이터를 polys에 복사

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()

    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))


    return boxes, polys, ret_score_text


if __name__ == '__main__':
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

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(
            net, image, args.text_threshold, args.link_threshold,
            args.low_text, args.cuda, args.poly, refine_net
        )

        # polys 데이터 정리
        for i in range(len(polys)):
            if polys[i] is None or not isinstance(polys[i], (list, np.ndarray)):
                polys[i] = bboxes[i]

        # save score text
        refiner_suffix = "_refined" if args.refine else ""
        render_img = score_text.copy()
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)  # Heatmap 변환
        mask_file = os.path.join(result_folder, f"res_{model_name}_{refiner_suffix}_{image_path.split('/')[-1].split('.')[0]}_mask.jpg")
        if cv2.imwrite(mask_file, ret_score_text):
            print(f"Mask saved successfully: {mask_file}")
        else:
            print(f"Failed to save mask: {mask_file}")

        # Save result with model name
        file_utils.saveResult(
            image_path, image[:, :, ::-1], polys,
            dir_jpg_name=os.path.join(result_jpg, f"{model_name}_{refiner_suffix}"),
            dir_txt_name=os.path.join(result_txt, f"{model_name}_{refiner_suffix}")
        )

        """ 블러 처리를 위해 새로 추가 """
        # 탐지된 폴리곤 영역에 블러링 적용
        blurred_image = craft_utils.blurRegionsWithLayering(image, polys)

        # 블러링된 결과 이미지 저장
        blurred_file = os.path.join(result_img_folder, f"res_{image_path.split('/')[-1].split('.')[0]}_blurred.jpg")
        if cv2.imwrite(blurred_file, blurred_image):
            print(f"Blurred result saved: {blurred_file}")
        else:
            print(f"Failed to save blurred result: {blurred_file}")


    # """ 여러 이미지 사진들로 F1 SCORE 추출하기 위해 추가 """
    # results = process_multiple_images(
    #     test_image_folder='../../Image/test_img', 
    #     ground_truth_folder='./text/test_gt', 
    #     result_folder='./result'
    # )

    # # 결과 출력
    # print("Mean Metrics:")
    # print(f"Precision: {results['mean']['precision']:.2f}")
    # print(f"Recall: {results['mean']['recall']:.2f}")
    # print(f"F1-Score: {results['mean']['f1_score']:.2f}")


    # print("elapsed time : {:.2f}s".format(time.time() - t))

