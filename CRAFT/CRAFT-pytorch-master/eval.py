import time
import torch
import subprocess
import os
from craft import CRAFT
import numpy as np
import shutil
import argparse


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
parser.add_argument('--trained_model', default='/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./data/test_img', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--score_img_folder', default='../Image - 복사본/test_img', type=str, help='folder path to score images')
parser.add_argument('--score_txt_folder', default='/mnt/e/Blur_Dot_Craft_train_s/result_train_txt/', type=str, help='folder path to score txt')
# parser.add_argument('--score_img_folder', default='../Image - 복사본/test_img', type=str, help='folder path to score images')
# parser.add_argument('--score_txt_folder', default='./submit', type=str, help='folder path to score txt')

args = parser.parse_args()

def eval_model(model_name, test_img_path, submit_path, save_flag=True):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CRAFT()
	model.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
	model.eval()
	
	start_time = time.time()
	os.chdir(submit_path)
	res = subprocess.getoutput('dir')
	# print(res)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	# print('zip 완료')
	res = subprocess.getoutput('mv submit.zip ../')
	# print('이동 완료')
	os.chdir('../')
	res = subprocess.getoutput('dir')
	# print(res)
	# print('이동완료')
	
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = args.trained_model
	test_img_path = args.score_img_folder
	submit_path = args.score_txt_folder
 
	eval_model(model_name, test_img_path, submit_path)
