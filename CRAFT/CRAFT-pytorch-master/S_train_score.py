import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from craft import CRAFT
from S_loss import Loss
import os
import time
import numpy as np
from S_config import cfg
from S_dataset import SynthTextDataset
from S_sync_batchnorm import convert_model
import subprocess
import eval
import test_text_score

# 저장 경로
SAVE_PATH = "/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/models/"
# 저장 파일명
SAVE_FILE = "top_model_train_wbs.pth"
# 저장 모델구조 및 파라미터 모두 저장
SAVE_MODEL = "top_model_all.pth"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def train(cfg):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CRAFT()
	model = convert_model(model)
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)

	trainset = SynthTextDataset(cfg)
	train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, \
                                   num_workers=cfg.num_workers, drop_last=cfg.drop_last)
	
	file_num = len(trainset)
	batch_num = int(file_num/cfg.batch_size)
	criterion = Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
	scheduler = lr_scheduler.ReduceLROnPlateau(
				optimizer, mode="max", patience=10, verbose=True
	)
	
	cnt = 0
	f1score_history = []
	for epoch in range(cfg.epoch_iter):	
		model.train()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_region, gt_affinity, conf_map) in enumerate(train_loader):
			model.train()
			start_time = time.time()
			img, gt_region, gt_affinity, conf_map = list(map(lambda x: x.to(device), [img, gt_region, gt_affinity, conf_map]))
			pred_region, pred_affinity = model(img)
			loss = criterion(gt_region, pred_region, gt_affinity, pred_affinity, conf_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
						   
		gt_path = cfg.synthtext_gt_path  # GT 파일 경로
		pred_path = os.path.join(cfg.result_txt_path, f"submit.zip")  # 예측 결과 저장 경로

		# 1. 모델 예측값 저장 (여기서는 더미 파일 생성 예시)
		test_command = [
			"python", "test_text_score.py",
			f"--trained_model={cfg.pths_path}",
			f"--test_folder={cfg.synthtext_img_path}"
		]
		
		try:
			subprocess.run(test_command, check=True)
			print(f"Prediction results saved ")
		except subprocess.CalledProcessError as e:
			print(f"Error during prediction: {e}")

		# 2. F1 SCORE 계산 

		score_command = [
			"python","eval.py",
			f"--trained_model={cfg.pths_path}",
			f"--score_img_folder={cfg.dataset_test_path}",
			f"--score_txt_folder='/mnt/e/Blur_Dot_Craft_train_s/result_train_txt/'"
			]

		try:
			subprocess.run(score_command, check=True)
			print(f"score results success ")
		except subprocess.CalledProcessError as e:
			print(f"Error during score calculate: {e}")

		eval_params = eval.default_evaluation_params()
		try:
			resDict = eval.evaluate_method(gt_path, pred_path, eval_params)
			f1score = resDict['method']['hmean']
			f1score_history.append(f1score)
			print(f"Epoch {epoch + 1}: F1-score (H-mean) = {f1score:.4f}")
		except Exception as e:
			print(f"Error during evaluation: {e}")

		print('Epoch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
			epoch+1, cfg.epoch_iter, time.time()-start_time, loss.item()))
		scheduler.step(loss)

		print()
		print(f"scheduler.num_bad_epochs: {scheduler.num_bad_epochs}", end=" ")
		# PyTorch에서 학습률 스케줄러(Scheduler)를 사용할 때, 현재 학습률이 개선되지 않은(epoch의 손실이 향상되지 않은) 연속적인 epoch의 수를 나타내는 변수
		print(f"scheduler.patience: {scheduler.patience}")
		print()

		if len(f1score_history) == 1:
			
			# 첫번째라서 무조건 모델 파라미터 저장
			torch.save(model.state_dict(), SAVE_PATH + SAVE_FILE)

			# 모델 전체 저장
			torch.save(model, SAVE_PATH + SAVE_MODEL)

		else:
			if f1score_history[-1] >= max(f1score_history):
				torch.save(model.state_dict(), SAVE_PATH + SAVE_FILE)
				# 모델 전체 저장
				torch.save(model, SAVE_PATH + SAVE_MODEL)

		# 손실 감소(성능 개선) 안 되는 경우 조기 종료
		if scheduler.num_bad_epochs >= scheduler.patience:
			print()
			print(f"{scheduler.patience} EPOCH 성능 개선 없어서 조기 종료")
			break

	



if __name__ == '__main__':
	train(cfg.train)
