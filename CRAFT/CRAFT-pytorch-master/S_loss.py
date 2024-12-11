import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
	def __init__(self):
		super(Loss, self).__init__()

	def forward(self, gt_region, pred_region, gt_affinity, pred_affinity, conf_map):
        # gt_region과 gt_affinity가 3D인 경우 채널 차원 추가
		if len(gt_region.shape) == 3:
			gt_region = gt_region.unsqueeze(1)
		if len(gt_affinity.shape) == 3:
			gt_affinity = gt_affinity.unsqueeze(1)

		print(f"gt_region shape after unsqueeze: {gt_region.shape}")
		print(f"gt_affinity shape after unsqueeze: {gt_affinity.shape}")

        # conf_map 크기 확인 및 조정
		if conf_map.dim() < gt_region.dim():
			conf_map = conf_map.unsqueeze(1)  # 채널 차원 추가

        # conf_map 크기를 gt_region 크기와 맞추기
		if conf_map.shape[2:] != gt_region.shape[2:]:
			conf_map = F.interpolate(conf_map, size=(gt_region.shape[2], gt_region.shape[3]), mode='nearest')

        # pred_region의 공간 크기 조정
		if pred_region.shape[2:] != gt_region.shape[2:]:
			pred_region = F.interpolate(pred_region, size=(gt_region.shape[2], gt_region.shape[3]), mode='nearest')

        # pred_region의 채널 크기 조정 (필요한 경우)
		if pred_region.shape[1] != gt_region.shape[1]:
			pred_region = pred_region.mean(dim=1, keepdim=True)  # 채널 평균화하여 gt_region과 동일한 채널로 맞춤

        # pred_affinity와 gt_affinity의 공간 크기 맞추기
		if pred_affinity.shape[2:] != gt_affinity.shape[2:]:
			pred_affinity = F.interpolate(pred_affinity, size=(gt_affinity.shape[2], gt_affinity.shape[3]), mode='nearest')

		# print(f"gt_region shape: {gt_region.shape}")
		# print(f"gt_affinity shape: {gt_affinity.shape}")
		# print(f"pred_region shape: {pred_region.shape}")
		# print(f"conf_map shape: {conf_map.shape}")

        # conf_map 크기 확장
		conf_map = conf_map.expand_as(gt_region)

        # 손실 계산
		loss = torch.mean(((gt_region - pred_region).pow(2) + (gt_affinity - pred_affinity).pow(2)) * conf_map)
		return loss
	
def get_loss(gt, pred, conf_map, neg_ratio, pos_min):
	b, c, h, w = gt.size()
	gt_pos_area = (gt > pos_min).float().view(-1)
	gt_pos_num = gt_pos_area.sum()
	gt_neg_num = b * c * h * w - gt_pos_num
	gt_neg_num = torch.min(gt_neg_num, neg_ratio * gt_pos_num)

	loss = ((gt - pred).pow(2) * conf_map).view(-1)
	pos_loss = loss * gt_pos_area
	neg_loss = loss * (1 - gt_pos_area)

	value, _ = torch.topk(neg_loss, int(gt_neg_num.item()), sorted=False)
	ohem_loss = value.sum() + pos_loss.sum()
	return ohem_loss / (gt_neg_num + gt_pos_num)
		

class Loss_OHEM(nn.Module):
	def __init__(self, neg_ratio, pos_min):
		super(Loss_OHEM, self).__init__()
		self.neg_ratio = neg_ratio
		self.pos_min = pos_min

	def forward(self, gt_region, pred_region, gt_affinity, pred_affinity, conf_map):
		region_loss = get_loss(gt_region, pred_region, conf_map, self.neg_ratio, self.pos_min)
		affinity_loss = get_loss(gt_affinity, pred_affinity, conf_map, self.neg_ratio, self.pos_min)
		print('region loss is {}, affinity loss is {}'.format(region_loss, affinity_loss))
		return region_loss + affinity_loss
