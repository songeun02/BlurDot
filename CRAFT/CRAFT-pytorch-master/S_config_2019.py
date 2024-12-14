from easydict import EasyDict as edict

cfg = edict()

cfg.utils = edict()
cfg.utils.eps = 1e-8
cfg.utils.sigma_den = 4
cfg.utils.gaussian_length = 100
cfg.utils.pixel_aug = 1
cfg.utils.min_box_len = 1.5
cfg.utils.fore_ratio = 0.75
cfg.utils.back_ratio = 0.05
cfg.utils.ignored_gt = ['###', '']

cfg.train = edict()
# cfg.train.synthtext_img_path = '/mnt/e/Blur_Dot/MLT/ch8_training_word_images_gt_part_2'
cfg.train.synthtext_img_path = '/mnt/e/Blur_Dot/DATA/ICDAR2019/Imgpart1/Imgpart1/tr_img1-1000/'
# cfg.train.synthtext_gt_path = '/mnt/e/Blur_Dot/MLT/coords_train_230000.txt'
cfg.train.synthtext_gt_path = '/mnt/e/Blur_Dot/DATA/ICDAR2019/icdar2019_tr_gt/icdar2019_tr_gt/tr_gt0-1000/'

cfg.train.synthtext_img_root = '/mnt/e/Blur_Dot/DATA/ICDAR2019/img' # 이미지 최상위 폴더
cfg.train.synthtext_gt_root = '/mnt/e/Blur_Dot/DATA/ICDAR2019/icdar2019_tr_gt' # 텍스트 최상위 폴더

cfg.train.result_txt_path = '/mnt/e/Blur_Dot_Craft_train_s/result_train_txt/'
cfg.train.scale = 0.5
cfg.train.crop_length = 640
cfg.train.mean = [0.485, 0.456, 0.406]
cfg.train.std = [0.229, 0.224, 0.225]
cfg.train.batch_size = 32
cfg.train.num_workers = 4
cfg.train.drop_last = True
cfg.train.shuffle = True
cfg.train.lr = 0.001
cfg.train.epoch_iter = 10
cfg.train.milestones = [0.5, 1.5]
cfg.train.gamma = 0.1
cfg.train.pths_path = '/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/weights/craft_mlt_25k.pth'
cfg.train.save_interval = 1000

cfg.ft = edict()
cfg.ft.icdar2013_img_path = '../data/ICDAR2013/train_img'
cfg.ft.icdar2013_gt_path = '../data/ICDAR2013/train_gt'
cfg.ft.icdar2017_img_path = ['/mnt/e/Blur_Dot/MLT/ch8_training_word_images_gt_part_2', '/mnt/e/Blur_Dot/MLT/ch8_validation_word_images_gt']
cfg.ft.icdar2017_gt_path = ['/mnt/e/Blur_Dot/MLT/coords_train_230000.txt', '/mnt/e/Blur_Dot/MLT/coords_val_5000.txt']
cfg.ft.ratio_17 = 10
cfg.ft.gpu_ids = [0,1,2] # first is supervision
cfg.ft.pretrain_pth = '/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/weights/craft_ic15_20k.pth'
cfg.ft.freeze_stage_num = 1 # <5
cfg.ft.conf_min = 0.5
cfg.ft.mix_prob = 0.16
cfg.ft.pos_min = cfg.utils.eps
cfg.ft.neg_ratio = 3
cfg.ft.height_jitter = 0.3
cfg.ft.color_jitter = [0.25, 0.25, 0.25, 0.125]
cfg.ft.angle_range = 15
cfg.ft.scale = 0.5
cfg.ft.min_length = 640
cfg.ft.max_length = 2240
cfg.ft.crop_length = 224
cfg.ft.mean = [0.485, 0.456, 0.406]
cfg.ft.std = [0.229, 0.224, 0.225]
cfg.ft.batch_size = 32
cfg.ft.drop_last = True
cfg.ft.shuffle = True
cfg.ft.lr = 0.0001
cfg.ft.epoch_iter = 10
cfg.ft.pths_path = '/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/models/'
cfg.ft.save_interval = 50

cfg.test = edict()
cfg.test.model_pth = '/home/songeun/LogoDetection/CRAFT_B/CRAFT-pytorch-master/weights/craft_ic15_20k.pth'
cfg.test.dataset_test_path = '/mnt/e/Blur_Dot/MLT/ch8_validation_word_images_gt'
cfg.test.submit_path = './submit'
cfg.test.save_dataset_res = True
cfg.test.region_thresh = 0.09
cfg.test.affinity_thresh = 0.07
cfg.test.remove_thresh = 6 * 1e-4
cfg.test.long_side = 960
