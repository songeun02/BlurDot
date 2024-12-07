# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls
    # imgs : 이미지 파일 목록 
    # masks : 마스크 파일 목록 
    # xmls : xml 파일 목록 

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path): 
        # os.walk(in_path) : 디렉토리를 순회하며 모든 파일과 하위 디렉토리 탐색 
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm': 
                # 이미지 파일 
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                # 마스크 파일 
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                # GT(Ground Truth) 관련 파일
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dir_jpg_name='./result_jpg/', dir_txt_name='./result_txt/', verticals=None, texts=None):
        # img_file : 원본 이미지 파일 경로 
        # img : 이미지 배열 
        # boxes : 텍스트 검출 결과 (Bounding Boxes or Quadrilateral(사각형) 형태)
        # verticals : 텍스트의 방향 여부 (수직, 수평 정보)
        # texts : 검출된 텍스트 

        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))
        # os.path.basename(path) : 상위 경로를 제외한 파일명만 반환 

        # result directory
        res_file = dir_txt_name + "res_" + filename + '.txt'
        res_img_file = dir_jpg_name + "res_" + filename + '.jpg'

        if not os.path.isdir(dir_jpg_name):
            os.mkdir(dir_jpg_name)

        if not os.path.isdir(dir_txt_name):
            os.mkdir(dir_txt_name)

        with open(res_file, 'w') as f:
            # boxes를 순회하며 각 박스의 좌표를 파일에 저장 
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                # 바운딩 박스를 빨간색으로 그림 
                ptColor = (0, 255, 255) # 노란색
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0) 

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    
                    # 텍스트를 이미지에 추가
                    # 검은 배경 텍스트로 그림자 효과 
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    # 실제 텍스트를 노란색으로 추가 
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)



