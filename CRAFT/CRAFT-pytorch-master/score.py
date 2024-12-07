import numpy as np

def read_model_output(file_path):
    """
    모델 검출 결과 텍스트 파일에서 좌표 읽기
    
    Args:
        file_path (str): 모델 검출 결과 파일 경로
    
    Returns:
        list: 검출된 텍스트 영역의 좌표 목록
    """
    coordinates = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                # BOM 문자 제거 및 콤마로 구분된 좌표 파싱
                parts = line.strip().replace('\ufeff', '').split(',')
                
                # 8개의 좌표만 취하고 그 이후는 무시
                coords = list(map(int, parts[:8]))
                
                if len(coords) == 8:  # 4개의 좌표쌍 (x1,y1,x2,y2,x3,y3,x4,y4)
                    coordinates.append(coords)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as e:
        print(f"Error parsing coordinates in file {file_path}: {e}")
        print("Check file encoding and coordinate format")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return coordinates

def read_ground_truth(file_path):
    """
    Ground Truth 텍스트 파일에서 좌표 읽기
    
    Args:
        file_path (str): Ground Truth 파일 경로
    
    Returns:
        list: Ground Truth 텍스트 영역의 좌표 목록
    """
    return read_model_output(file_path)  # 동일한 로직 사용

def calculate_iou(box1, box2):
    """
    두 바운딩 박스의 IoU(Intersection over Union) 계산
    
    Args:
        box1 (list): 첫 번째 박스의 8개 좌표 [x1,y1,x2,y2,x3,y3,x4,y4]
        box2 (list): 두 번째 박스의 8개 좌표 [x1,y1,x2,y2,x3,y3,x4,y4]
    
    Returns:
        float: IoU 값
    """
    # 박스를 최소 둘러싸는 사각형으로 변환
    def get_bounding_rect(box):
        x_coords = box[0::2]
        y_coords = box[1::2]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    # 사각형 형태로 변환
    rect1 = get_bounding_rect(box1)
    rect2 = get_bounding_rect(box2)
    
    # 교차 영역 계산
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    
    # 교차 영역이 없는 경우
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # 교차 영역 크기
    intersection = (x2 - x1) * (y2 - y1)
    
    # 각 박스의 영역 계산
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    
    # IoU 계산
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

def calculate_f1_score(detected_data, ground_truth_data, iou_threshold=0.5):
    """
    F1 Score 계산
    
    Args:
        detected_data (list): 모델이 검출한 텍스트 영역 좌표
        ground_truth_data (list): Ground Truth 텍스트 영역 좌표
        iou_threshold (float, optional): IoU 임계값. Defaults to 0.5.
    
    Returns:
        tuple: (Precision, Recall, F1 Score)
    """
    # 매칭된 Ground Truth 개수 추적
    matched_gt = [False] * len(ground_truth_data)
    
    # True Positive 계산
    true_positives = 0
    for det_box in detected_data:
        for i, gt_box in enumerate(ground_truth_data):
            if not matched_gt[i] and calculate_iou(det_box, gt_box) >= iou_threshold:
                true_positives += 1
                matched_gt[i] = True
                break
    
    # Precision, Recall 계산
    precision = true_positives / len(detected_data) if len(detected_data) > 0 else 0
    recall = true_positives / len(ground_truth_data) if len(ground_truth_data) > 0 else 0
    
    # F1 Score 계산 (조화 평균)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score