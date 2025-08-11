import cv2
import os
from screeninfo import get_monitors
import pygetwindow as gw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## =-----------------------------------------==================================================
# ===  템플릿매칭 이미지 인식을 통해 데이터와 비교해서 IOU, 신뢰성 체크 
## =-----------------------------------------==================================================


# 이미지 파일 목록 (디렉토리 내 이미지 파일 경로 가져오기)
root = os.path.dirname(os.path.realpath(__file__))

data_root = r"C:\Users\ICLAB\Desktop\Beam_Gan_Re\ImageData"


Ori_dir = os.path.join(data_root, 'TestSet_New_3500_Ori')  ## Ori 이미지
Ori_files = sorted([os.path.join(Ori_dir, f) for f in os.listdir(Ori_dir) if f.endswith(('.jpg', '.png'))])

Pro_dir = os.path.join(data_root, 'TestSet_New_3500_Pro')  ## pro 이미지
Pro_files = sorted([os.path.join(Pro_dir, f) for f in os.listdir(Pro_dir) if f.endswith(('.jpg', '.png'))])


Label_dir = os.path.join(data_root, 'TestSet_New_1000_Label')  ## 라벨링 데이터 
Label_files = sorted([os.path.join(Label_dir, f) for f in os.listdir(Label_dir) if f.endswith(('.txt'))])


Img_dir = os.path.join(data_root, 'FruitImage_Real')  ## 추적할 과일 이미지 
Img_files = sorted([os.path.join(Img_dir, f) for f in os.listdir(Img_dir) if f.endswith(('.jpg', '.png'))])


## 저장 디렉토리 확인/생성
save_dir = os.path.join(data_root, 'Save_0801_Tem_Why')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


## 다중 이미지 인식을 위한 카드 이미지 크기 다양화 및 매칭을 위한 저장저장
card_images = [cv2.imread(img_file) for img_file in Img_files]
CardName = [os.path.splitext(os.path.basename(img_file))[0] for img_file in Img_files]


target_sizes  = [70,80]

template_list = []
reRotation_templates=[]

yolo_dict={}


# 원래 비율을 유지하며 리사이즈 함수
def resize_with_aspect_ratio(image, target_size):
    newImage=cv2.resize(image, (target_size,target_size))
    return newImage



# 로테이션 
def rotate_image(image,angle):
    """이미지를 주어진 각도로 회전시키고 회전 후 크기를 자동으로 조정하여 잘리지 않도록 합니다."""
    # 이미지의 중심을 회전 중심으로 설정
    center = (image.shape[1] // 2, image.shape[0] // 2)

     # 회전 행렬 생성
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 후 이미지 크기 계산 (회전된 이미지의 bounding box 크기)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    new_width = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
    new_height = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)

    # 회전 행렬 수정하여 이미지 크기를 새로 계산한 크기로 변환
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]


    rotated_image = cv2.warpAffine(image, matrix, (new_width, new_height))

    resized_image = cv2.resize(rotated_image, (image.shape[1], image.shape[0]))  # 원래 크기로 리사이즈

    return resized_image



# 박스 그리기 
def draw_detection_on_image(
    image,
    detecting_box,
    gt_box=None,
    class_name=None,
    score=None,
    iou=None
):
    img = image.copy()
    H, W = img.shape[:2]

    # 이미지 크기 기반 스케일
    font_scale = max(W, H) / 800 * 0.3     # 기본 800픽셀일 때 0.7
    thickness = max(int(W/400), 1)         # 두께 최소 1, 최대 3정도 자동
    padding = max(int(H/100), 5)           # 텍스트 주변 여백

    x1, y1, x2, y2 = detecting_box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), thickness)

    # 라벨 텍스트
    label = []
    if class_name:
        label.append(f"{class_name}")
    if score is not None:
        label.append(f"S:{score:.2f}")
    if iou is not None:
        label.append(f"IoU:{iou:.2f}")
    label_text = " ".join(label)
    if label_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        # y좌표가 박스 위에 잘림 방지
        text_x = x1
        text_y = max(y1 - text_height - padding, padding)

        # 반투명 박스
        overlay = img.copy()
        cv2.rectangle(overlay, 
                      (text_x, text_y), 
                      (text_x + text_width, text_y + text_height + baseline), 
                      (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # 아웃라인 + 텍스트 (흰색)
        cv2.putText(img, label_text, (text_x, text_y + text_height), font, font_scale, (0,0,0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, label_text, (text_x, text_y + text_height), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # GT 박스 (빨강)
    if gt_box is not None:
        gx1, gy1, gx2, gy2 = gt_box
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0,0,255), thickness)

    return img


def TemplateCreate():
    # 템플릿 구성
    for idx, card in enumerate(card_images):
        resized_templates = [resize_with_aspect_ratio(card, size) for size in target_sizes]
        reRotation_templates.clear()

        for img in resized_templates:
            # cv2.imshow("Check", img)
            # cv2.waitKey(0)  # 키 입력을 기다림 (0: 무한 대기)
            # cv2.destroyWindow("Check")  # 창 닫기
            for angle in  range(0,5,1):
                reRotation_templates.append(rotate_image(img, angle))

            # for angle in  range(0,355,5):
            #     reRotation_templates.append(rotate_image(img, angle))


        for img in reRotation_templates:
            resized_templates.append(img)


        template_list.append(resized_templates)




def precision_Recall(best_score, iou_score, Threding):
    total_tp=0
    total_fp=0
    total_fn=0

    #print(best_score)
    if best_score > 0.99:
        if iou_score >= Threding:
                total_tp = 1
        else:  ## 박스가 이상함 
            total_fp = 1
    else:  ## 디텍팅 실패     >> 해당 이미지에 모든 과일이 포함되어 있는데, 디텍팅 안되었다는건 >> 있는데 못잡은거다 그죠?
        total_fn = 1

    return total_tp, total_fp, total_fn, Threding



## 템플릿 매칭
def TemplateMatching_mode(back_files, pathName):
    """탬플릿 매칭 평가 및 성능 지표 계산"""

    save_Image = os.path.join(save_dir, f'{pathName}')
    if not os.path.exists(save_Image):
        os.makedirs(save_Image)



    stats_data = {}  # 클래스별 통계 저장
    stats_Averagedata = {}  # 클래스별 통계 전체
    
    total_tp, total_fp, total_fn = 0, 0, 0  # 전체 TP, FP, FN 저장

    for NameClass in CardName:
        stats_data[NameClass] = {"TP": 0, "FP": 0, "FN": 0, "Accuracy" : 0,"Precision": 0, "Recall": 0, "F1-score": 0, "Score": 0}

    for index_Threding in np.arange(0.55, 0.75, 0.01):
        index_Threding=round(index_Threding, 2)
        stats_Averagedata[str(index_Threding)]={"total_tp":0, "total_fp":0, "total_fn":0, "Accuracy" : 0, "Precision":0, "Recall":0}



    for imagePath in back_files:
        print(imagePath)
        frame = cv2.imread(imagePath)

        DetectingFrame = frame

        Name = os.path.splitext(os.path.basename(imagePath))[0]

        for i, templates in enumerate(template_list):
            best_score = -1  
            best_top_left = None

            # 과일 I가 있나?
            for template in templates:
                th, tw = template.shape[:2]
                res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_top_left = max_loc

            detected_class = CardName[i]
            
            iou_score=0

            if best_score > 0.5:  ## 디텍팅
                
                for DictName in yolo_dict:
                    if DictName in Name:
                        for class_id, bbox in yolo_dict[DictName]:
                            if class_id == detected_class:
                                DataBox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                                break

                detectingBox = [best_top_left[0], best_top_left[1], best_top_left[0] + tw, best_top_left[1] + th]
                iou_score = compute_iou(detectingBox, DataBox)
                

                if iou_score >= 0.5:  ## 맞춤 
                        stats_data[detected_class]["TP"] += 1
                        stats_data[detected_class]["Score"] += best_score
                        total_tp += 1

                        #저장
                        #DetectingFrame = draw_detection_on_image(DetectingFrame, detectingBox, DataBox, class_name=detected_class, score=best_score, iou=iou_score)
                        
                else: ## 안맞음..
                    total_fp += 1
                    stats_data[detected_class]["FP"] += 1



                

            else:  ## 디텍팅 실패     >> 해당 이미지에 모든 과일이 포함되어 있는데, 디텍팅 안되었다는건 >> 있는데 못잡은거다 그죠?
                total_fn += 1
                stats_data[detected_class]["FN"] += 1




            for index_Threding in np.arange(0.55, 0.75, 0.01):

                index_Threding=round(index_Threding, 2)
                temp_tp, temp_fp, temp_fn, Threding_temp = precision_Recall(best_score, iou_score, index_Threding)
                
                stats_Averagedata[str(Threding_temp)]["total_tp"]+=temp_tp
                stats_Averagedata[str(Threding_temp)]["total_fp"]+=temp_fp
                stats_Averagedata[str(Threding_temp)]["total_fn"]+=temp_fn




        #저장
        # base, ext = os.path.splitext(os.path.basename(imagePath))
        # save_path = os.path.join(save_Image, f"{base}_detected.jpg")

        # print(save_path)
        # cv2.imwrite(save_path, DetectingFrame)

  

            
    for Threding_tempNum, values in stats_Averagedata.items():
        accuracy,precision, recall, _ = compute_precision_recall_f1(values["total_tp"], values["total_fp"], values["total_fp"])
        stats_Averagedata[str(Threding_tempNum)]["Precision"]=precision
        stats_Averagedata[str(Threding_tempNum)]["Recall"]=recall
        stats_Averagedata[str(Threding_tempNum)]["Accuracy"]=accuracy
        

    # Precision, Recall, F1-score, AP, mAP 계산
    for class_name, values in stats_data.items():
        tp, fp, fn = values["TP"], values["FP"], values["FN"]
        accuracy,precision, recall, f1_score = compute_precision_recall_f1(tp, fp, fn)
        stats_data[class_name]["Accuracy"] = accuracy
        stats_data[class_name]["Precision"] = precision
        stats_data[class_name]["Recall"] = recall
        stats_data[class_name]["F1-score"] = f1_score
        
        tempScore = stats_data[class_name]["Score"]/len(back_files)
        stats_data[class_name]["Score"]=tempScore



    



    # 데이터 저장
    save_class_values_to_txt(stats_data,stats_Averagedata, pathName)
    save_class_values_to_excel(stats_data,stats_Averagedata, pathName)



## yolo 데이터 받아오기 
def read_yolo_file(file_path):
    file_name = os.path.basename(file_path).split(".")[0].replace("Ori","")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, bbox = yolo_to_bbox(line.strip())
            yolo_dict[file_name].append((CardName[class_id], bbox))
    
## yolo 데이터 >> 좌표값
def yolo_to_bbox(yolo_data, img_width=640, img_height=360):
    class_id, x_center, y_center, width, height = map(float, yolo_data.split())
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return int(class_id), (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))

##사각형 겹침 정도 분석
def compute_iou(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def compute_precision_recall_f1(tp, fp, fn):
   
    Accuracy =tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return round(Accuracy,4), round(precision, 4), round(recall, 4), round(f1_score, 4)

def compute_map(average_precisions):
    """mAP(mean Average Precision) 계산"""
    return round(sum(average_precisions) / len(average_precisions), 4) if average_precisions else 0

def compute_average_precision(precision_list, recall_list):
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    
    if len(precision_list) == 0 or len(recall_list) == 0:
        return 0
    
    # Precision-Recall 곡선의 AUC 계산
    return round(np.trapz(precision_list, recall_list), 4)





def save_class_values_to_excel(stats_data, stats_Averagedata, save_path):
    """ 성능 지표를 엑셀로 저장 """
    df = pd.DataFrame(stats_data).T.reset_index()
    df.columns = ["Class", "TP", "FP", "FN", "Accuracy", "Precision", "Recall", "F1-score", "Score"]

    # 전체 성능 요약 추가
    df_A = pd.DataFrame(stats_Averagedata).T.reset_index()
    df_A.columns = ["Threding","total_tp", "total_fp", "total_fn", "Accuracy", "Precision", "Recall"]


    path = os.path.join(save_dir, f"{save_path}_Performance.xlsx")
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Class-wise Performance", index=False)

        df_A.to_excel(writer, sheet_name="Overall Performance", index=False)



def save_class_values_to_txt(stats_data, stats_Averagedata, save_path):
    """탐지 성능 지표를 텍스트 파일로 저장"""
    path = os.path.join(save_dir, f"{save_path}_Performance.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Class-wise Performance ===\n")
        f.write("Class\tTP\tFP\tFN\tAccuracy\tPrecision\tRecall\tF1-score\tScore\n")
        f.write("-" * 90 + "\n")

        for class_name, values in stats_data.items():
            f.write(f"{class_name}\t{values['TP']}\t{values['FP']}\t{values['FN']}\t{values['Accuracy']}\t"
                    f"{values['Precision']}\t{values['Recall']}\t{values['F1-score']}\t{values['Score']}\n")

        f.write("=== All ===\n")
        f.write("Th\tTP\tFP\tFN\tAccuracy\tPrecision\tRecall\n")
        f.write("-" * 90 + "\n")

        for class_name, values in stats_Averagedata.items():
            f.write(f"{class_name}\t{values['total_tp']}\t{values['total_fp']}\t{values['total_fn']}\t{values['Accuracy']}\t"
                    f"{values['Precision']}\t{values['Recall']}\n")
            






if __name__ == "__main__":
    

    ## yolo 데이터 선별 
    for file_path in Label_files:
        file_name = os.path.basename(file_path).split(".")[0].replace("Ori","")
        yolo_dict[file_name] = []

    for path in Label_files:
        read_yolo_file(path)



    # 템플릿 생성 
    TemplateCreate()
    ## 템플릿 매칭 결과 비교

    # TemplateMatching_mode(Ori_files,"Ori")
    # TemplateMatching_mode(Pro_files[:10],"Pro")
    

    # Model_Groups = {"autoencoder", "dncnn", "resnet50", "srcnn", "unet","bestmodel_Com5_Wgan"}
    # for Model_Name in Model_Groups:
    #     Gener_dir = os.path.join(data_root, fr'GeneratorOutputs\Fin_0730\{Model_Name}\30')  ## pro 이미지
        
    #     Gener_files = sorted([os.path.join(Gener_dir, f) for f in os.listdir(Gener_dir) if f.endswith(('.jpg', '.png'))])

    #     TemplateMatching_mode(Gener_files,f"{Model_Name}")


    Model_Groups = {"5", "30", "50"}
    for Model_Name in Model_Groups:
        Gener_dir = os.path.join(data_root, fr'GeneratorOutputs\Fin_0730\bestmodel_Com5_Wgan_All_v2\{Model_Name}')  ## pro 이미지
        
        Gener_files = sorted([os.path.join(Gener_dir, f) for f in os.listdir(Gener_dir) if f.endswith(('.jpg', '.png'))])

        TemplateMatching_mode(Gener_files,f"bestmodel_Com5_Wgan_All_v2_{Model_Name}")


    # TemplateMatching_mode(Gener_files_2,"Gen_2")
    # TemplateMatching_mode(Gener_files_3,"Gen_3")
    
    #TemplateMatching_mode(Gray_files,"Gray")



  

