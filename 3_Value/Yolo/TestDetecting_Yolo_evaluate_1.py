import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


## Yolo 데이터와 비교해서 IOU, 신뢰성 체크


# 이미지 파일 목록 (디렉토리 내 이미지 파일 경로 가져오기)
root = os.path.dirname(os.path.realpath(__file__))

data_root = r"C:\Users\ICLAB\Desktop\Beam_Gan_Re\ImageData"


Ori_dir = os.path.join(data_root, 'TestSet_New_100_Ori')  ## Ori 이미지
Ori_files = sorted([os.path.join(Ori_dir, f) for f in os.listdir(Ori_dir) if f.endswith(('.jpg', '.png'))])

Pro_dir = os.path.join(data_root, 'TestSet_New_3500_Pro')  ## pro 이미지
Pro_files = sorted([os.path.join(Pro_dir, f) for f in os.listdir(Pro_dir) if f.endswith(('.jpg', '.png'))])


Label_dir = os.path.join(data_root, 'YoloLabel_100')  ## 라벨링 데이터
Label_files = sorted([os.path.join(Label_dir, f) for f in os.listdir(Label_dir) if f.endswith(('.txt'))])


Img_dir = os.path.join(data_root, 'FruitImage_Real')  ## 추적할 과일 이미지
Img_files = sorted([os.path.join(Img_dir, f) for f in os.listdir(Img_dir) if f.endswith(('.jpg', '.png'))])

## 저장 디렉토리 확인/생성
save_dir = os.path.join(data_root, 'Save_0806_fin_Yolo')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)




## 다중 이미지 인식을 위한 카드 이미지 크기 다양화 및 매칭을 위한 저장저장
CardName = [os.path.splitext(os.path.basename(img_file))[0] for img_file in Img_files]
yolo_dict={}

#yolo 모델 불러오기
yoloPath    = os.path.join(root, "08066best.pt")
model = YOLO(yoloPath)  # YOLOv8 기본 모델 사용


def precision_Recall(best_score, iou_score, Threding):
    total_tp=0
    total_fp=0
    total_fn=0

    print(best_score)
    if best_score > 0.7:
        if iou_score >= Threding:
                total_tp = 1
        else:  ## 박스가 이상함
            total_fp = 1
    else:  ## 디텍팅 실패     >> 해당 이미지에 모든 과일이 포함되어 있는데, 디텍팅 안되었다는건 >> 있는데 못잡은거다 그죠?
        total_fn = 1

    return total_tp, total_fp, total_fn, Threding


def Yolo_mode(image_files, pathName):
    """YOLO 모델을 사용하여 이미지에서 객체를 감지하고 최고 신뢰도를 가진 객체만 저장"""

    stats_data = {}  # 클래스별 통계 저장
    stats_Averagedata = {}  # 클래스별 통계 전체

    save_dir_Img = os.path.join(save_dir, f'{pathName}')
    if not os.path.exists(save_dir_Img):
            os.makedirs(save_dir_Img)

    total_tp, total_fp, total_fn = 0, 0, 0  # 전체 TP, FP, FN 저장

    total_Count=len(image_files)

    for NameClass in CardName:
        stats_data[NameClass] = {"TP": 0, "FP": 0, "FN": 0, "Accuracy" : 0,"Precision": 0, "Recall": 0, "F1-score": 0, "Score": 0}

    for index_Threding in np.arange(0, 1, 0.01):
        index_Threding=round(index_Threding, 2)
        stats_Averagedata[str(index_Threding)]={"total_tp":0, "total_fp":0, "total_fn":0, "Accuracy" : 0, "Precision":0, "Recall":0}




    for image_path in image_files:
        results = model(image_path)
        frame = cv2.imread(image_path)
        Name = os.path.splitext(os.path.basename(image_path))[0]

        max_confidences = {}  # 각 클래스별 최고 신뢰도 바운딩 박스

        for result in results:
            for box in result.boxes:
                YoloInd = int(box.cls)
                confidence = float(box.conf)
                Yolobox = box.xyxy[0].cpu().numpy().astype(int)  # (x1, y1, x2, y2) 형식

                if YoloInd not in max_confidences or confidence > max_confidences[YoloInd][1]:
                    max_confidences[YoloInd] = (Yolobox.tolist(), confidence)



        DataBox         = []
        detectingBox    = []


        print()

        for key, value in max_confidences.items(): ## 저장된 Yolo결과값
            detected_class = CardName[key]

            if(value[1]>0.1):## yolo 신뢰성 50퍼 이상만

                for DictName in yolo_dict: ## 파일이름찾기
                    if DictName in Name:
                        for class_id, bbox in yolo_dict[DictName]:  ##  라벨링된 딕셔너리 정보찾기
                            if class_id == CardName[key]:
                                DataBox = [bbox[0],bbox[1],bbox[2],bbox[3]]
                                break

                detectingBox = value[0]
                iou_score=compute_iou(detectingBox,DataBox) ## 겹침 정도 파악


            # 인식 결과 저장
                if iou_score >= 0.3:
                    stats_data[detected_class]["TP"] += 1
                    stats_data[detected_class]["Score"] += value[1]
                    total_tp += 1
                else: ## 안맞음..
                    total_fp += 1
                    stats_data[detected_class]["FP"] += 1

                    
            # else:  ## 디텍팅 실패     >> 해당 이미지에 모든 과일이 포함되어 있는데, 디텍팅 안되었다는건 >> 있는데 못잡은거다 그죠?
            #     total_fn += 1
            #     stats_data[detected_class]["FN"] += 1



            # print(total_tp, total_fp, total_fn)
            # for index_Threding in np.arange(0, 1, 0.5):

            #     index_Threding=round(index_Threding, 2)
            #     temp_tp, temp_fp, temp_fn, Threding_temp = precision_Recall(value[1], iou_score, index_Threding)


            #     print("------------------------")
            #     print(str(Threding_temp))
            #     print(str(detected_class))
            #     print(stats_Averagedata[str(Threding_temp)])
            #     print(temp_tp, temp_fp, temp_fn)
            #     stats_Averagedata[str(Threding_temp)]["total_tp"]+=temp_tp
            #     stats_Averagedata[str(Threding_temp)]["total_fp"]+=temp_fp
            #     stats_Averagedata[str(Threding_temp)]["total_fn"]+=temp_fn
            #     print(stats_Averagedata[str(Threding_temp)])
            #     print("------------------------")






            cv2.rectangle(frame, [detectingBox[0],detectingBox[1]], [detectingBox[2],detectingBox[3]], (0, 255, 0), 2)  ##
            cv2.putText(frame, f"{CardName[key]} ({value[1]:.2f})",
                        (detectingBox[0], detectingBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)


            cv2.rectangle(frame, [DataBox[0],DataBox[1]], [DataBox[2],DataBox[3]], (0, 0,255 ), 2)
            cv2.putText(frame, f"{CardName[key]} ({value[1]:.2f})",
                        (detectingBox[0], detectingBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)



            




            
        capture_path = os.path.join(save_dir_Img, f'Detecting_{os.path.basename(image_path)}.jpg')

        cv2.imwrite(capture_path, frame)



    for NameClass in CardName:
        stats_data[NameClass]["FN"] = total_Count-stats_data[NameClass]["TP"] -stats_data[NameClass]["FP"] 


    for Threding_tempNum, values in stats_Averagedata.items():
        accuracy,precision, recall, f1_score = compute_precision_recall_f1(values["total_tp"], values["total_fp"], values["total_fn"])
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

        ##tempScore = stats_data[class_name]["Score"]/len(image_files)
        ##stats_data[class_name]["Score"]=tempScore




    # 데이터 저장
    save_class_values_to_txt(stats_data,stats_Averagedata, pathName)
    save_class_values_to_excel(stats_data,stats_Averagedata, pathName)



## yolo 데이터 받아오기
def read_yolo_file(file_path):##


    file_name = os.path.basename(file_path).split("-")[0].replace("Ori","")

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
        f.write("Class\tTP\tFP\tFN\tPrecision\tRecall\tF1-score\tScore\n")
        f.write("-" * 90 + "\n")

        for class_name, values in stats_data.items():
            f.write(f"{class_name}\t{values['TP']}\t{values['FP']}\t{values['FN']}\t"
                    f"{values['Precision']}\t{values['Recall']}\t{values['F1-score']}\t{values['Score']}\n")

        f.write("=== All ===\n")
        f.write("Th\tTP\tFP\tFN\tPrecision\tRecall\n")
        f.write("-" * 90 + "\n")

        for class_name, values in stats_Averagedata.items():
            f.write(f"{class_name}\t{values['total_tp']}\t{values['total_fp']}\t{values['total_fn']}\t"
                    f"{values['Precision']}\t{values['Recall']}\n")



if __name__ == "__main__":


    ## yolo 데이터 선별
    for file_path in Label_files:
        file_name = os.path.basename(file_path).split("-")[0].replace("Ori","")
        yolo_dict[file_name] = []

    for path in Label_files:
        read_yolo_file(path)


    ## yolo 데이터
    Yolo_mode(Ori_files,"Ori")


    Pro_dir = os.path.join(data_root, fr'GeneratorOutputs\Fin_0730\Projected_1000')  ## pro 이미지
    Pro_files = sorted([os.path.join(Pro_dir, f) for f in os.listdir(Pro_dir) if f.endswith(('.jpg', '.png'))])


    Yolo_mode(Pro_files,"Pro")


    # "autoencoder", "dncnn","srcnn",
    Model_Groups = { "resnet50",  "autoencoder", "dncnn","srcnn", "unet"}
    for Model_Name in Model_Groups:
        Gener_dir = os.path.join(data_root, fr'GeneratorOutputs\Fin_0730\{Model_Name}\30')  ## pro 이미지
        
        Gener_files = sorted([os.path.join(Gener_dir, f) for f in os.listdir(Gener_dir) if f.endswith(('.jpg', '.png'))])
        Yolo_mode(Gener_files,f"{Model_Name}")


    Model_Name="bestmodel_Com5_Wgan"
    Gener_dir = os.path.join(data_root, fr'GeneratorOutputs\Fin_0730\{Model_Name}\5')  ## pro 이미지
        
    Gener_files = sorted([os.path.join(Gener_dir, f) for f in os.listdir(Gener_dir) if f.endswith(('.jpg', '.png'))])
    Yolo_mode(Gener_files,f"{Model_Name}_5")


    # Yolo_mode(Gener_files_1,"Gen_1")
    # Yolo_mode(Gener_files_2,"Gen_2")
    #Yolo_mode(Gener_files_3,"Gen_3")


    #Yolo_mode(Gray_files,"Gray")



