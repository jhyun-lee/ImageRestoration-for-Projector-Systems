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


Ori_dir = os.path.join(data_root, 'TestSet_New_3500_Ori')  ## Ori 이미지
Ori_files = sorted([os.path.join(Ori_dir, f) for f in os.listdir(Ori_dir) if f.endswith(('.jpg', '.png'))])

Pro_dir = os.path.join(data_root, 'TestSet_New_3500_Pro')  ## pro 이미지
Pro_files = sorted([os.path.join(Pro_dir, f) for f in os.listdir(Pro_dir) if f.endswith(('.jpg', '.png'))])


Label_dir = os.path.join(data_root, 'TestSet_New_1000_Label')  ## 라벨링 데이터
Label_files = sorted([os.path.join(Label_dir, f) for f in os.listdir(Label_dir) if f.endswith(('.txt'))])


Img_dir = os.path.join(data_root, 'FruitImage_Real')  ## 추적할 과일 이미지
Img_files = sorted([os.path.join(Img_dir, f) for f in os.listdir(Img_dir) if f.endswith(('.jpg', '.png'))])

## 저장 디렉토리 확인/생성
save_dir = os.path.join(data_root, 'Save_0802_Yolo')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)





## 다중 이미지 인식을 위한 카드 이미지 크기 다양화 및 매칭을 위한 저장저장
CardName = [os.path.splitext(os.path.basename(img_file))[0] for img_file in Img_files]
yolo_dict={}

#yolo 모델 불러오기 
yoloPath    = os.path.join(root, "0303best.pt")
model = YOLO(yoloPath)  # YOLOv8 기본 모델 사용





def Yolo_mode(image_files, pathName):
    """YOLO 모델을 사용하여 이미지에서 객체를 감지하고 최고 신뢰도를 가진 객체만 저장"""
    ClassValue_dic={}   ## 정확성 

    for NameClass in CardName:
        ClassValue_dic[NameClass] = 0


    ClassIou_dic={}     ## 겹침정도 

    for NameClass in CardName:
        ClassIou_dic[NameClass] = 0

    Count=0
    

    for image_path in image_files:
        Count+=1

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

        for key, value in max_confidences.items(): ## 저장된 Yolo결과값


            
            if(value[1]>0.5):## yolo 신뢰성 50퍼 이상만
                for DictName in yolo_dict: ## 파일이름찾기
                    if DictName in Name:
                        for class_id, bbox in yolo_dict[DictName]:  ##  라벨링된 딕셔너리 정보찾기
                            if class_id == CardName[key]:
                                DataBox = [bbox[0],bbox[1],bbox[2],bbox[3]]
                                break

                detectingBox = value[0]

                TempIou=compute_iou(detectingBox,DataBox) ## 겹침 정도 파악
                ClassIou_dic[CardName[key]]+=TempIou


            # 인식 결과 저장 
                if TempIou>=0.1:  ## iou 0.1이상만 걸러내기 
                    ClassValue_dic[CardName[key]]+= value[1] ## 정확성 
                    cv2.rectangle(frame, [detectingBox[0],detectingBox[1]], [detectingBox[2],detectingBox[3]], (0, 0, 255), 2)  ## 
                    #cv2.rectangle(frame, [DataBox[0],DataBox[1]], [DataBox[2],DataBox[3]], (0, 255,0 ), 2)
                    
                    cv2.putText(frame, f"{CardName[key]} ({value[1]:.2f})",
                                (detectingBox[0], detectingBox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)


        capture_path = os.path.join(save_dir, f'Detecting_{Name}.jpg')
        cv2.imwrite(capture_path, frame)


    for Key,value in ClassValue_dic.items():
        ClassValue_dic[Key]= round(value / Count, 2)

    for Key,value in ClassIou_dic.items():
        ClassIou_dic[Key]= round(value / Count, 2)

    save_class_values_to_excel(ClassIou_dic,ClassValue_dic,pathName)
    save_class_values_to_text(ClassIou_dic,ClassValue_dic,pathName)



## yolo 데이터 받아오기 
def read_yolo_file(file_path):## 
    file_name = os.path.splitext(os.path.basename(file_path))[0].replace("Ori","")
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


def save_class_values_to_excel(class_lou, class_value, save_path):
    df = pd.DataFrame({
        "Class": list(class_lou.keys()),
        "IoU Overlap": list(class_lou.values()),
        "Detection Accuracy": list(class_value.values())
    })

    path = os.path.join(save_dir,f"{save_path}Data.xlsx")
    print(path)

    df.to_excel(path , index=False)


def save_class_values_to_text(class_lou, class_value, save_path):
    """ 정확성과 겹침 정도 데이터를 텍스트 파일로 저장 """
    path = os.path.join(save_dir,f"{save_path}Data.txt")

    with open(path, "w") as f:
        f.write("Class\tIoU Overlap\tDetection Accuracy\n")
        for key in class_lou:
            f.write(f"{key}\t{class_lou[key]:.2f}\t{class_value[key]:.2f}\n")



if __name__ == "__main__":
    

    ## yolo 데이터 선별 
    for file_path in Label_files:
        file_name = os.path.basename(file_path).split(".")[0].replace("Ori","")
        yolo_dict[file_name] = []

    for path in Label_files:
        read_yolo_file(path)


    ## yolo 데이터 
    ## yolo 데이터
    Yolo_mode(Ori_files,"Ori")
    Yolo_mode(Pro_files[:100],"Pro")



    Model_Groups = {"autoencoder", "dncnn", "resnet50", "srcnn", "unet","bestmodel_Com5_Wgan"}
    for Model_Name in Model_Groups:
        Gener_dir = os.path.join(data_root, fr'GeneratorOutputs\Fin_0730\{Model_Name}\30')  ## pro 이미지
        
        Gener_files = sorted([os.path.join(Gener_dir, f) for f in os.listdir(Gener_dir) if f.endswith(('.jpg', '.png'))])
        Yolo_mode(Gener_files[:100],f"{Model_Name}")

  

