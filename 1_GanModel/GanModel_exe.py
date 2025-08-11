import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, losses
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import threading
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from tensorflow.keras.utils import to_categorical
from FeatureDataCreate import features 

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import GanModel

from datetime import datetime
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 사용

# 무작위성 제거 (고정된 결과 보장)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

gen_losses = []
disc_losses = []


# 데이터 경로 설정
root = os.path.dirname(os.path.realpath(__file__))

parent_dir = os.path.dirname(root)
data_base_dir = os.path.join(parent_dir, 'ImageData')

data_dir_ori = os.path.join(data_base_dir, 'ModelLearning_0507_WarOri')
data_dir_pro = os.path.join(data_base_dir, 'ModelLearning_0507_WarPro')

data_dir_addori = os.path.join(data_base_dir, 'ModelLearning_0330_WarOri')
data_dir_addpro = os.path.join(data_base_dir, 'ModelLearning_0330_WarPro')



# 이미지데이터 모델에 전달 go
class ImageDataset(Sequence):
    def __init__(self, info_all_list, input_size=(360, 640), batch_size=8):
        self.info_all_list = info_all_list

        self.input_size = input_size
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(len(info_all_list) / batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            batch = self.info_all_list[i * self.batch_size:(i + 1) * self.batch_size]

            distorted_imgs, colors, originals = [], [], []

            for pro_path, ori_path, rgb in batch:
                img_distorted = cv2.imread(pro_path)
                img_original = cv2.imread(ori_path)

                if img_distorted is None or img_original is None:
                    continue 

                img_distorted = cv2.resize(img_distorted, self.input_size[::-1]) / 255.0
                img_original = cv2.resize(img_original, self.input_size[::-1]) / 255.0

                rgb_array = np.array(rgb) / 255.0

                distorted_imgs.append(img_distorted)
                originals.append(img_original)
                colors.append(rgb_array)
                
            yield {
                "image_input": np.array(distorted_imgs, dtype=np.float32),
                "color_input": np.array(colors, dtype=np.float32),
                
            }, np.array(originals, dtype=np.float32)


def save_loss_plot(gen_losses, disc_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss", color="blue")
    plt.plot(disc_losses, label="Discriminator Loss", color="red")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Generator & Discriminator Loss Over Time")
    plt.legend()
    plt.savefig(save_path)

def save_loss_csv(gen_losses, disc_losses, save_path="loss_log.csv"):
    df = pd.DataFrame({
        "Epoch": list(range(1, len(gen_losses) + 1)),
        "Generator_Loss": gen_losses,
        "Discriminator_Loss": disc_losses
    })
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Loss 기록 저장 완료: {save_path}")

def ComparisionColorData(ProList):
    Color = features.RGB_Data(ProList)
    return Color


## 문자열 변화---------------------------
def ProToORi(path):
    temp = os.path.basename(path).replace("projected", "")
    temp = temp.split("-")[0]
    return temp


def ProToImg(path):
    temp = os.path.basename(path).split(".")[0].replace("projected_", "")
    temp = "_".join(temp.split("_")[1:])
    return temp


## 캐시 저장---------------------------
def save_cache_infoalllist(info_list, length, data_base_dir):

    file_name = f"info_{length}_list_cache.pkl"
    save_path = os.path.join(data_base_dir, file_name)

    # 폴더가 없으면 생성
    os.makedirs(data_base_dir, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)
    print(f"InfoAllList 저장 완료: {save_path}")


def load_cache_infoalllist(length, data_base_dir):
    """지정된 경로에서 info_list 캐시 파일을 불러옵니다."""
    
    file_name = f"info_{length}_list_cache.pkl"
    cache_path = os.path.join(data_base_dir, file_name)
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            info_list = pickle.load(f)
        print(f"InfoAllList 캐시 불러오기 완료: {cache_path}")
        return info_list
    else:
        print(f"캐시 파일이 존재하지 않습니다: {cache_path}")
        return None
    


def TrainDataSet(Count_num):
        # 데이터 제너레이터 생성
    file_list_ori=[]
    file_list_pro=[]

    file_list_ori_1 = [os.path.join(data_dir_ori, f) for f in os.listdir(data_dir_ori) if "Ori" in f]#[:100]
    file_list_pro_1 = [os.path.join(data_dir_pro, f) for f in os.listdir(data_dir_pro) if "projected" in f]#[:100]

    file_list_ori_2 = [os.path.join(data_dir_addori, f) for f in os.listdir(data_dir_addori) if "Ori" in f]#[:100]
    file_list_pro_2 = [os.path.join(data_dir_addpro, f) for f in os.listdir(data_dir_addpro) if "projected" in f]#[:100]

    

    file_list_ori.extend(file_list_ori_1)
    file_list_ori.extend(file_list_ori_2)
    

    file_list_pro.extend(file_list_pro_1)
    file_list_pro.extend(file_list_pro_2)



    file_list_ori = sorted(file_list_ori)
    file_list_pro = sorted(file_list_pro)
    
    


    file_list_pro = random.sample(file_list_pro, min(Count_num, len(file_list_pro)))
    InfoAllList=load_cache_infoalllist(len(file_list_pro),data_base_dir)


    if InfoAllList is None:
        Color_List = ComparisionColorData(file_list_pro)

        ori_dict = {
            os.path.basename(path).replace("Ori", "").split(".")[0]: path
            for path in file_list_ori
        }
        Count=0
        InfoAllList =[]
        for pro_path, Color in zip(file_list_pro, Color_List):
            print(f"{len(file_list_pro)} / {Count}")
            pro_name = os.path.basename(pro_path)
            ori_id = ProToORi(pro_path)
            ori_path = ori_dict.get(ori_id)

            if ori_path:
                Count+=1
                InfoAllList.append([pro_path, ori_path, Color])  # 순서 주의: Pro, Ori, 등급, 색 
            else:
                print(f"❗ 매칭 실패: {pro_name}")

        save_cache_infoalllist(InfoAllList,Count,data_base_dir)
    



    train_dataset = tf.data.Dataset.from_generator(
    lambda: ImageDataset(InfoAllList, batch_size=batch_size),
    output_signature=(
            {
                "image_input": tf.TensorSpec(shape=(None, 360, 640, 3), dtype=tf.float32),
                "color_input": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(None, 360, 640, 3), dtype=tf.float32),
        )
    )

    return train_dataset


def TestSet(TestImage):

    test_List = [TestImage]

    Color_List = ComparisionColorData(test_List)

    test_info = [
    (
            test_List[0],
            test_List[0],                   # 있을 경우 넣고, 없으면 dummy 써도 됨
            Color_List[0]                   # RGB tuple
        )
    ]

    test_dataset = ImageDataset(test_info, batch_size=1)
    test_input, _ = next(iter(test_dataset))

    return test_input



if __name__ == '__main__':
    # 데이터 로드
    batch_size = 8
    loadEpoch =0
    Epoch = 150

    
    train_dataset = TrainDataSet(99999)

    TestImage = 'projected0301152038-_R15_G20_B240.jpg'
    test_input = TestSet(TestImage)



    # 모델 빌드 및 컴파일
    # Build models

    if loadEpoch!=0:
        print(f"generator_epoch_{loadEpoch}")

        Name=f"bestmodel_Com5_bce"
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d")
        root = os.path.dirname(os.path.realpath(__file__))
        save_name = f"test_results_{formatted_time}_{Name}"
        save_dir = os.path.join(root,save_name)


        generator = GanModel.build_generator_V4()
        generator.load_weights(os.path.join(save_dir, f"generator_epoch_{loadEpoch}.h5"))

        discriminator = GanModel.build_discriminator_Com_v3(use_similarity=True)
        discriminator = tf.keras.models.load_model(os.path.join(save_dir, f"discriminator_epoch_{loadEpoch}.h5"), compile=False)
        # 이전까지의 loss 기록 로드
        loss_csv_path = os.path.join(save_dir, "loss_log.csv")
        if os.path.exists(loss_csv_path):
            df = pd.read_csv(loss_csv_path)
            gen_losses = df["Generator_Loss"].tolist()[:loadEpoch]
            disc_losses = df["Discriminator_Loss"].tolist()[:loadEpoch]
        else:
            gen_losses = []
            disc_losses = []
    else:
        generator = GanModel.build_generator_V4()
        discriminator = GanModel.build_discriminator_Com_v3(use_similarity=True)



    GanModel.Gan_train(
        Name=f"bestmodel_Com5_Wgan_All_v2",
        generator = GanModel.build_generator_V4(),
        discriminator =  GanModel.build_discriminator_Com_v3(use_similarity=True),
        train_dataset=train_dataset,
        Test_List=test_input,  # shape: (1, H, W, 3), 또는 dict 형태

        epochs=Epoch,
        start_epoch=loadEpoch,

        generator_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001, 
            beta_1=0.0,  
            beta_2=0.9
        ),
        discriminator_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00005,  
            beta_1=0.0, 
            beta_2=0.9
        ),

        loss_type='wgan',                            # bce ragan wgan 
        l1_weight=20,                               # L1 loss weight
        perceptual_weight=0.1,                      # perceptual loss weight (0이면 사용 X)
        feature_extractor=GanModel.build_vgg_feature_extractor(),         # feature_extractor
        save_loss_plot=save_loss_plot,              # 함수 or None
        save_loss_csv=save_loss_csv                # 함수 or None
    )
            




 



