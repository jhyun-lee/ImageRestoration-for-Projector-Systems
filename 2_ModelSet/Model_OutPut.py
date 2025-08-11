import os
import cv2
import numpy as np
import tensorflow as tf
from FeatureDataCreate import features
from tensorflow.keras.utils import Sequence
import random
import pickle
import ModelSet as Model_GR_sh


# 무작위성 제거 (고정된 결과 보장)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



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
                
                img_distorted = (img_distorted) / 255.0
                img_original = (img_original) / 255.0
                rgb_array = np.array(rgb) / 255.0
                distorted_imgs.append(img_distorted)
                originals.append(img_original)
                colors.append(rgb_array)
            yield {
                "image_input": np.array(distorted_imgs, dtype=np.float32),
                "color_input": np.array(colors, dtype=np.float32),
            }, np.array(originals, dtype=np.float32)


def ComparisionColorData(ProList):
    Color = features.RGB_Data(ProList)
    return Color


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


def save_cache_infoalllist(info_list, length, data_base_dir):
    file_name = f"info_{length}_Testlist_cache.pkl"
    save_path = os.path.join(data_base_dir, file_name)

    # 폴더가 없으면 생성
    os.makedirs(data_base_dir, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)
    print(f"InfoAllList 저장 완료: {save_path}")


def load_cache_infoalllist(length, data_base_dir):
    file_name = f"info_{length}_Testlist_cache.pkl"
    cache_path = os.path.join(data_base_dir, file_name)
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            info_list = pickle.load(f)
        print(f"InfoAllList 캐시 불러오기 완료: {cache_path}")
        return info_list
    else:
        print(f"캐시 파일이 존재하지 않습니다: {cache_path}")
        return None

## 문자열 변화---------------------------
def ProToORi(path):
    temp = os.path.basename(path).replace("projected", "")
    temp = temp.split("-")[0]
    return temp


def generator_inference_all(generator_weight_dir,  # 디렉토리 경로 받음
                             test_list, 
                             save_dir, 
                             ModelName=None,
                             Model_Num=1):

    os.makedirs(save_dir, exist_ok=True)

    # 1~10 모델 번호 리스트 (필요시 바꿀 수 있음)
    model_nums = range(Model_Num,Model_Num+1)

    for model_num in model_nums:
        weight_path = os.path.join(generator_weight_dir, f"{ModelName}_epoch_{model_num}.h5")
        if not os.path.exists(weight_path):
            print(f"⚠️ 가중치 파일 없음: {weight_path}, 건너뜀")
            continue

        # 모델별 결과 저장 폴더 생성
        model_save_dir = os.path.join(save_dir, str(model_num))
        os.makedirs(model_save_dir, exist_ok=True)

        # Generator 생성 및 가중치 로드

        ModelName = ModelName.lower()

        if ModelName == "unet":
            autoencoder = Model_GR_sh.build_unet_shallow()
        elif ModelName == "resnet50":
            autoencoder = Model_GR_sh.build_resnet50_shallow()
        elif ModelName == "dncnn":
            autoencoder = Model_GR_sh.build_dncnn_shallow()
        elif ModelName == "autoencoder":
            autoencoder = Model_GR_sh.build_autoencoder_shallow()
        elif ModelName == "srcnn":
            autoencoder = Model_GR_sh.build_srcnn_shallow()
            
        else:
            raise ValueError(f"Unknown postfix: {ModelName.postfix}")
        autoencoder.load_weights(weight_path)
        print(f"✅ Generator weight loaded: {weight_path}")

        # 테스트 데이터 준비 (원래대로)
        pro_list = [item[0] for item in test_list]
        Color_List = ComparisionColorData(pro_list)
        test_info = [(item[0], item[1], color) for item, color in zip(test_list, Color_List)]
        test_dataset = ImageDataset(test_info, batch_size=1)

        # 이미지 생성 및 저장
        for idx, (input_dict, _) in enumerate(test_dataset):
            output = autoencoder([input_dict["image_input"], input_dict["color_input"]], training=False)
            output_img = (output[0].numpy() * 255.0).astype(np.uint8)

            img_name = os.path.basename(test_list[idx][0]).replace(".jpg", f"_gen_model{model_num}.jpg")
            save_path = os.path.join(model_save_dir, img_name)

            cv2.imwrite(save_path, output_img)
            print(f"[Model {model_num}] [{idx+1}/{len(test_list)}] 저장 완료: {save_path}")




# 데이터 경로 설정
root = os.path.dirname(os.path.realpath(__file__))

parent_dir = os.path.dirname(root)
data_base_dir = os.path.join(parent_dir, 'ImageData')

data_dir_ori = os.path.join(data_base_dir, 'TestSet_New_3500_Ori')
data_dir_pro = os.path.join(data_base_dir, 'TestSet_New_3500_Pro')

file_list_ori = [os.path.join(data_dir_ori, f) for f in os.listdir(data_dir_ori) if "Ori" in f]#[:100]
file_list_pro = [os.path.join(data_dir_pro, f) for f in os.listdir(data_dir_pro) if "projected" in f]#[:100]
file_list_pro = random.sample(file_list_pro, min(5000, len(file_list_pro)))



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




ModelName = "unet"

generator_inference_all(
    generator_weight_dir=f"2_ModelSet/test_results_0723_{ModelName}",
    test_list=InfoAllList,
    save_dir=os.path.join(data_base_dir, fr"GeneratorOutputs\NoGenerator\{ModelName}"),
    ModelName=ModelName,
    Model_Num=30
)




