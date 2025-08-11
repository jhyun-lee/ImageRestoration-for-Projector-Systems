import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input, Model, layers
from tensorflow.keras import backend as K

import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from FeatureDataCreate import features
from datetime import datetime
import argparse

import ModelSet as Model_GR_sh


from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 랜덤 시드 고정
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# 경로 설정
root = os.path.dirname(os.path.realpath(__file__))

parent_dir = os.path.dirname(root)
data_base_dir = os.path.join(parent_dir, 'ImageData')

data_dir_ori = os.path.join(data_base_dir, 'ModelLearning_0507_WarOri')
data_dir_pro = os.path.join(data_base_dir, 'ModelLearning_0507_WarPro')

data_dir_addori = os.path.join(data_base_dir, 'ModelLearning_0330_WarOri')
data_dir_addpro = os.path.join(data_base_dir, 'ModelLearning_0330_WarPro')



current_time = datetime.now()
formatted_time = current_time.strftime("%m%d")




def resize_and_pad(img, initial_size=(640, 360), final_size=(640, 640)):
    resized = cv2.resize(img, initial_size, interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    pad_top = (final_size[0] - h) // 2
    pad_bottom = final_size[0] - h - pad_top
    pad_left = (final_size[1] - w) // 2
    pad_right = final_size[1] - w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def unpad_and_resize(img, original_size=(640, 360)):
    h, w = img.shape[:2]
    top = (h - original_size[1]) // 2
    bottom = top + original_size[1]
    return cv2.resize(img[top:bottom, :], (original_size[0], original_size[1]), interpolation=cv2.INTER_AREA)



def tf_unpad_center_crop(images, original_size=(640, 360)):
    h, w = tf.shape(images)[1], tf.shape(images)[2]
    top = (h - original_size[1]) // 2
    left = (w - original_size[0]) // 2
    return tf.image.crop_to_bounding_box(images, top, left, original_size[1], original_size[0])


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



def save_loss_plot(losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="MAE Loss", color="blue")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder Loss Over Time")
    plt.legend()
    plt.savefig(save_path)


def save_loss_csv_auto(losses, save_path):
    df = pd.DataFrame({"Epoch": list(range(1, len(losses) + 1)), "Loss": losses})
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Autoencoder Loss 저장 완료: {save_path}")


def ComparisionColorData(ProList):
    Color = features.RGB_Data(ProList)
    return Color


def ProToORi(path):
    temp = os.path.basename(path).replace("projected", "")
    temp = temp.split("-")[0]
    return temp

def save_cache_infoalllist(info_list, length, data_base_dir):
    
    file_name = f"info_{length}_list_cache.pkl"
    save_path = os.path.join(data_base_dir, file_name)
    os.makedirs(data_base_dir, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(info_list, f)
    print(f"InfoAllList 저장 완료: {save_path}")



def load_cache_infoalllist(length, data_base_dir):
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




def train_autoencoder(model,ModelName, train_dataset, test_input, save_dir,
                      epochs=30, start_epoch=0, feature_extractor=None, l1_weight=10,perceptual_weight=0.01):
    losses = []
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        print(f"[Epoch {epoch+1}/{epochs}]")
        epoch_losses = []

        for step, (inputs, targets) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                preds = model([inputs["image_input"], inputs["color_input"]], training=True)
                if preds.shape[1:3] != targets.shape[1:3]:
                    preds = tf.image.resize(preds, (targets.shape[1], targets.shape[2]))


                l1_loss = loss_fn(targets, preds)
                perceptual = perceptual_loss(targets, preds, feature_extractor) if feature_extractor else 0
                total_loss = l1_weight*l1_loss + perceptual_weight * perceptual

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(total_loss.numpy())

            if step % 50 == 0:
                print(f"Step {step}: Loss = {total_loss.numpy():.5f} (L1: {l1_loss:.5f}, Perceptual: {perceptual:.5f})")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

        # 테스트 결과 저장
        test_result = model([test_input["image_input"], test_input["color_input"]], training=False)[0].numpy()
        result_image = (np.clip(test_result, 0, 1) * 255).astype(np.uint8)
        result_file = os.path.join(save_dir, f"epoch_{epoch+1}_result.jpg")

        cv2.imwrite(result_file, result_image)

        model.save(os.path.join(save_dir, f"{ModelName}_epoch_{epoch+1}.h5"))

        save_loss_plot(losses, save_path=os.path.join(save_dir, f"loss_plot_{ModelName}.jpg"))
        save_loss_csv_auto(losses, save_path=os.path.join(save_dir, f"loss_log_{ModelName}.csv"))



# ---------------------- Perceptual Loss ----------------------
def build_vgg_feature_extractor(input_shape=(360, 640, 3)):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    outputs = vgg.get_layer("block3_conv3").output
    model = Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model



def perceptual_loss(y_true, y_pred, feature_extractor):
    y_true = preprocess_input(y_true * 255.0)
    y_pred = preprocess_input(y_pred * 255.0)
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    return tf.reduce_mean(tf.abs(true_features - pred_features))



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

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--postfix", type=str, default="Unet")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size

    file_list_ori = sorted([os.path.join(data_dir_ori, f) for f in os.listdir(data_dir_ori) if "Ori" in f] )
    file_list_pro = sorted([os.path.join(data_dir_pro, f) for f in os.listdir(data_dir_pro) if "projected" in f] )
    

    file_list_pro = random.sample(file_list_pro, min(99999, len(file_list_pro)))


    InfoAllList=load_cache_infoalllist(len(file_list_pro),data_base_dir)


    if InfoAllList is None:
        Color_List = ComparisionColorData(file_list_pro)
        ori_dict = {os.path.basename(p).replace("Ori", "").split(".")[0]: p for p in file_list_ori}
        InfoAllList = []
        Count = 0
        for pro_path, Color in zip(file_list_pro, Color_List):
            ori_id = ProToORi(pro_path)
            ori_path = ori_dict.get(ori_id)

            pro_name = os.path.basename(pro_path)
            if ori_path:
                InfoAllList.append([pro_path, ori_path, Color])
                Count += 1
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


    ModelName = args.postfix.lower()

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
        raise ValueError(f"Unknown postfix: {args.postfix}")
    


    save_name = f"test_results_{formatted_time}_{ModelName}"
    save_dir = os.path.join(root,save_name)
    os.makedirs(save_dir, exist_ok=True)

    autoencoder.summary()

    TestImage = 'projected0301152038-_R15_G20_B240.jpg'

    img_to_test = cv2.imread(TestImage)
    save_path = os.path.join(save_dir, "epoch_0_result.jpg")

    cv2.imwrite(save_path, img_to_test)


    test_input = TestSet(TestImage)

    train_autoencoder(
        model=autoencoder,
        ModelName=ModelName,
        train_dataset=train_dataset,
        test_input=test_input,
        save_dir=save_dir,
        epochs=epochs,
        start_epoch=0,
        l1_weight=20.0, 
        perceptual_weight=0.1,
        feature_extractor=build_vgg_feature_extractor()
    )

