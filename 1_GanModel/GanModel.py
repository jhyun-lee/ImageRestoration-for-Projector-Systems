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

from datetime import datetime
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 사용

# 무작위성 제거
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

current_time = datetime.now()
formatted_time = current_time.strftime("%m%d")

gen_losses = []
disc_losses = []


## 제너레이터 블럭
def residual_block(inputs, filters):
    res = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    res = layers.Conv2D(filters, (3, 3), padding='same')(res)
    return layers.Add()([inputs, res])

def attention_block(x):
    attn = layers.LayerNormalization()(x)
    attn = layers.Conv2D(x.shape[-1], 1, activation='relu', padding='same')(attn)
    attn = layers.DepthwiseConv2D(3, padding='same')(attn)
    return layers.Add()([x, attn])

def build_generator_V4(input_image_shape=(360, 640, 3), input_color_shape=(3,)):
    img_input = layers.Input(shape=input_image_shape, name="image_input")
    color_input = layers.Input(shape=input_color_shape, name="color_input")

    # Encoder
    x1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(img_input)
    p1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(p1)
    p2 = layers.MaxPooling2D((2, 2))(x2)
    x3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(p2)
    p3 = layers.MaxPooling2D((2, 2))(x3)   # (45,80,256) if input (360,640,3)

    # Condition Injection
    cond = layers.Dense(256, activation='relu')(color_input)
    cond = layers.Reshape((1, 1, 256))(cond)
    cond = layers.Lambda(lambda c: tf.tile(c[0], [1, tf.shape(c[1])[1], tf.shape(c[1])[2], 1]))([cond, p3])
    x = layers.Concatenate()([p3, cond])

    x = attention_block(x)
    x = residual_block(x, x.shape[-1])
    x = residual_block(x, x.shape[-1])

    # Decoder + skip
    u1 = layers.UpSampling2D((2, 2))(x)
    u1 = layers.Concatenate()([u1, x3])
    u1 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(u1)

    u2 = layers.UpSampling2D((2, 2))(u1)
    u2 = layers.Concatenate()([u2, x2])
    u2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(u2)

    u3 = layers.UpSampling2D((2, 2))(u2)
    u3 = layers.Concatenate()([u3, x1])
    u3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(u3)

    output = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(u3)
    return Model([img_input, color_input], output, name="Generator")


from tensorflow_addons.layers import InstanceNormalization

# Wgan용용
def build_discriminator_Com_v3(input_image_shape=(360, 640, 3), use_similarity=True):
    original_input = layers.Input(shape=input_image_shape, name="original_input")
    generated_input = layers.Input(shape=input_image_shape, name="generated_input")

    # 원본 이미지 특징 추출
    x1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu")(original_input)
    x1 = InstanceNormalization()(x1)  
    x1 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", activation="relu")(x1)
    x1 = InstanceNormalization()(x1)  
    
    # 생성 이미지 특징 추출
    x2 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu")(generated_input)
    x2 = InstanceNormalization()(x2)  
    x2 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", activation="relu")(x2)
    x2 = InstanceNormalization()(x2)  

    # Similarity Branch
    if use_similarity:
        similarity = layers.Lambda(lambda x: 1.0 - tf.abs(x[0] - x[1]) / (tf.abs(x[0]) + tf.abs(x[1]) + 1e-8))([original_input, generated_input])
        similarity = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(similarity)
        similarity = InstanceNormalization()(similarity)  
        similarity = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(similarity)
        similarity = InstanceNormalization()(similarity)  
        combined = layers.Concatenate()([x1, x2, similarity])
    else:
        combined = layers.Concatenate()([x1, x2])

    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same", activation="relu")(combined)
    x = InstanceNormalization()(x)  
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    
    output = layers.Dense(1)(x)

    return Model([original_input, generated_input], output, name="Discriminator_v3")



# ---------------------- VGG Feature Extractor ----------------------
def build_vgg_feature_extractor(input_shape=(360, 640, 3)):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    outputs = vgg.get_layer("block3_conv3").output
    model = Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model

def build_vgg_feature_extractor_Mini(input_shape=(180, 320, 3)):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    outputs = vgg.get_layer("block3_conv3").output
    model = Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model

# ---------------------- Perceptual Loss ----------------------
def perceptual_loss(y_true, y_pred, feature_extractor):
    y_true = preprocess_input(y_true * 255.0)
    y_pred = preprocess_input(y_pred * 255.0)
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    return tf.reduce_mean(tf.abs(true_features - pred_features))


# ---------------------- Gradient Penalty ----------------------
def compute_gradient_penalty(discriminator,
                             cond_imgs,        # ← images (왜곡 입력)
                             real_imgs,        # ← targets
                             fake_imgs):       # ← generated_images
    alpha = tf.random.uniform([real_imgs.shape[0], 1, 1, 1], 0., 1.)
    interp = alpha * real_imgs + (1 - alpha) * fake_imgs  # 보간본
    with tf.GradientTape() as tape:
        tape.watch(interp)
        d_interp = discriminator([cond_imgs, interp], training=True)
    grads = tape.gradient(d_interp, interp)               # 데이터 축만 미분
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    return tf.reduce_mean((norm - 1.)**2)



# ---------------------- RaGAN Loss ----------------------
def relativistic_loss(bce,D_real, D_fake):
    real_loss = bce(tf.ones_like(D_real), D_real - tf.reduce_mean(D_fake))
    fake_loss = bce(tf.zeros_like(D_fake), D_fake - tf.reduce_mean(D_real))
    return real_loss + fake_loss



# Regularization Loss
def gradient_channel_prior_loss(y_true, y_pred):
    def get_gradient(x):
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, 1:, :, :] - x[:, :-1, :, :]
        return dx, dy

    gx_true, gy_true = get_gradient(y_true)
    gx_pred, gy_pred = get_gradient(y_pred)
    return tf.reduce_mean(tf.abs(gx_true - gx_pred)) + tf.reduce_mean(tf.abs(gy_true - gy_pred))

def total_variation_loss(x):
    return tf.reduce_mean(tf.image.total_variation(x))



# ---------------------- GAN Loss Wrapper ----------------------
def get_generator_loss(D_real, D_fake, loss_type='bce'):
    bce = losses.BinaryCrossentropy(from_logits=True)

    if loss_type == 'bce':
        return losses.BinaryCrossentropy()(tf.ones_like(D_fake), D_fake)
    
    elif loss_type == 'ragan':
        return bce(tf.ones_like(D_fake), D_fake - tf.reduce_mean(D_real))
    
    elif loss_type == 'wgan':
        return -tf.reduce_mean(D_fake)




def get_discriminator_loss(D_real, D_fake, loss_type='bce', gp=None):
    bce = losses.BinaryCrossentropy(from_logits=True)

    if loss_type == 'bce':
        return losses.BinaryCrossentropy()(tf.ones_like(D_real), D_real) + \
               losses.BinaryCrossentropy()(tf.zeros_like(D_fake), D_fake)
    elif loss_type == 'ragan':
        return relativistic_loss(bce,D_real, D_fake)
    
    elif loss_type == 'wgan':
        loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
        if gp is not None:
            loss += 5.0 * gp 
        return loss



# ---------------------- Training Step ----------------------
def train_step(step_idx,n_critic,

        generator, discriminator, inputs, targets, 
               generator_optimizer, discriminator_optimizer, 
               loss_type='bce', l1_weight=20, perceptual_weight=0.0,
               feature_extractor=None):

    images = inputs["image_input"]
    colors = inputs["color_input"]

    with tf.GradientTape() as disc_tape:
        generated_images = generator([images, colors], training=True)
        real_output = discriminator([images, targets], training=True)
        fake_output = discriminator([images, generated_images], training=True)

        gp = compute_gradient_penalty(
                discriminator,
                cond_imgs   = images,            # 왜곡 입력 (고정)
                real_imgs   = targets,           # 원본
                fake_imgs   = generated_images   # 복원본
        ) if loss_type == 'wgan' else None
        

        disc_loss = get_discriminator_loss(real_output, fake_output, loss_type=loss_type, gp=gp)

    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))


    gen_loss = tf.constant(0.0)

    if tf.equal(tf.math.mod(step_idx, n_critic), 0):
        with tf.GradientTape() as gen_tape:
            generated_images = generator([images, colors], training=True)
            real_output = discriminator([images, targets], training=False)
            fake_output = discriminator([images, generated_images], training=False)



            gen_loss_base = get_generator_loss(real_output, fake_output, loss_type=loss_type)
            l1_loss = losses.MeanAbsoluteError()(targets, generated_images)


            perceptual = perceptual_loss(targets, generated_images, feature_extractor) if perceptual_weight > 0 and feature_extractor else 0
            gen_loss = gen_loss_base + l1_weight * l1_loss + perceptual_weight * perceptual


        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        # ✅ 로그 출력
        tf.print("[Loss Report]", 
                    "gen_total =", gen_loss, 
                    "| gen_base =", gen_loss_base, 
                    "| l1 =", l1_weight * l1_loss, "/", l1_loss,
                    "| perc =", perceptual_weight * perceptual, "/", perceptual,
                    "| disc =", disc_loss)

    return gen_loss, disc_loss


# ---------------------- Training Loop ----------------------
def Gan_train(Name,generator, discriminator, train_dataset, Test_List, epochs, start_epoch,
              generator_optimizer, discriminator_optimizer,
              loss_type='bce', l1_weight=20, perceptual_weight=0.0, feature_extractor=None,
              save_loss_plot=None, save_loss_csv=None):

    root = os.path.dirname(os.path.realpath(__file__))
    save_name = f"test_results_{formatted_time}_{Name}"
    save_dir = os.path.join(root,save_name)
    os.makedirs(save_dir, exist_ok=True)

    gen_losses=[]
    disc_losses=[]
    
    n_critic = 3 if loss_type=='wgan' else 1

    for epoch in range(start_epoch, epochs):
        gen_Sum = []
        disc_Sum = []

        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (inputs, targets) in enumerate(train_dataset):
            gen_loss, disc_loss = train_step(step,n_critic,
                                            generator, discriminator, inputs, targets,
                                             generator_optimizer, discriminator_optimizer,
                                             loss_type=loss_type, l1_weight=l1_weight,
                                             perceptual_weight=perceptual_weight, feature_extractor=feature_extractor)

            if step % 50 == 0:
                print(f"Step {step}: Gen Loss = {gen_loss.numpy()}, Disc Loss = {disc_loss.numpy()}")
                gen_Sum.append(gen_loss.numpy())
                disc_Sum.append(disc_loss.numpy())

        epoch_gen_loss = sum(gen_Sum) / len(gen_Sum)
        epoch_disc_loss = sum(disc_Sum) / len(disc_Sum)

        gen_losses.append(epoch_gen_loss)
        disc_losses.append(epoch_disc_loss)

        if save_loss_plot:
            save_loss_plot(gen_losses, disc_losses, os.path.join(save_dir, "loss_plot.jpg"))
        if save_loss_csv:
            save_loss_csv(gen_losses, disc_losses, os.path.join(save_dir, "loss_log.csv"))

        generated_image = generator(Test_List, training=False)[0].numpy()

        result_image = (generated_image * 255).astype("uint8")
        result_file = os.path.join(save_dir, f"epoch_{epoch + 1}_result.jpg")
        cv2.imwrite(result_file, result_image)

        generator.save(os.path.join(save_dir, f"generator_epoch_{epoch + 1}.h5"))
        discriminator.save(os.path.join(save_dir, f"discriminator_epoch_{epoch + 1}.h5"))

           

