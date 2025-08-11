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


def build_unet_shallow(input_image_shape=(360, 640, 3), input_color_shape=(3,)):
    img_input = layers.Input(shape=input_image_shape, name="image_input")
    color_input = layers.Input(shape=input_color_shape, name="color_input")

    # Encoder
    d1 = layers.Conv2D(64, 3, padding="same", activation="relu")(img_input)  # (360, 640, 64)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(d1)                            # (180, 320, 64)

    d2 = layers.Conv2D(128, 3, padding="same", activation="relu")(p1)        # (180, 320, 128)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(d2)                            # (90, 160, 128)

    d3 = layers.Conv2D(256, 3, padding="same", activation="relu")(p2)        # (90, 160, 256)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(d3)                            # (45, 80, 256)

    # Condition Injection
    cond = layers.Dense(256, activation='relu')(color_input)
    cond = layers.Reshape((1, 1, 256))(cond)
    cond = layers.Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], tf.shape(x[1])[2], 1]))([cond, p3])
    x = layers.Concatenate()([p3, cond])  # (45, 80, 512)

    # Bottleneck
    b = layers.Conv2D(256, 3, padding="same", activation="relu")(x)         # (45, 80, 256)

    # Decoder
    u1 = layers.UpSampling2D(size=(2, 2))(b)                                 # (90, 160, 256)
    u1 = layers.Conv2D(256, 3, padding="same", activation="relu")(u1)
    u1 = layers.Concatenate()([u1, d3])                                      # (90, 160, 512)

    u2 = layers.UpSampling2D(size=(2, 2))(u1)                                # (180, 320, 256)
    u2 = layers.Conv2D(128, 3, padding="same", activation="relu")(u2)
    u2 = layers.Concatenate()([u2, d2])                                      # (180, 320, 256)

    u3 = layers.UpSampling2D(size=(2, 2))(u2)                                # (360, 640, 128)
    u3 = layers.Conv2D(64, 3, padding="same", activation="relu")(u3)
    u3 = layers.Concatenate()([u3, d1])                                      # (360, 640, 128)

    output = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(u3)

    return Model([img_input, color_input], output, name="UNet_MaxPool_3layer")


def build_resnet50_shallow(input_image_shape=(360, 640, 3), input_color_shape=(3,)):
    img_input = Input(shape=input_image_shape, name="image_input")
    color_input = Input(shape=input_color_shape, name="color_input")

    # ResNet50 encoder, 일부만 사용
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=img_input)
    x = base_model.get_layer("conv2_block3_out").output    # (90, 160, 256)

    # Condition injection (Dense(256) → tile → concat)
    cond = layers.Dense(256, activation='relu')(color_input)
    cond = layers.Reshape((1, 1, 256))(cond)
    cond = layers.Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], tf.shape(x[1])[2], 1]))([cond, x])

    x = layers.Concatenate()([x, cond])   # (45, 80, 512)

    # Bottleneck (비선형 변환)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)  # (90, 160, 256)

    # Decoder (3단계 업샘플)
    x = layers.UpSampling2D()(x)                # (180, 320, 256)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)                # (360, 640, 128)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    # 마지막 업샘플이 필요 없는 구조 (이미 360x640 됨)

    output = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)

    return Model(inputs=[img_input, color_input], outputs=output, name="ResNet50_Shallow_ColorInject")


def build_dncnn_shallow(input_image_shape=(360, 640, 3), input_color_shape=(3,)):
    img_input = layers.Input(shape=input_image_shape, name="image_input")
    color_input = layers.Input(shape=input_color_shape, name="color_input")

    # 색상 조건 임베딩 후 이미지 크기로 타일링
    color_dense = layers.Dense(64, activation='relu')(color_input)
    color_reshape = layers.Reshape((1, 1, 64))(color_dense)
    color_tiled = layers.Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], tf.shape(x[1])[2], 1]))([color_reshape, img_input])

    x = layers.Concatenate()([img_input, color_tiled])

    # 첫 Conv layer
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # 중간 Block 5개 (Conv+BN+ReLU)
    for _ in range(5):
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # 마지막 Conv layer
    x = layers.Conv2D(3, 3, padding='same')(x)

    # DnCNN은 residual 방식: 입력 + 예측치 = 복원
    output = layers.Add()([img_input, x])
    output = layers.Activation('sigmoid')(output)

    return Model([img_input, color_input], output, name="DnCNN_7layer")


def build_autoencoder_shallow(input_image_shape=(360, 640, 3), input_color_shape=(3,)):
    img_input = layers.Input(shape=input_image_shape, name="image_input")
    color_input = layers.Input(shape=input_color_shape, name="color_input")

    # Encoder: Conv + MaxPooling 기반
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(img_input)       # (360, 640, 64)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)                                  # (180, 320, 64)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)              # (180, 320, 128)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)                                  # (90, 160, 128)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)              # (90, 160, 256)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)                                  # (45, 80, 256)

    # Condition injection (Dense(256) → tile → concat)
    cond = layers.Dense(256, activation='relu')(color_input)
    cond = layers.Reshape((1, 1, 256))(cond)
    cond = layers.Lambda(lambda z: tf.tile(z[0], [1, tf.shape(z[1])[1], tf.shape(z[1])[2], 1]))([cond, x])

    x = layers.Concatenate()([x, cond])                                           # (45, 80, 512)

    # Bottleneck
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)              # (45, 80, 256)

    # Decoder: Conv2DTranspose 기반 업샘플링
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)  # (90, 160, 128)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)   # (180, 320, 64)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)   # (360, 640, 64)

    output = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)        # (360, 640, 3)

    return Model([img_input, color_input], output, name="AutoEncoder_Matched_MaxPool")


def build_srcnn_shallow(input_image_shape=(360, 640, 3), input_color_shape=(3,)):
    img_input = layers.Input(shape=input_image_shape, name="image_input")
    color_input = layers.Input(shape=input_color_shape, name="color_input")

    color_dense = layers.Dense(32, activation='relu')(color_input)
    color_reshape = layers.Reshape((1, 1, 32))(color_dense)
    color_tiled = layers.Lambda(
        lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], tf.shape(x[1])[2], 1])
    )([color_reshape, img_input])

    x = layers.Concatenate()([img_input, color_tiled])

    # 첫 Conv (9x9)
    x = layers.Conv2D(64, 9, padding='same', activation='relu')(x)
    # 중간 Conv (1x1 → 3x3 → 1x1)
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
    # 출력 Conv (5x5)
    x = layers.Conv2D(3, 5, padding='same', activation='sigmoid')(x)

    return Model([img_input, color_input], x, name="SRCNN_Deep")












