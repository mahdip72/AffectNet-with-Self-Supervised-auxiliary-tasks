import tensorflow as tf
import random
import json
import adabelief_tf
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow_addons as tfa
from lr_scheduler import WarmUpCosine
from collections import Counter
import tensorflow_probability as tfp
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


def get_class_weights(y, inverse=False):
    counter = Counter(y)
    if not inverse:
        majority = max(counter.values())
        return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}
    if inverse:
        minority = min(counter.values())
        return {cls: 1 / (float(count) / float(minority)) for cls, count in counter.items()}


def rand_crop(image, fmin, fmax):
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.framework import ops
    image = ops.convert_to_tensor(image, name='image')

    if fmin <= 0.0 or fmin > 1.0:
        raise ValueError('fmin must be within (0, 1]')
    if fmax <= 0.0 or fmax > 1.0:
        raise ValueError('fmin must be within (0, 1]')

    img_shape = array_ops.shape(image)
    depth = image.get_shape()[2]
    my_frac2 = tf.random.uniform([1], minval=fmin, maxval=fmax, dtype=tf.float32, seed=42, name="uniform_dist")
    fraction_offset = tf.cast(math_ops.div(1.0, math_ops.div(math_ops.sub(1.0, my_frac2[0]), 2.0)), tf.int32)
    bbox_h_start = math_ops.div(img_shape[0], fraction_offset)
    bbox_w_start = math_ops.div(img_shape[1], fraction_offset)
    bbox_h_size = img_shape[0] - bbox_h_start * 2
    bbox_w_size = img_shape[1] - bbox_w_start * 2

    bbox_begin = array_ops.pack([bbox_h_start, bbox_w_start, 0])
    bbox_size = array_ops.pack([bbox_h_size, bbox_w_size, -1])
    image = array_ops.slice(image, bbox_begin, bbox_size)

    # The first two dimensions are dynamic and unknown.
    image.set_shape([None, None, depth])
    return image


def assigning_weight(label):
    label_weights = {0: 1.8, 1: 1.0, 2: 5.28, 3: 9.54, 4: 21.08, 5: 35.34, 6: 5.4, 7: 35.85}
    return label_weights[label.numpy()]


def flip_random_crop(image):
    image = tf.image.random_flip_left_right(image)
    #     image = random_resize_crop(image, crop_size=CROP_TO)
    return image


@tf.function
def float_parameter(level, maxval):
    return tf.cast(level * maxval / 10.0, tf.float32)


@tf.function
def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)


@tf.function
def solarize(image, level=6):
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 255 - image)


def color_jitter(x, strength=0.4):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength)
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength)
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def bluring(image, size=1, sigma=1.5):
    # kernel_size = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
    image = tfa.image.gaussian_filter2d(image, filter_shape=[size * 2 + 1, size * 2 + 1],
                                        sigma=[sigma, sigma])
    return image


def resizing(image, min_size=64):
    size = tf.random.uniform(shape=[], minval=min_size, maxval=128, dtype=tf.int32)
    image = tf.image.resize(image, (size, size))
    image = tf.image.resize(image, (128, 128))
    return image


def no_augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.central_crop(img, 0.80)
    img = tf.image.resize(img, [112, 112])
    return img


def weak_augment(image):
    rot = tf.random.uniform(shape=[1], minval=-15, maxval=15, dtype=tf.float32)
    rad = tf.divide((tf.cast(rot, tf.float32)) * np.pi, 180)
    image = tfa.image.rotate(image, rad)
    image = tf.squeeze(image)
    image = rand_crop(image, fmin=0.69, fmax=0.99)
    image = tf.image.resize(image, [112, 112])

    image = tf.cast(image, tf.float32)
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.4)
    image = random_apply(color_drop, image, p=0.2)
    # image = random_apply(solarize, image, p=0.3)
    # image = random_apply(bluring, image, p=0.2)
    # image = random_apply(resizing, image, p=0.2)
    image = tf.image.resize(image, [112, 112])
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.reshape(image, [112, 112, 3])
    return image


def strong_augment(image):
    rot_label = tf.random.uniform(shape=[1], minval=-25, maxval=25, dtype=tf.int32)
    rad = tf.divide((tf.cast(rot_label, tf.float32)) * np.pi, 180)
    image = tfa.image.rotate(image, rad)
    image = tf.squeeze(image)

    size = tf.random.uniform(shape=[], minval=150, maxval=200, dtype=tf.int32)
    image = tf.image.random_crop(image, (size, size, 3))
    image = tf.image.resize(image, [112, 112])

    image = tf.cast(image, tf.float32)
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.9)
    image = random_apply(color_drop, image, p=0.3)
    # image = random_apply(solarize, image, p=0.3)
    image = random_apply(bluring, image, p=0.2)
    image = random_apply(resizing, image, p=0.2)

    image = tf.image.resize(image, [112, 112])
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = tfa.image.random_cutout(tf.expand_dims(image, 0), mask_size=(40, 40), constant_values=0)
    image = tf.squeeze(image)

    image = tf.reshape(image, [112, 112, 3])
    return image


def train_preprocessing(augment_level='no'):
    def compute(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
        img = tf.image.resize(img, [200, 200])

        if augment_level == 'no':
            img = no_augment(img)
        elif augment_level == 'weak':
            img = weak_augment(img)
        elif augment_level == 'strong':
            img = strong_augment(img)

        # img = tf.image.resize(img, [112, 112])

        # if augment_level == 'strong':
        #     img = tfa.image.random_cutout(tf.expand_dims(img, 0), mask_size=(60, 60),
        #                                   constant_values=0)
        #     img = tf.squeeze(img)

        # img = tf.image.resize(img, [150, 150])
        # img = rand_crop(img, fmin=0.75, fmax=1.0)
        # img = tf.image.resize(img, [112, 112])

        all_labels = {
            'emotion': tf.one_hot(label, depth=8),
        }

        all_sample_weights = {
            'emotion': tf.py_function(func=assigning_weight, inp=[label], Tout=[tf.float32]),
            # 'emotion': 1,
        }

        return img, all_labels, all_sample_weights

    return compute


def valid_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    # img = tf.image.resize(img, [350, 350])
    img = tf.image.resize(img, [150, 150])

    # img = tf.image.random_crop(img, (size, size, 3))
    original_img = tf.image.central_crop(img, 0.85)
    # img = rand_crop(img, fmin=0.7, fmax=1.0)
    # img = tf.image.central_crop(img, 0.7)

    original_img = tf.image.resize(original_img, [112, 112])
    # img = tf.image.resize(img, [112, 112])

    all_labels = {
        'emotion': tf.one_hot(label, depth=8),
    }

    return original_img, all_labels


class GCRAdaBelief(adabelief_tf.AdaBeliefOptimizer):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


def main():
    autotune = tf.data.experimental.AUTOTUNE
    tf.keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_float16')

    # configs
    batch = 128
    epoch = 100
    warmup = 5

    # augment_level = 'no'
    augment_level = 'weak'
    # augment_level = 'strong'

    configs = {
        'aument': f"{augment_level} augment",
        'epoch': epoch,
        'batch': batch,
        'mixed_precision': True,
        'train_data': 'all',
        'label_smothing': 0.0,
        'train_resolution': 112,
        'warmup': warmup
    }

    model_name = f'sl-{augment_level}_augment-batch_{batch}'
    # model_name = 'test'

    if not os.path.exists(f"./results/{model_name}"):
        os.makedirs(f"./results/{model_name}")

    with open(f"./results/{model_name}/configs.txt", 'w') as file:
        for k, v in configs.items():
            file.write(str(k) + ': ' + str(v) + '\n\n')

    # dataset dir includes label and Manually_Annotated_Images folders
    dataset_dir = os.path.abspath("S:/Datasets/FER/AffectNet")

    train_csv_dir = os.path.join(dataset_dir, 'label', 'training.csv')

    valid_csv_dir = os.path.join(dataset_dir, 'label', 'validation.csv')

    train_csv_data = pd.read_csv(train_csv_dir)
    train_csv_data = train_csv_data[~train_csv_data['subDirectory_filePath'].str.contains(".tif", case=False)]
    train_csv_data = train_csv_data[~train_csv_data['subDirectory_filePath'].str.contains(".bmp", case=False)]
    train_csv_data = train_csv_data[train_csv_data['expression'] <= 7]
    train_csv_data['subDirectory_filePath'] = os.path.join(dataset_dir,
                                                           'Manually_Annotated_Images/Manually_Annotated_Images/') + \
                                              train_csv_data['subDirectory_filePath'].astype(str)

    valid_csv_data = pd.read_csv(valid_csv_dir, names=['subDirectory_filePath', 'face_x', 'face_y', 'face_width',
                                                       'face_height', 'facial_landmarks', 'expression', 'valence',
                                                       'arousal'],
                                 low_memory=False)

    valid_csv_data = valid_csv_data[~valid_csv_data['subDirectory_filePath'].str.contains(".tif", case=False)]
    valid_csv_data['subDirectory_filePath'] = os.path.join(dataset_dir,
                                                           'Manually_Annotated_Images/Manually_Annotated_Images/') + \
                                              valid_csv_data['subDirectory_filePath'].astype(str)
    valid_csv_data = valid_csv_data[valid_csv_data['expression'] <= 7]

#     train_csv_data = train_csv_data.groupby("expression").sample(
#         n=np.min(train_csv_data['expression'].value_counts()),
        # n=500,
#         random_state=1)

    input_data = train_csv_data.iloc[:, 0]
    train_labels = train_csv_data.iloc[:, 6]

    # class weights calculation
    # class_weights = get_class_weights(train_csv_data.expression.values, inverse=False)
    # class_weights = dict(sorted(class_weights.items()))
    # print(class_weights)
    # sample_weights = np.array([class_weights[i] for i in train_labels])

    train_ds = tf.data.Dataset.from_tensor_slices((np.array(input_data), np.array(train_labels)))
    train_ds = train_ds.shuffle(int(len(train_csv_data))).map(train_preprocessing(augment_level=augment_level),
                                                              num_parallel_calls=autotune)
    train_ds = train_ds.batch(batch).prefetch(autotune)

    valid_labels = valid_csv_data.iloc[:, 6]
    input_data = valid_csv_data.iloc[:, 0]
    valid_ds = tf.data.Dataset.from_tensor_slices((np.array(input_data), np.array(valid_labels)))
    valid_ds = valid_ds.map(valid_preprocessing, num_parallel_calls=autotune).batch(batch).prefetch(autotune)

    # for testing:
    # for i, j, w in train_ds:
    # for i, j in valid_ds:
    #     print(i.numpy().shape)
    #     imgs = i.numpy()
    #     imgs = (imgs * 255).astype(np.uint8)
    #     for inx, b in enumerate(range(imgs.shape[0])):
    #         img = imgs[b, :, :, :]
    #         print('weights:', w['emotion'].numpy()[b])
    #         print('labels:', j['emotion'].numpy()[b])
    #         cv2.imshow('', cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_RGB2BGR))
    #         cv2.waitKey(0)

    tf.keras.backend.clear_session()
    backbone = ResNet50(include_top=False,
                        input_shape=(112, 112, 3),
                        weights=None,
                        )

    # backbone.load_weights('results/ssl_barlow_twins-high_augment-batch_128_2/checkpoint.h5',
    #                       by_name=True)
    # for l in backbone.layers:
    #     l.trainable = False

    # backbone.summary()
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    emotion_dropout = tf.keras.layers.Dropout(0.3)(x)

    emotion = tf.keras.layers.Dense(8, activation='softmax', name='emotion', dtype=tf.float32)(emotion_dropout)

    # supervised head
    heads = [
        emotion
    ]

    model = tf.keras.Model(inputs=backbone.input,
                           outputs=heads
                           )
    model.summary()

    STEPS_PER_EPOCH = len(train_ds)
    TOTAL_STEPS = STEPS_PER_EPOCH * epoch
    WARMUP_STEPS = int(warmup * STEPS_PER_EPOCH)

    lr_decayed_fn = WarmUpCosine(
        learning_rate_base=1e-3,
        total_steps=TOTAL_STEPS,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )

    # opt = adabelief_tf.AdaBeliefOptimizer(learning_rate=lr_decayed_fn,
    #                                       print_change_log=False)
    opt = GCRAdaBelief(learning_rate=lr_decayed_fn,
                       print_change_log=False)

    op = tfa.optimizers.Lookahead(opt)

    losses = {
        'emotion': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
    }

    metrics = {
        'emotion': [
            tf.keras.metrics.CategoricalAccuracy('acc'),
            tfa.metrics.F1Score(8, name='f1', average='macro'),
        ]}

    model.compile(loss=losses,
                  optimizer=op,
                  metrics=metrics,
                  )

    val_loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/val_checkpoint.h5",
                                          monitor='val_loss', verbose=0,
                                          save_best_only=True, save_weights_only=False,
                                          mode='min', save_freq='epoch')

    csv_callback = CSVLogger(f"./results/{model_name}/training_log.csv",
                             append=False)

    callbacks_list = [
        # loss_checkpoint,
        # val_loss_checkpoint,
        csv_callback,
        tf.keras.callbacks.TensorBoard(log_dir=f"./results/{model_name}/log",
                                       update_freq='epoch')
    ]

    model.fit(train_ds,
              validation_data=valid_ds,
              callbacks=callbacks_list,
              verbose=2,
              epochs=epoch)

    print(f'finish sl with {augment_level} augment')


if __name__ == '__main__':
    main()
