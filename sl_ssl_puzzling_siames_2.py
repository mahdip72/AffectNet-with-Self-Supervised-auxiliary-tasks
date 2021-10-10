import tensorflow as tf
import random
import adabelief_tf
import numpy as np
import pandas as pd
import os
import cv2
from lr_scheduler import WarmUpCosine
import tensorflow_addons as tfa
from collections import Counter
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Optimizer

number_of_tiles = 9


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


def split_image(image3, tile_size, puzzle_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    split_img = tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])

    if puzzle_size == 4:
        return tf.gather(split_img,
                         [0, 2, 1, 3],
                         axis=0)
    if puzzle_size == 9:
        return tf.gather(split_img,
                         [0, 3, 6, 1, 4, 7, 2, 5, 8],
                         axis=0)


def unsplit_image(tiles4, image_shape, puzzle_size):
    if puzzle_size == 4:
        tiles4 = tf.gather(tiles4,
                           [0, 2, 1, 3],
                           axis=0)
    if puzzle_size == 9:
        tiles4 = tf.gather(tiles4,
                           [0, 3, 6, 1, 4, 7, 2, 5, 8],
                           axis=0)

    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])


def jigsaw_puzzle(img, puzzle_size):
    if puzzle_size == 9:
        # img = tf.image.resize(img, [225, 225])
        # split_img = split_image(img, [75, 75], puzzle_size=puzzle_size)
        img = tf.image.resize(img, [129, 129])
        split_img = split_image(img, [43, 43], puzzle_size=puzzle_size)
        idx = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8])
        idx = tf.random.shuffle(idx)
        split_img = tf.gather(split_img, idx, axis=0)

        rec_img = unsplit_image(split_img, tf.shape(img), puzzle_size=puzzle_size)
        # rec_img = tf.image.resize(rec_img, [224, 224])
        rec_img = tf.image.resize(rec_img, [128, 128])
        return rec_img, idx

    elif puzzle_size == 4:
        # img = tf.image.resize(img, [224, 224])
        # split_img = split_image(img, [112, 112], puzzle_size=puzzle_size)
        img = tf.image.resize(img, [128, 128])
        split_img = split_image(img, [64, 64], puzzle_size=puzzle_size)
        idx = tf.constant([0, 1, 2, 3])
        idx = tf.random.shuffle(idx)
        split_img = tf.gather(split_img, idx, axis=0)

        rec_img = unsplit_image(split_img, tf.shape(img), puzzle_size=puzzle_size)
        # rec_img = tf.image.resize(rec_img, [224, 224])
        # rec_img = tf.image.resize(rec_img, [112, 112])
        return rec_img, idx

    else:
        return img


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


def custom_augment(image):
    image = tf.cast(image, tf.float32)
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.9)
    image = random_apply(color_drop, image, p=0.3)
    # image = random_apply(solarize, image, p=0.3)
    image = random_apply(bluring, image, p=0.1)
    image = random_apply(resizing, image, p=0.2)
    return image


def do_augmenting(augmented_img):
    # augmented_img = tf.image.random_flip_left_right(augmented_img)
    # rot_label = tf.random.uniform(shape=[1], minval=-15, maxval=15, dtype=tf.int32)
    # rad = tf.divide((tf.cast(rot_label, tf.float32)) * np.pi, 180)
    # rot_label = tf.random.uniform(shape=[1], minval=0, maxval=3, dtype=tf.int32)
    # rad = tf.divide((tf.cast(rot_label * 90, tf.float32)) * np.pi, 180) + rad
    # augmented_img = tfa.image.rotate(augmented_img, rad)
    # augmented_img = tf.squeeze(augmented_img)

    # augmented_img = rand_crop(augmented_img, fmin=0.75, fmax=0.99)
    size = tf.random.uniform(shape=[], minval=150, maxval=200, dtype=tf.int32)
    augmented_img = tf.image.random_crop(augmented_img, (size, size, 3))

    augmented_img = custom_augment(augmented_img)
    augmented_img = tf.image.resize(augmented_img, [128, 128])

    # augmented_img = tf.image.random_hue(augmented_img, 0.08)
    # augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)

    # augmented_img = tf.image.random_contrast(augmented_img, lower=0.6, upper=1.4)
    # augmented_img = tf.image.random_brightness(augmented_img, max_delta=0.05)
    # augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)

    # kernel_size = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
    # augmented_img = tfa.image.gaussian_filter2d(augmented_img, filter_shape=[kernel_size * 2 + 1, kernel_size * 2 + 1],
    #                                             sigma=[1.5, 1.5])

    # var = tf.random.uniform(shape=[], minval=0, maxval=0.03, dtype=tf.float32)
    # noise = tf.random.normal(shape=[112, 112, 3], mean=0.0,
    #                          stddev=tf.random.uniform(shape=[], minval=0, maxval=var, dtype=tf.float32),
    #                          dtype=tf.float32)
    # augmented_img = tf.add(augmented_img, noise)
    # augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)
    # img, shuffle_label = jigsaw_puzzle(img, number_of_tiles)

    # augmented_img = tfa.image.random_cutout(tf.expand_dims(augmented_img, 0), mask_size=(44, 44),
    #                                         constant_values=0)
    # augmented_img = tf.squeeze(augmented_img)
    return augmented_img


def train_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    # img = tf.image.resize(img, [350, 350])
    img = tf.image.resize_with_pad(img, 200, 200)
    # img = tf.image.resize_with_pad(img, 150, 150)
    # img = tf.image.resize(img, [150, 150])

    # size = tf.random.uniform(shape=[], minval=200, maxval=300, dtype=tf.int32)
    # original_img = tf.image.central_crop(img, 0.8)
    # original_img = tf.image.random_crop(img, (size, size, 3))

    original_img = rand_crop(img, fmin=0.75, fmax=1.0)
    original_img = tf.image.resize(original_img, [112, 112])

    # augmented_img = tf.image.resize(img, [112, 112])
    augmented_img = do_augmenting(img)
    # augmented_img = rand_crop(img, fmin=0.7, fmax=1.0)
    # augmented_img = tf.image.resize(augmented_img, [112, 112])

    # rand_height = tf.random.uniform(shape=[], minval=25, maxval=90, dtype=tf.int32)
    # rand_width = tf.random.uniform(shape=[], minval=35, maxval=77, dtype=tf.int32)
    # augmented_img = tfa.image.cutout(tf.expand_dims(augmented_img, 0),
    #                                  mask_size=(36, 36),
    #                                  offset=(rand_height, rand_width),
    #                                  constant_values=0)
    # augmented_img = tf.squeeze(augmented_img)

    puzzled_img, shuffle_label = jigsaw_puzzle(augmented_img, number_of_tiles)
    # puzzled_img = augmented_img
    # puzzled_img = tf.image.resize(puzzled_img, [112, 112])

    augmented_img = tf.image.resize(augmented_img, [112, 112])


    all_labels = {
        'emotion': tf.one_hot(label, depth=8),
    }

    shuffle_label = tf.one_hot(shuffle_label, depth=number_of_tiles)
    for i in range(number_of_tiles):
        all_labels[f'part_{i + 1}'] = shuffle_label[i]

    all_sample_weights = {
        'emotion': tf.py_function(func=assigning_weight, inp=[label], Tout=[tf.float32]),
        # 'emotion': 1,
    }

    for i in range(number_of_tiles):
        all_sample_weights[f'part_{i + 1}'] = 0.5

    return original_img, augmented_img, puzzled_img, all_labels, all_sample_weights


def resize_batch(original_img, augmented_img, puzzled_img, all_labels, all_sample_weights):
    rand_size_1 = tf.random.uniform(shape=[], minval=113, maxval=128, dtype=tf.int32)
    rand_size_2 = tf.random.uniform(shape=[], minval=64, maxval=111, dtype=tf.int32)
    puzzled_imgs = {
        '113_128': tf.image.resize(puzzled_img, [rand_size_1, rand_size_1]),
        '112': tf.image.resize(puzzled_img, [112, 112]),
        '64_111': tf.image.resize(puzzled_img, [rand_size_2, rand_size_2]),
    }
    return original_img, augmented_img, puzzled_imgs, all_labels, all_sample_weights


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


class SiameseModel(tf.keras.Model):
    def __init__(self, encoder, loss_fns, mixed_prec=True):
        super(SiameseModel, self).__init__()
        self.loss_fns = loss_fns
        self.mixed_prec = mixed_prec
        self.encoder = encoder
        self.all_loss_tracker = tf.keras.metrics.Mean(name="all_loss")
        self.puzzle_loss_tracker = tf.keras.metrics.Mean(name="puzzle_loss")
        self.emotion_loss_tracker = tf.keras.metrics.Mean(name="emotion_loss")
        self.emotion_acc_tracker = tf.keras.metrics.Mean(name="emotion_acc")
        self.emotion_f1_tracker = tfa.metrics.F1Score(num_classes=8, average='macro', name="emotion_f1")

    @property
    def metrics(self):
        return [self.all_loss_tracker,
                self.puzzle_loss_tracker,
                self.emotion_loss_tracker,
                self.emotion_acc_tracker,
                self.emotion_f1_tracker,
                ]

    def call(self, inputs):
        return self.encoder(inputs)

    @staticmethod
    def multiply_list(myList):
        # Multiply elements one by one
        result = 1
        for x in myList:
            result = result * x
        return result

    def train_step(self, data):
        # Unpack the data.
        img1, img2, puzzled_imgs, label, sample_weight = data
        # img1, img2, label, sample_weight = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            # emotion_1, _, _, _, _ = self(img1, training=True)
            emotion_2, _, _, _, _, _, _, _, _, _ = self(img2, training=True)
            emotion_3, p1_3, p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, p8_3, p9_3 = self(puzzled_imgs['113_128'],
                                                                                       training=True)
            emotion_4, p1_4, p2_4, p3_4, p4_4, p5_4, p6_4, p7_4, p8_4, p9_4 = self(puzzled_imgs['112'],
                                                                                       training=True)
            emotion_5, p1_5, p2_5, p3_5, p4_5, p5_5, p6_5, p7_5, p8_5, p9_5 = self(puzzled_imgs['64_111'],
                                                                                       training=True)

            # emotion_loss_1 = self.loss_fns['emotion'](label['emotion'], emotion_1, sample_weight['emotion'])
            emotion_loss_2 = self.loss_fns['emotion'](label['emotion'], emotion_2, sample_weight['emotion'])
            emotion_loss_3 = self.loss_fns['emotion'](label['emotion'], emotion_3, sample_weight['emotion'])
            emotion_loss_4 = self.loss_fns['emotion'](label['emotion'], emotion_4, sample_weight['emotion'])
            emotion_loss_5 = self.loss_fns['emotion'](label['emotion'], emotion_5, sample_weight['emotion'])

            emotions_losses_list = [
                # emotion_loss_1,
                emotion_loss_2,
                emotion_loss_3,
                emotion_loss_4,
                emotion_loss_5,
            ]

            # emotions_loss = sum(emotions_losses_list) / len(emotions_losses_list)
            emotions_loss = tf.pow(self.multiply_list(emotions_losses_list), 1 / len(emotions_losses_list))

            puzzles_heads = [
                [p1_3, p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, p8_3, p9_3],
                [p1_4, p2_4, p3_4, p4_4, p5_4, p6_4, p7_4, p8_4, p9_4],
                [p1_5, p2_5, p3_5, p4_5, p5_5, p6_5, p7_5, p8_5, p9_5],
            ]

            puzzle_loss = 0
            for p1, p2, p3, p4, p5, p6, p7, p8, p9 in puzzles_heads:
                puzzle_loss += self.loss_fns['part_1'](label['part_1'], p1)
                puzzle_loss += self.loss_fns['part_2'](label['part_2'], p2)
                puzzle_loss += self.loss_fns['part_3'](label['part_3'], p3)
                puzzle_loss += self.loss_fns['part_4'](label['part_4'], p4)
                puzzle_loss += self.loss_fns['part_5'](label['part_5'], p5)
                puzzle_loss += self.loss_fns['part_6'](label['part_6'], p6)
                puzzle_loss += self.loss_fns['part_7'](label['part_7'], p7)
                puzzle_loss += self.loss_fns['part_8'](label['part_8'], p8)
                puzzle_loss += self.loss_fns['part_9'](label['part_9'], p9)

            puzzle_loss *= 0.3

            # final loss
            loss = emotions_loss + puzzle_loss
#             loss = emotions_loss

            if mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Monitor loss and acc
        self.all_loss_tracker.update_state(loss)
        self.puzzle_loss_tracker.update_state(puzzle_loss)
        self.emotion_loss_tracker.update_state(emotions_loss)
        self.emotion_acc_tracker.update_state(tf.keras.metrics.categorical_accuracy(label['emotion'], emotion_4))
        self.emotion_f1_tracker.update_state(label['emotion'], emotion_4)

        return {"all_loss": self.all_loss_tracker.result(),
                "puzzle_loss": self.puzzle_loss_tracker.result(),
                "emotion_loss": self.emotion_loss_tracker.result(),
                "emotion_acc": self.emotion_acc_tracker.result(),
                "emotion_f1": self.emotion_f1_tracker.result(),
                }

    def test_step(self, data):
        # Unpack the data.
        img1, label = data

        # Forward pass through the encoder and predictor.
        emotion_1, _, _, _, _, _, _, _, _, _ = self(img1, training=False)

        # emotions_loss = tf.keras.losses.categorical_crossentropy(emotion_1, label['emotion'])
        emotions_loss = self.loss_fns['emotion'](label['emotion'], emotion_1)

        # Monitor loss.
        self.emotion_loss_tracker.update_state(emotions_loss)
        self.emotion_acc_tracker.update_state(tf.keras.metrics.categorical_accuracy(label['emotion'], emotion_1))
        self.emotion_f1_tracker.update_state(label['emotion'], emotion_1)

        return {
            "emotion_loss": self.emotion_loss_tracker.result(),
            "emotion_acc": self.emotion_acc_tracker.result(),
            "emotion_f1": self.emotion_f1_tracker.result(),
        }


def main():
    autotune = tf.data.experimental.AUTOTUNE
    tf.keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_float16')
    batch = 128
    weight_decay = 0.1
    # np.random.seed(12)

    # dataset dir includes label and Manually_Annotated_Images folders
    dataset_dir = os.path.abspath("S:/Datasets/FER/AffectNet")

    model_name = f'sl_wo_siamese-batch_{batch}-1%-wd_{weight_decay}'
    # model_name = f'sl-ssl_puzzling_{int(np.sqrt(number_of_tiles))}×{int(np.sqrt(number_of_tiles))}-batch_{batch}-1%-wd_{weight_decay}'
    # model_name = f'sl-ssl_puzzling_{int(np.sqrt(number_of_tiles))}×{int(np.sqrt(number_of_tiles))}_barlow_twins-batch_{batch}_1'

    if not os.path.exists(f"./results/{model_name}"):
        os.makedirs(f"./results/{model_name}")

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
        # n=np.min(train_csv_data['expression'].value_counts()),
#         n=500,
#         random_state=1)

    input_data = train_csv_data.iloc[:, 0]
    train_labels = train_csv_data.iloc[:, 6]

    # class weights calculation
    # class_weights = get_class_weights(train_csv_data.expression.values, inverse=False)
    # class_weights = dict(sorted(class_weights.items()))
    # print(class_weights)
    # sample_weights = np.array([class_weights[i] for i in train_labels])

    train_ds = tf.data.Dataset.from_tensor_slices((np.array(input_data), np.array(train_labels)))
    train_ds = train_ds.shuffle(int(len(train_csv_data))).map(train_preprocessing, num_parallel_calls=autotune)
    train_ds = train_ds.batch(batch).map(resize_batch, num_parallel_calls=autotune).prefetch(autotune)

    valid_labels = valid_csv_data.iloc[:, 6]
    input_data = valid_csv_data.iloc[:, 0]
    valid_ds = tf.data.Dataset.from_tensor_slices((np.array(input_data), np.array(valid_labels)))
    valid_ds = valid_ds.map(valid_preprocessing, num_parallel_calls=autotune).batch(batch).prefetch(autotune)

    # for testing:
    # for i, j, p, l, w in train_ds:
    #     print(i.numpy().shape)
    #     original_imgs = i.numpy()
    #     augmented_imgs = j.numpy()
    #     print(p['113_128'].shape)
    #     puzzled_imgs = p['64_111'].numpy()
    #     original_imgs = (original_imgs * 255).astype(np.uint8)
    #     for inx, b in enumerate(range(original_imgs.shape[0])):
    #         original_img = original_imgs[b, :, :, :]
    #         augmented_img = augmented_imgs[b, :, :, :]
    #         puzzled_img = puzzled_imgs[b, :, :, :]
    #         cv2.imshow('original', cv2.cvtColor(cv2.resize(original_img, (224, 224)), cv2.COLOR_RGB2BGR))
    #         cv2.imshow('augmented', cv2.cvtColor(cv2.resize(augmented_img, (224, 224)), cv2.COLOR_RGB2BGR))
    #         cv2.imshow('puzzle augmented', cv2.cvtColor(cv2.resize(puzzled_img, (224, 224)), cv2.COLOR_RGB2BGR))
    #         cv2.waitKey(0)

    tf.keras.backend.clear_session()
    backbone = ResNet50(include_top=False,
                        # input_shape=(112, 112, 3),
                        weights=None,
                        )
    # backbone.summary()
    
    representation = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)

    # self supervised heads
    puzzle_dropout = tf.keras.layers.Dropout(0.3)(representation)
    for i in range(number_of_tiles):
        globals()[f"part_{i + 1}"] = tf.keras.layers.Dense(number_of_tiles,
                                                           activation='softmax',
                                                           name=f'part_{i + 1}',
                                                           dtype='float32')(puzzle_dropout)

    emotion_dropout = tf.keras.layers.Dropout(0.3)(representation)
    emotion = tf.keras.layers.Dense(8, activation='softmax', name="emotion",
                                    dtype=tf.float32)(emotion_dropout)

    heads = [
        emotion,
    ]

    # self-supervised heads
    heads += [globals()[f"part_{i + 1}"] for i in range(number_of_tiles)]

    model = tf.keras.Model(inputs=backbone.input,
                           outputs=heads
                           )

    model.summary()
    # op = tf.keras.optimizers.Adam(learning_rate=0.001)

    STEPS_PER_EPOCH = 125
    TOTAL_STEPS = STEPS_PER_EPOCH * 600
    WARMUP_EPOCHS = 10
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

    # lr_decayed_fn = WarmUpCosine(
    #     learning_rate_base=1e-3,
    #     total_steps=TOTAL_STEPS,
    #     warmup_learning_rate=0.0,
    #     warmup_steps=WARMUP_STEPS
    # )

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        0.001, TOTAL_STEPS)

    step = tf.Variable(0, trainable=False)
    wd = lambda: weight_decay * lr_decayed_fn(step)

    opt = adabelief_tf.AdaBeliefOptimizer(learning_rate=lr_decayed_fn,
#                                           weight_decay=wd,
                                          print_change_log=False)
    # opt = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn,
    #                               momentum=0.9, nesterov=True)
    op = tfa.optimizers.Lookahead(opt)

    losses = {
        'emotion': tf.keras.losses.CategoricalCrossentropy(),
    }

    for i in range(number_of_tiles):
        losses[f'part_{i + 1}'] = tf.keras.losses.CategoricalCrossentropy()

    model = SiameseModel(model, mixed_prec=True, loss_fns=losses)

    model.compile(
        optimizer=op,
    )
    # model.compute_output_shape(input_shape=(None, 112, 112, 3))
    # model.build(backbone.input)

    loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/checkpoint",
                                      monitor='val_emotion_loss', verbose=0,
                                      save_best_only=True, save_weights_only=True,
                                      mode='min', save_freq='epoch')

    csv_callback = CSVLogger(f"./results/{model_name}/training_log.csv",
                             append=False)


    callbacks_list = [
        # loss_checkpoint,
        # lr_callback,
        csv_callback,
        tf.keras.callbacks.TensorBoard(log_dir=f"./results/{model_name}/log",
                                       update_freq='epoch')
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        callbacks=callbacks_list,
        verbose=2,
        epochs=600)

    print('finish')


if __name__ == '__main__':
    main()
