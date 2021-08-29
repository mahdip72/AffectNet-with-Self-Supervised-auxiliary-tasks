import tensorflow as tf
import random
import adabelief_tf
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow_addons as tfa
from collections import Counter
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

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


def puzzle_piecess_rotation(split_img, puzzle_size):
    if puzzle_size == 9:
        # rotation puzzle pieces
        rot_labels = tf.convert_to_tensor(
            [
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)]
        )
        rot_labels = tf.random.shuffle(rot_labels)
        rot_idx = tf.cast(rot_labels, tf.float32) * 90
        rot_idx = tf.divide(tf.multiply(rot_idx, 3.1415), 180)
        split_img = tfa.image.rotate(split_img, rot_idx)
        return rot_labels, split_img

    elif puzzle_size == 4:
        # rotation puzzle pieces
        rot_labels = tf.convert_to_tensor(
            [
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
                tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)]
        )
        rot_labels = tf.random.shuffle(rot_labels)
        rot_idx = tf.cast(rot_labels, tf.float32) * 90
        rot_idx = tf.divide(tf.multiply(rot_idx, 3.1415), 180)
        split_img = tfa.image.rotate(split_img, rot_idx)
        return rot_labels, split_img


def jigsaw_puzzle(img, puzzle_size):
    if puzzle_size == 9:
        img = tf.image.resize(img, [225, 225])
        split_img = split_image(img, [75, 75], puzzle_size=puzzle_size)
        idx = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # idx = tf.random.shuffle(idx)
        split_img = tf.gather(split_img, idx, axis=0)

        rot_labels, split_img = puzzle_piecess_rotation(split_img, puzzle_size=puzzle_size)
        rec_img = unsplit_image(split_img, tf.shape(img), puzzle_size=puzzle_size)
        rec_img = tf.image.resize(rec_img, [224, 224])
        return rec_img, idx, rot_labels

    elif puzzle_size == 4:
        img = tf.image.resize(img, [224, 224])
        split_img = split_image(img, [112, 112], puzzle_size=puzzle_size)
        idx = tf.constant([0, 1, 2, 3])
        # idx = tf.random.shuffle(idx)
        split_img = tf.gather(split_img, idx, axis=0)

        rot_labels, split_img = puzzle_piecess_rotation(split_img, puzzle_size=puzzle_size)
        rec_img = unsplit_image(split_img, tf.shape(img), puzzle_size=puzzle_size)
        rec_img = tf.image.resize(rec_img, [224, 224])
        return rec_img, idx, rot_labels

    else:
        return img


def assigning_weight(label):
    label_weights = {0: 1.8, 1: 1.0, 2: 5.28, 3: 9.54, 4: 21.08, 5: 35.34, 6: 5.4, 7: 35.85}
    return label_weights[label.numpy()]


def no_augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.central_crop(img, 0.69)
    img = tf.image.resize(img, [224, 224])
    return img


def weak_augment(img):
    img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
    img = tf.clip_by_value(img, 0.0, 1.0)

    rot = tf.random.uniform(shape=[1], minval=-15, maxval=15, dtype=tf.float32)
    rad = tf.divide((tf.cast(rot, tf.float32)) * np.pi, 180)
    img = tfa.image.rotate(img, rad)
    img = tf.squeeze(img)

    img = rand_crop(img, fmin=0.69, fmax=0.99)
    img = tf.image.resize(img, [224, 224])
    return img


def strong_augment(img):
    # img = tf.image.random_hue(img, 0.2)
    img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.clip_by_value(img, 0.0, 1.0)

    var = tf.random.uniform(shape=[], minval=0, maxval=0.05, dtype=tf.float32)
    noise = tf.random.normal(shape=[350, 350, 3], mean=0.0,
                             stddev=tf.random.uniform(shape=[], minval=0, maxval=var, dtype=tf.float32),
                             dtype=tf.float32)
    img = tf.add(img, noise)
    img = tf.clip_by_value(img, 0.0, 1.0)

    kernel_size = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
    img = tfa.image.gaussian_filter2d(img, filter_shape=[kernel_size * 2 + 1, kernel_size * 2 + 1],
                                      sigma=[1.5, 1.5])

    rot = tf.random.uniform(shape=[1], minval=-20, maxval=20, dtype=tf.float32)
    rad = tf.divide((tf.cast(rot, tf.float32)) * np.pi, 180)
    img = tfa.image.rotate(img, rad)
    img = tf.squeeze(img)

    img = rand_crop(img, fmin=0.69, fmax=0.99)
    img = tf.image.resize(img, [224, 224])
    return img


def train_preprocessing(augment_level='no'):
    def compute(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
        img = tf.image.resize(img, [350, 350])

        if augment_level == 'no':
            img = no_augment(img)
        elif augment_level == 'weak':
            img = weak_augment(img)
        elif augment_level == 'strong':
            img = strong_augment(img)

        img = tf.image.resize(img, [224, 224])

        img, _, rot_labels = jigsaw_puzzle(img, number_of_tiles)

        if augment_level == 'strong':
            img = tfa.image.random_cutout(tf.expand_dims(img, 0), mask_size=(60, 60),
                                          constant_values=0)
            img = tf.squeeze(img)

        all_labels = {
            'emotion': tf.one_hot(label, depth=8),
        }

        for i in range(number_of_tiles):
            all_labels[f'rotation_{i + 1}'] = tf.one_hot(rot_labels[i], depth=4)

        all_sample_weights = {
            'emotion': tf.py_function(func=assigning_weight, inp=[label], Tout=[tf.float32]),
        }

        for i in range(number_of_tiles):
            all_sample_weights[f'rotation_{i + 1}'] = 1

        return img, all_labels, all_sample_weights

    return compute


def valid_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    img = tf.image.resize(img, [350, 350])

    # img = tf.image.random_crop(img, (size, size, 3))
    # img = rand_crop(img, fmin=0.8, fmax=1.0)
    img = tf.image.central_crop(img, 0.8)

    img = tf.image.resize(img, [224, 224])

    all_labels = {
        'emotion': tf.one_hot(label, depth=8),
    }

    for i in range(number_of_tiles):
        all_labels[f'rotation_{i + 1}'] = tf.one_hot(0, depth=4)

    return img, all_labels


def main(augment_level):
    autotune = tf.data.experimental.AUTOTUNE
    tf.keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_float16')
    # np.random.seed(12)

    # dataset dir includes label and Manually_Annotated_Images folders
    dataset_dir = os.path.abspath("S:/Datasets/FER/AffectNet")

    # augment_level = 'no'
    # augment_level = 'weak'
    # augment_level = 'strong'

    model_name = f'sl-ssl_rotation_{int(np.sqrt(number_of_tiles))}×{int(np.sqrt(number_of_tiles))}-{augment_level}_augment'
    # model_name = 'test'

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
    valid_csv_data = valid_csv_data[valid_csv_data['expression'] <= 7]
    valid_csv_data['subDirectory_filePath'] = os.path.join(dataset_dir,
                                                           'Manually_Annotated_Images/Manually_Annotated_Images/') + \
                                              valid_csv_data['subDirectory_filePath'].astype(str)

    batch = 32

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
    #     print(w)
    #     imgs = i.numpy()
    #     imgs = (imgs * 255).astype(np.uint8)
    #     for inx, b in enumerate(range(imgs.shape[0])):
    #         img = imgs[b, :, :, :]
    #         print('weights:', w)
    #         print('labels:', j)
    #         cv2.imshow('', cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_RGB2BGR))
    #         cv2.waitKey(0)

    tf.keras.backend.clear_session()
    backbone = ResNet50(include_top=False,
                        input_shape=(224, 224, 3),
                        # weights='imagenet',
                        weights=None,
                        )

    # backbone.summary()
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    emotion_dropout = tf.keras.layers.Dropout(0.3)(x)

    emotion = tf.keras.layers.Dense(8, activation='softmax', name='emotion', dtype=tf.float32)(emotion_dropout)

    ssl_branch_dropout = tf.keras.layers.Dropout(0.2)(x)

    # self supervised heads
    for i in range(number_of_tiles):
        # rotation head
        globals()[f"rotation_head_{i + 1}"] = tf.keras.layers.Dense(4,
                                                                    activation='softmax',
                                                                    name=f'rotation_{i + 1}',
                                                                    dtype='float32')(ssl_branch_dropout)

    # supervised head
    heads = [
        emotion
    ]

    # self-supervised heads
    heads += [globals()[f"rotation_head_{i + 1}"] for i in range(number_of_tiles)]

    model = tf.keras.Model(inputs=backbone.input,
                           outputs=heads
                           )
    model.summary()

    # op = tf.keras.optimizers.Adam(learning_rate=0.001)

    op = adabelief_tf.AdaBeliefOptimizer(learning_rate=0.001,
                                         print_change_log=False)

    # op = tf.keras.optimizers.SGD(learning_rate=0.000001, decay=0.0001, momentum=0.8, nesterov=False)

    losses = {
        'emotion': tf.keras.losses.CategoricalCrossentropy(),
    }

    for i in range(number_of_tiles):
        losses[f'rotation_{i + 1}'] = 'categorical_crossentropy'

    metrics = {
        'emotion': [
            tf.keras.metrics.CategoricalAccuracy('acc'),
            tfa.metrics.F1Score(8, name='f1', average='macro'),
        ]}

    for i in range(number_of_tiles):
        metrics[f'rotation_{i + 1}'] = [tf.keras.metrics.CategoricalAccuracy(name='acc')]

    model.compile(loss=losses,
                  optimizer=op,
                  metrics=metrics,
                  # loss_weights=loss_weights,
                  )

    val_loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/val_checkpoint.h5",
                                          monitor='val_loss', verbose=0,
                                          save_best_only=True, save_weights_only=False,
                                          mode='min', save_freq='epoch')

    loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/checkpoint.h5",
                                      monitor='loss', verbose=0,
                                      save_best_only=True, save_weights_only=False,
                                      mode='min', save_freq='epoch')

    csv_callback = CSVLogger(f"./results/{model_name}/training_log.csv",
                             append=False)

    def lr_scheduler(epoch, lr):
        if epoch == 15 or epoch == 40:
            lr = lr / 10
            return lr
        else:
            return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    callbacks_list = [
        loss_checkpoint,
        val_loss_checkpoint,
        lr_callback,
        csv_callback,
        tf.keras.callbacks.TensorBoard(log_dir=f"./results/{model_name}/log",
                                       update_freq='epoch')
    ]

    model.fit(train_ds,
              # steps_per_epoch=30,
              validation_data=valid_ds,
              callbacks=callbacks_list,
              verbose=2,
              epochs=80)

    print(f'finish sl + ssl rotation with {augment_level} augment')


if __name__ == '__main__':
    main('no')
