import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH = 80

N_CLASSES = 724
N_EXAMPLES = 577 * 80

def _decode(raw_bytes):
    img = tf.cast(tf.image.decode_png(raw_bytes, channels=1), dtype=tf.float32)
    img = (img / 255) * 2 - 1
    return img


def _parse_batch(record_batch, sample_rate, duration):
    n_samples = 40064  # sample_rate * duration

    # Create a description of the features
    feature_description = {
        'mel': tf.io.FixedLenFeature([], tf.string),
        'bands': tf.io.FixedLenFeature([1], tf.int64),
        'bins': tf.io.FixedLenFeature([1], tf.int64),
        'sr': tf.io.FixedLenFeature([1], tf.int64),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    mel = tf.map_fn(_decode, example['mel'], dtype=tf.float32,
                    back_prop=False, parallel_iterations=10)
    # tf.print(tf.reshape(example['mel'], [-1, 313, 128, 1]).shape)
    # mel = tf.reshape(example['mel'], [-1, 313, 128, 1])
    # mel = example['mel']
    return mel, example['label']


def get_dataset_from_tfrecords(tfrecords_dir='tfrecords', split='train',
                               batch_size=BATCH, sample_rate=16000, duration=10,
                               n_epochs=10):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, '{}*.tfrec'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type='ZLIB',
                                 num_parallel_reads=AUTOTUNE)

    # Prepare batches
    ds = ds.shuffle(N_EXAMPLES * batch_size)
    ds = ds.batch(batch_size, drop_remainder=True)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    if split == 'train':
        ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTOTUNE)


def main():
    train_ds = get_dataset_from_tfrecords()
    mel, lab = next(iter(train_ds))
    print(mel.shape)
    plt.imshow(mel.numpy()[0], origin='lower')
    plt.show()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(313, 128, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.AveragePooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.AveragePooling2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.AveragePooling2D((2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.AveragePooling2D((2, 2)),

        # Linear
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='linear'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='linear'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(724, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # model = tf.keras.models.load_model('model.h5')
    model.fit(train_ds, epochs=100,
              steps_per_epoch=N_EXAMPLES / BATCH,
              callbacks=[tensorboard_callback])


if __name__ == '__main__':
    main()
