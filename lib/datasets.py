import functools
import tensorflow as tf

from lib.utils import load_json


SHUFFLE_BUFFER_SIZE = 1000
NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
PREFETCH_BUFFER_SIZE = tf.data.experimental.AUTOTUNE


def build_paired_dataset(db_path, batch_size, training=True):

    def preprocess(noise, label, root, training):
        noise = tf.strings.join([root, noise], '/')
        label = tf.strings.join([root, label], '/')
        noise = read_image(noise)
        label = read_image(label)
        if training:
            noise, label = augument_image(noise, label)
        return noise, label


    db = load_json(db_path)
    ds = tf.data.Dataset.from_tensor_slices((db['noise'], db['label']))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
    ds = ds.map(functools.partial(preprocess, 
                                  root=db['root'], 
                                  training=training),
                num_parallel_calls=NUM_PARALLEL_CALLS)
    ds = ds.batch(batch_size)
    return ds


def build_unpaired_dataset(db_path, batch_size, training=True):

    def preprocess(fake_noise, fake_label, root, real_images, training):
        fake_noise = tf.strings.join([root, fake_noise], '/')
        fake_label = tf.strings.join([root, fake_label], '/')
        real_image = tf.py_function(np.random.choice(real_images), [real_images], 'string')
        real_image = tf.strings.join([root, real_image], '/')
        fake_noise = read_image(fake_noise)
        fake_label = read_image(fake_label)
        real_image = read_image(real_image)
        if training:
            fake_noise, fake_label = augument_image(fake_noise, fake_label)
        return fake_noise, real_image, fake_label

    db = load_json(db_path)
    ds = tf.data.Dataset.from_tensor_slices((db['fake_input'], db['fake_label']))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
    ds = ds.map(functools.partial(preprocess, 
                                  root=db['root'], 
                                  real_images=db['real_image'],
                                  training=training),
                num_parallel_calls=NUM_PARALLEL_CALLS)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(PREFETCH_BUFFER_SIZE)
    return ds


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype='float32')
    return image
        

def augument_image(x, y, size=(256, 256, 6)):
    z = tf.concat([x, y], axis=-1)
    z = tf.image.random_flip_left_right(z)
    z = tf.image.random_flip_up_down(z)
    x, y = tf.split(z, [3, 3], axis=-1)
    return x, y
