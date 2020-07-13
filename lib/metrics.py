import tensorflow as tf

def compute_psnr(image1, image2):
    image1 = tf.clip_by_value(image1, 0.0, 1.0)
    image2 = tf.clip_by_value(image2, 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(image1, image2, max_val=1.0))


def compute_ssim(image1, image2):
    image1 = tf.clip_by_value(image1, 0.0, 1.0)
    image2 = tf.clip_by_value(image2, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(image1, image2, max_val=1.0))
