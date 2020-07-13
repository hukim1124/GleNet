import tensorflow as tf


KERNEL_INITIALIZER = 'he_normal'
KERNEL_REGULARUZER = tf.keras.regularizers.l2(5e-5)


# -----------------------------------------------------------------------------
# Proposed models
# -----------------------------------------------------------------------------
def build_GEN(name='GEN'):
    
    im = tf.keras.Input((None, None, 3))

    t = tf.keras.layers.Lambda(tf.image.resize,
                               arguments={'size': [256, 256]}, 
                               name='resize')(im)

    t = tf.keras.layers.Conv2D(16, 5,
                               padding='same', use_bias=False,
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name+'stage1_conv')(t)
    t = tf.keras.layers.BatchNormalization(name='stage1_bn')(t)
    t = tf.keras.layers.Activation('swish', name='stage1_swish')(t)

    t = inverted_residual_block(t, 16, 24, 5, 2, 6, name='stage2')
    t = inverted_residual_block(t, 24, 40, 5, 2, 6, name='stage3')
    t = inverted_residual_block(t, 40, 80, 5, 2, 6, name='stage4')
    t = inverted_residual_block(t, 80, 112, 5, 2, 6, name='stage5')
    
    t = tf.keras.layers.Conv2D(768, 1, use_bias=False,
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name+'stage6_conv')(t)
    t = tf.keras.layers.BatchNormalization(name='stage6_bn')(t)
    t = tf.keras.layers.Activation('swish', name='stage6_swish')(t)
    t = tf.keras.layers.GlobalAveragePooling2D(name='stage6_pool')(t)

    t = tf.keras.layers.Dense(768, use_bias=False,
                              kernel_initializer=KERNEL_INITIALIZER,
                              kernel_regularizer=KERNEL_REGULARUZER,
                              name='stage7_dense')(t)
    t = tf.keras.layers.Activation('sigmoid', name='stage7_sigmoid')(t)
    t = tf.keras.layers.Reshape((256, 3), name='predict_reshape')(t)
    x = IntensitiyTransform(3, 256, name='it')([im, t])

    return tf.keras.Model(im, x, name=name)


def build_LEN(name='LEN'):

    im = tf.keras.Input((None, None, 3))
    shape = tf.keras.Input((2), dtype='int32')
    
    x1 = tf.keras.layers.Conv2D(16, 5,
                                padding='same', use_bias=False,
                                kernel_initializer=KERNEL_INITIALIZER,
                                kernel_regularizer=KERNEL_REGULARUZER,
                                name=name+'stage1_conv')(im)
    x1 = tf.keras.layers.BatchNormalization(name='stage1_bn')(x1)
    x1 = tf.keras.layers.Activation('swish', name='stage1_swish')(x1)

    x2 = inverted_residual_block(x1, 16, 24, 5, 2, 6, name='stage2')
    x3 = inverted_residual_block(x2, 24, 40, 5, 2, 6, name='stage3')
    x4 = inverted_residual_block(x3, 40, 80, 5, 2, 6, name='stage4')
    x4 = inverted_residual_block(x4, 80, 40, 5, 1, 6, name='stage5')

    x4 = tf.keras.layers.UpSampling2D(interpolation='bilinear', name='stage6_upsample')(x4)
    x3 = tf.keras.layers.Concatenate(name='stage6_concat')([x3, x4])

    x3 = inverted_residual_block(x3, 80, 24, 5, 1, 6, name='stage7')

    x3 = tf.keras.layers.UpSampling2D(interpolation='bilinear', name='stage8_upsample')(x3)
    x2 = tf.keras.layers.Concatenate(name='stage8_concat')([x2, x3])

    x2 = inverted_residual_block(x2, 48, 16, 5, 1, 6, name='stage9')

    x2 = tf.keras.layers.UpSampling2D(interpolation='bilinear', name='stage10_upsample')(x2)
    x1 = tf.keras.layers.Concatenate(name='stage10_concat')([x1, x2])

    x1 = tf.keras.layers.Conv2D(3, 5,
                                padding='same', use_bias=False,
                                kernel_initializer=KERNEL_INITIALIZER,
                                kernel_regularizer=KERNEL_REGULARUZER,
                                name=name+'stage11_conv')(x1)
    x = tf.keras.layers.Add(name='stage11_add')([im, x1])

    return tf.keras.Model(im, x, name=name)


# def build_discriminator(input_shape, name='discriminator'):
    
#     im = tf.keras.Input(input_shape)

#     x = tf.keras.layers.Lambda(tf.image.resize,
#                                arguments={'size': [224, 224]}, 
#                                name='resize')(im)

#     x = conv_block(x, 16, 'stage1')
#     x = conv_block(x, 32, 'stage2')
#     x = conv_block(x, 64, 'stage3')
#     x = conv_block(x, 128, 'stage4')
#     x = conv_block(x, 256, 'stage5')

#     x = tf.keras.layers.Conv2D(1024, 1,
#                                use_bias=False,
#                                kernel_initializer=KERNEL_INITIALIZER,
#                                kernel_regularizer=KERNEL_REGULARUZER,
#                                name=name+'top_conv')(x)
#     x = tf.keras.layers.BatchNormalization(name='top_bn')(x)
#     x = tf.keras.layers.Activation('swish', name='top_swish')(x)
#     x = tf.keras.layers.GlobalAveragePooling2D(name='top_pool')(x)
#     x = tf.keras.layers.Dropout(0.2, name='top_dropout')(x)
    
#     x = tf.keras.layers.Dense(1, 
#                               use_bias=False,
#                               kernel_initializer=KERNEL_INITIALIZER,
#                               kernel_regularizer=KERNEL_REGULARUZER,
#                               name='predict_dense')(x)
    
#     return tf.keras.Model(im, x, name=name)


# -----------------------------------------------------------------------------
# Custom layer for intensity transform
# -----------------------------------------------------------------------------
class IntensitiyTransform(tf.keras.layers.Layer):    

    def __init__(self, channels, intensities, **kwargs):
        super(IntensitiyTransform, self).__init__(**kwargs)
        self.channels = channels
        self.scale = intensities - 1

    def call(self, inputs):
        im, it = inputs
        x = tf.map_fn(self._intensity_transform, [im, it], dtype='float32')
        return x
    
    def _intensity_transform(self, inputs):
        im, it = inputs
        im = tf.cast(tf.math.round(self.scale * im), dtype='int32')
        im = tf.split(im, num_or_size_splits=self.channels, axis=-1)
        it = tf.split(it, num_or_size_splits=self.channels, axis=-1)
        x = tf.concat([tf.gather_nd(a, b) for a, b in zip(it, im)], axis=-1)
        return x

# -----------------------------------------------------------------------------
# VGG16 model for perceptual loss
# -----------------------------------------------------------------------------
def build_vgg16():
    vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, 
                                              weights='imagenet', 
                                              input_tensor=None, 
                                              input_shape=(None, None, 3), 
                                              pooling=None)
    image = vgg16.input
    x1 = vgg16.get_layer('block1_conv2').output
    x2 = vgg16.get_layer('block2_conv2').output
    x3 = vgg16.get_layer('block3_conv2').output 
    model = tf.keras.Model(image, [x1, x2, x3])
    return model


# -----------------------------------------------------------------------------
# Building Block
# -----------------------------------------------------------------------------
def inverted_residual_block(inputs, 
                            filters_in=32, 
                            filters_out=16,
                            kernel_size=3,
                            strides=1,
                            expand_ratio=1,
                            se_ratio=0.25,
                            name=''):    
    # Expansion Phase
    filters = filters_in * expand_ratio
    x = tf.keras.layers.Conv2D(filters, 1, use_bias=False, 
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name+'_expand_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name+'_expand_bn')(x)
    x = tf.keras.layers.Activation('swish', name=name+'_expand_swish')(x)

    # Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, 
                                        padding='same', use_bias=False,
                                        depthwise_initializer=KERNEL_INITIALIZER,
                                        depthwise_regularizer=KERNEL_REGULARUZER,
                                        name=name+'_dwconv')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_bn')(x)
    x = tf.keras.layers.Activation('swish', name=name+'_swish')(x)

    # Squeeze and Excitation Phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(name=name+'_se_squeeze')(x)
        se = tf.keras.layers.Reshape((1, 1, filters), name=name+'_se_reshape')(se)
        se = tf.keras.layers.Conv2D(filters_se, 1, use_bias=False, 
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    kernel_regularizer=KERNEL_REGULARUZER,
                                    name=name+'_se_reduce_conv')(se)
        se = tf.keras.layers.Activation('swish', name=name+'_se_expand_swish')(se)
        se = tf.keras.layers.Conv2D(filters, 1, use_bias=False, 
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    kernel_regularizer=KERNEL_REGULARUZER,
                                    name=name+'_se_expand_conv')(se)
        se = tf.keras.layers.Activation('sigmoid', name=name+'_se_expand_sigmoid')(se)
        x = tf.keras.layers.Multiply(name=name+'_se_excite')([x, se])
    
    # Output phase
    x = tf.keras.layers.Conv2D(filters_out, 1, use_bias=False, 
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name+'_project_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_project_bn')(x)
    
    return x
