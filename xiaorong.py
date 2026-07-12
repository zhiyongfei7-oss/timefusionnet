import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
print(tf.sysconfig.get_build_info()["cuda_version"])  # CUDA版本
print(tf.sysconfig.get_build_info()["cudnn_version"])  # cuDNN版本