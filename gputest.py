import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# tests whether tensorflow detects and will run on GPUs.