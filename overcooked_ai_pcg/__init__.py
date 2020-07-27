import os

_current_dir = os.path.dirname(os.path.abspath(__file__))

# data path
GAN_DIR = os.path.join(_current_dir, "GAN_training")
GAN_DATA_DIR = os.path.join(GAN_DIR, "data")
GAN_TRAINING_DIR = os.path.join(GAN_DATA_DIR, "training")
GAN_LOSS_DIR = os.path.join(GAN_DATA_DIR, "loss")

# plot pic path
ERR_LOG_PIC = os.path.join(GAN_LOSS_DIR, "err.png")

# G_param file path
G_PARAM_FILE = os.path.join(GAN_DATA_DIR, "G_param.json")