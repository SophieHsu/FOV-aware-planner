import os

_current_dir = os.path.dirname(os.path.abspath(__file__))

# GAN paths
# data path
GAN_DIR = os.path.join(_current_dir, "GAN_training")
GAN_DATA_DIR = os.path.join(GAN_DIR, "data")
GAN_TRAINING_DIR = os.path.join(GAN_DATA_DIR, "training")
GAN_LOSS_DIR = os.path.join(GAN_DATA_DIR, "loss")

# plot pic path
ERR_LOG_PIC = os.path.join(GAN_LOSS_DIR, "err.png")

# G_param file path
G_PARAM_FILE = os.path.join(GAN_DATA_DIR, "G_param.json")


# LSI paths
# data path
LSI_DIR = os.path.join(_current_dir, "LSI")
LSI_DATA_DIR = os.path.join(LSI_DIR, "data")
LSI_LOG_DIR = os.path.join(LSI_DATA_DIR, "log")

# config LSI search
LSI_CONFIG_DIR = os.path.join(LSI_DATA_DIR, "config")
LSI_CONFIG_EXP_DIR = os.path.join(LSI_CONFIG_DIR, "experiment")
LSI_CONFIG_MAP_DIR = os.path.join(LSI_CONFIG_DIR, "elite_map")
LSI_CONFIG_ALGO_DIR = os.path.join(LSI_CONFIG_DIR, "algorithms")
