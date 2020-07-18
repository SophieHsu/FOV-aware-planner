import os
from overcooked_ai_py.utils import load_dict_from_file

_current_dir = os.path.dirname(os.path.abspath(__file__))
GAN_TRAIN_DIR = _current_dir + "/GAN_training"
DATA_DIR = GAN_TRAIN_DIR + "/data/"
LAYOUTS_DIR = DATA_DIR + "layouts/"

def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))