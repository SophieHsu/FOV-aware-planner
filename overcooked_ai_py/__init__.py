import os
from gym.envs.registration import register
from overcooked_ai_py.utils import load_dict_from_file

register(
    id='Overcooked-v0',
    entry_point='overcooked_ai_py.mdp.overcooked_env:Overcooked',
)

_current_dir = os.path.dirname(os.path.abspath(__file__))

# Original paths
DATA_DIR = _current_dir + "/data/"
ASSETS_DIR = _current_dir + "/assets/"
COMMON_TESTS_DIR = "/".join(_current_dir.split("/")[:-1]) + "/common_tests/"
HUMAN_DATA_DIR = DATA_DIR + "human_data/"
LAYOUTS_DIR = DATA_DIR + "layouts/"
IMAGE_DIR = os.path.join(_current_dir, "images")
PCG_EXP_IMAGE_DIR = os.path.join(IMAGE_DIR, "pcg_exp")
HUMAN_LVL_IMAGE_DIR = os.path.join(IMAGE_DIR, "human_levels")

# LSI paths
LSI_DIR = os.path.join(_current_dir, "LSI")
LSI_DATA_DIR = os.path.join(LSI_DIR, "data")
LSI_LOG_DIR = os.path.join(LSI_DATA_DIR, "log")
LSI_IMAGE_DIR = os.path.join(LSI_DATA_DIR, "images")

# config LSI search
LSI_CONFIG_DIR = os.path.join(LSI_DATA_DIR, "config")
LSI_CONFIG_EXP_DIR = os.path.join(LSI_CONFIG_DIR, "experiment")
LSI_CONFIG_MAP_DIR = os.path.join(LSI_CONFIG_DIR, "elite_map")
LSI_CONFIG_ALGO_DIR = os.path.join(LSI_CONFIG_DIR, "algorithms")
LSI_CONFIG_AGENT_DIR = os.path.join(LSI_CONFIG_DIR, "agents")

# human study
LSI_HUMAN_STUDY_DIR = os.path.join(LSI_DATA_DIR, "human_study")
LSI_HUMAN_STUDY_CONFIG_DIR = os.path.join(LSI_HUMAN_STUDY_DIR, "config")
LSI_HUMAN_STUDY_LOG_DIR = os.path.join(LSI_HUMAN_STUDY_DIR, "result")
LSI_HUMAN_STUDY_AGENT_DIR = os.path.join(LSI_HUMAN_STUDY_CONFIG_DIR, "agents")
LSI_HUMAN_STUDY_RESULT_DIR = os.path.join(LSI_HUMAN_STUDY_DIR, "result")

# steak test
LSI_STEAK_TEST_DIR = os.path.join(LSI_DATA_DIR, "steak_test")
STEAK_TEST_DIR = os.path.join(LSI_STEAK_TEST_DIR, "result")
LSI_STEAK_STUDY_CONFIG_DIR = os.path.join(LSI_STEAK_TEST_DIR, "config")
LSI_STEAK_STUDY_AGENT_DIR = os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "agents")
LSI_STEAK_STUDY_RESULT_DIR = os.path.join(LSI_STEAK_TEST_DIR, "result")

def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))