import os

_current_dir = os.path.dirname(os.path.abspath(__file__))

# LSI paths
# data path
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


LSI_STEAK_TEST_DIR = os.path.join(LSI_DATA_DIR, "steak_test")
STEAK_TEST_DIR = os.path.join(LSI_STEAK_TEST_DIR, "result")
LSI_STEAK_STUDY_CONFIG_DIR = os.path.join(LSI_STEAK_TEST_DIR, "config")
LSI_STEAK_STUDY_AGENT_DIR = os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "agents")
LSI_STEAK_STUDY_RESULT_DIR = os.path.join(LSI_STEAK_TEST_DIR, "result")