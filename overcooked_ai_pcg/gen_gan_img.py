import os
from overcooked_ai_pcg.helper import visualize_lvl
from overcooked_ai_py import read_layout_dict
from overcooked_ai_py import LAYOUTS_DIR



for layout_file in os.listdir(os.path.join(LAYOUTS_DIR, "train_gan_large")):
    if layout_file.endswith(".layout") and layout_file.startswith("gen"):
        layout_name = layout_file.split('.')[0]
        raw_layout = read_layout_dict("train_gan_large" + "/" + layout_name)
        raw_layout = raw_layout['grid']
        visualize_lvl(raw_layout + "\n", "gan_train_imgs", f"{layout_name}.png")