import numpy as np
import os
import json
from overcooked_ai_py import read_layout_dict

obj_types = "12XSPOD "

def read_in_training_data(data_path):
    lvls = []
    for layout_file in os.listdir(data_path):
        if layout_file.endswith(".layout") and layout_file.startswith("gen"):
            layout_name = layout_file.split('.')[0]
            raw_layout = read_layout_dict(layout_name)
            raw_layout = raw_layout['grid'].split('\n')
            lvl = []
            for row in raw_layout:
                lvl_row = []
                for tile in row:
                    print(tile)
                    lvl_row.append(obj_types.index(tile))
                lvl.append(np.array(lvl_row))
            lvls.append(np.array(lvl))

    return np.array(lvls)

# print(read_in_training_data(os.path.join("data", "layouts")))