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

            np_lvl = np.zeros((len(raw_layout), len(raw_layout[0])))
            for x, row in enumerate(raw_layout):
                row = row.strip()
                for y, tile in enumerate(row):
                    np_lvl[x][y] = obj_types.index(tile)
            lvls.append(np_lvl.astype(np.uint8))

    return np.array(lvls)

print(read_in_training_data(os.path.join("data", "layouts")))
