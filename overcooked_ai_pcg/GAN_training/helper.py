import numpy as np
import os
import json
from overcooked_ai_py import read_layout_dict
import dcgan
import torch
from torch.autograd import Variable
from overcooked_ai_py import LAYOUTS_DIR

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

# print(read_in_training_data(LAYOUTS_DIR))


def gan_generate(batch_size, model_path):
    nz = 32
    x = np.random.randn(batch_size, nz, 1, 1)

    generator = dcgan.DCGAN_G(isize=32, nz=nz, nc=len(obj_types), ngf=64, ngpu=1)
    generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(x).view(batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :10, :15]
    im = levels.data.cpu().numpy()
    im = np.argmax( im, axis = 1)
    lvl_int = im[0]

    lvl_str = ""
    for lvl_row in lvl_int:
        for tile_int in lvl_row:
            lvl_str += obj_types[tile_int]
        lvl_str += "\n"
    print(lvl_str)

# gan_generate(1, "data/training/netG_epoch_199_999.pth")