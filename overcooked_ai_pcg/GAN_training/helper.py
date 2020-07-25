import numpy as np
import os
import json
import dcgan
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from overcooked_ai_py import read_layout_dict
from overcooked_ai_py import LAYOUTS_DIR
from overcooked_ai_pcg import ERR_LOG_PIC

obj_types = "12XSPOD "


def vertical_flip(np_lvl):
    """
    Return the vertically flipped version of the input np level.
    """
    np_lvl_vflip = np.zeros(np_lvl.shape)
    height, width = np_lvl.shape
    for x in range(height):
        for y in range(width):
            np_lvl_vflip[x][y] = np_lvl[x][width-y-1]
    
    return np_lvl_vflip.astype(np.uint8)

def horizontal_flip(np_lvl):
    """
    Return the horizontally flipped version of the input np level.
    """
    np_lvl_hflip = np.zeros(np_lvl.shape)
    height = np_lvl.shape[0]
    for x in range(height):
        np_lvl_hflip[x] = np_lvl[height-x-1]
    return np_lvl_hflip.astype(np.uint8)

def read_in_training_data(data_path):
    """
    Read in .layouts file and return the data

    Args:
        data_path: path to the directory containing the training data

    returns: a 3D np array of size num_lvl x lvl_height x lvl_width 
             containing the encoded levels
    """
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
            
            # data agumentation: add flipped levels to data set
            np_lvl = np_lvl.astype(np.uint8)
            np_lvl_vflip = vertical_flip(np_lvl)
            np_lvl_hflip = horizontal_flip(np_lvl)
            np_lvl_vhflip = vertical_flip(np_lvl_hflip)
            lvls.append(np_lvl)
            lvls.append(np_lvl_vflip)
            lvls.append(np_lvl_hflip)
            lvls.append(np_lvl_vhflip)

    return np.array(lvls)

# print(read_in_training_data(LAYOUTS_DIR))


def gan_generate(batch_size, model_path):
    """
    Generate level string given the path to the train netG model

    Args:
        model_path: path to the trained netG model
    """
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
    return lvl_str

def plot_err(average_errG_log, average_errD_log, average_errD_fake_log, average_errD_real_log):
    """
    Given lists of recorded errors and plot them.
    """
    plt.plot(average_errG_log, 'r', label="err_G")
    plt.plot(average_errD_log, 'b', label="err_D")
    plt.plot(average_errD_fake_log, 'm', label="err_D_fake")
    plt.plot(average_errD_real_log, 'g', label="err_D_real")
    plt.legend()
    plt.savefig(ERR_LOG_PIC)
    plt.show()


# lvl_str = gan_generate(1, "data/training/netG_epoch_54999_999.pth")
# print(lvl_str)