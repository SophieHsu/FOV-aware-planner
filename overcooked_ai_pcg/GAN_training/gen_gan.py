import os
import numpy as np
import time
import torch
from torch.autograd import Variable
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_pcg import GAN_TRAINING_DIR
from overcooked_ai_pcg.GAN_training import dcgan
from overcooked_ai_pcg.mip_repair import repair_lvl
from overcooked_ai_pcg.GAN_training.helper import obj_types, lvl_number2str, setup_env_from_grid

def gan_generate(batch_size, model_path):
    """
    Generate level string from random latent vector given the path to the train netG model

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
    
    lvl_repaired = repair_lvl(lvl_int)
    lvl_str = lvl_number2str(lvl_repaired)
    return lvl_str

def main():
    config = {
        "start_order_list": None,
        "cook_time": 20,
        "num_items_for_soup": 3,
        "delivery_reward": 20,
        "rew_shaping_params": None
    }
    lvl_str = gan_generate(1, os.path.join(GAN_TRAINING_DIR, "netG_epoch_54999_999.pth"))
    grid = [layout_row.strip() for layout_row in lvl_str.split("\n")][:-1]

    agent1, agent2, env = setup_env_from_grid(grid, config)
    done = False
    while not done:
        env.render()
        joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        time.sleep(0.5)
        
if __name__ == "__main__":
    main()