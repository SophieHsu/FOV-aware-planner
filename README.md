# Overcooked-AI-PCG

Overcooked-AI-PCG is a Procedural Content Generation project aiming to generate Overcooked-Ai game levels that would cause undesirable behaviors while Human and AI play cooperatively in the environment.

## Overcooked-AI
<p align="center">
  <!-- <img src="overcooked_ai_js/images/screenshot.png" width="350"> -->
  <img src="overcooked_ai_py/images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

### Introduction

Overcooked-AI is a benchmark environment for fully cooperative multi-agent performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/). 

The goal of the game is to deliver soups as fast as possible. Each soup requires taking 3 items and placing them in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents).

For more information about the environment, please check out the [original Overcooked-AI repo](https://github.com/HumanCompatibleAI/overcooked_ai).

### Install Overcooked-AI

It is useful to setup a conda environment with Python 3.7:

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

To complete the installation after cloning the repo, run the following commands:

```
cd overcooked_ai
python setup.py develop
```

### Verifying Installation

To verify your python installation, you can try running the following command from the inner `overcooked_ai_py` folder:

```
python run_tests.py
```

If you're thinking of using the planning code extensively, you should run (this can take 5-10 mins): `python run_tests_full_planning_suite.py`


### Overcooked-AI Code Structure Overview

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically
- `actions`: actions that agents can take
- `graphics`: render related functions

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`run_tests.py`: script to run all tests

### Python Visualizations
To test the visualization mechanism of Overcooked-AI, please run the following:

```bash
cd overcooked_ai_py
python test_render.py
```

A pygame window should pop up and two agents should start performing random actions in the environment.

## PCG GAN

### GAN Training

To train the GAN that generates Overcooked-AI levels, run the following:

```bash
cd overcooked_ai_pcg/GAN_training
python train_gan.py --cuda
```

### Making More Training Data

The size of the training levels is fixed to be 15(width) x 10(height). The available tile types are:

```
'1': Player 1
'2': Player 2
'X': Wall
'S': Serve Point
'P': Pot
'O': Onion Dispenser
'D': Dish Dispenser 
' ': Floor
```

Please make sure that the levels you make satisfy **ALL** of the following constraints:

1. The level must be **rigidly surrounded**. i.e. the first and last row, and the first and last column can be anything except `‘1’`, `‘2’`, and `‘ ’`.

2. There are **exactly 2 players** at different positions. But they cannot be at the first and last row, and the first and last column.

3. There is **at least one** `‘O’`.

4. There is **at least one** `‘D’`.

5. There is **at least one** `‘P’`.

6. There is **at least one** `‘S’`.

7. `‘O’`, `‘D’`, `‘P’`, `‘S’` can be **anywhere**.

8. Both of the players must be able to reach at least one of `‘O’`, `‘D’`, `‘P’`, and `‘S’`.

9. The size is exactly **15(width) x 10(height)**

Please grab a version of `overcooked_ai_py/data/layouts/base.layout` to make the levels and place it under `overcooked_ai_py/data/layouts`. **Be sure to add prefix `gen` to its file name to differentiate it from non-GAN-training layouts.**

### Overcooked-AI-PCG Code structure Overview

`overcooked_ai_pcg` contains:

`GAN_training`:
- `dcgan.py`: Deep Convolutional Generative Adversarial Network Code
- `train_gan.py`: GAN training script
- `helper.py`: helper functions