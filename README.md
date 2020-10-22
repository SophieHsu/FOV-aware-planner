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

It is useful to setup a conda environment with Python 3.7 using [Anaconda](https://www.anaconda.com/products/individual):

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

To complete the installation after cloning the repo, run the following commands:

```
cd overcooked_ai
pip install -e .
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

## PCG for Overcooked-AI

### GAN Training

To train the GAN that generates Overcooked-AI levels, run the following:

```bash
cd overcooked_ai_pcg/GAN_training
python train_gan.py --cuda
```

### Mixed Integer Linear Programming Solver

The solver is defined in `overcooked_ai_pcg/milp_repair.py`.

It uses [cplex optimizer of IBM](https://www.ibm.com/analytics/cplex-optimizer). Please follow the step [here](https://www.ibm.com/products/ilog-cplex-optimization-studio) to install **IBM ILOG CPLEX Optimization Studio** and the python interface of it. Once you have downloaded the installation file, this [guide](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_installing.html) may be helpful.

### Generate level using trained GAN and MILP solver

To use trained GAN and the MILP solver to generate Overcooked-AI levels, run the following:

```bash
cd overcooked_ai_pcg/
python gen_lvl.py
```

The program will generate a level from random latent vector sampled from normal distribution and then use MILP solver defined in `overcooked_ai_pcg/milp_repair.py` to fix the level.

### Latent Space Illumination

The Overcooked experiments use [Ray](https://docs.ray.io/en/master/) to run in a
distributed fashion. To begin, make sure you have the Conda environment set up
and your dependencies installed.

Next, change into the `LSI` directory:

```bash
cd overcooked_ai_pcg/LSI
```

Now, start the main machine (head node) for Ray:

```bash
ray start --head
```

You can provide extra compute for the script by running Ray can run on multiple
machines. If you would like to do so, **make sure that the other machines also
have the same Conda environment and dependencies installed**. `ray start --head`
outputs a command you can run on the other computers to start them. It should
look something like:

```bash
ray start --address IP_ADDRESS --redis-password PASSWORD
```

Once you have started Ray, run (only on the main machine):

```bash
python run_search.py -c <exp_config_file_path>
```

`exp_config_file_path` is the filepath to the experiment config file. It is
default to be `overcooked_ai_pcg/LSI/data/config/experiment/MAPELITES_demo.tml`

`run_search.py` will output log messages to the command line. Furthermore, visit
the Ray dashboard at http://localhost:8265 to see the status of the Ray workers.

When `run_search.py` ends, it will likely output several `Task failed: IOError`
messages because several tasks have been killed. This is expected behavior, as
once we have enough evaluations, we terminate remaining evaluations to save
computation.

#### LSI config files

There are three kinds of config files, each configuring different components of the LSI experiments.


##### `experiment` config files
They are under `overcooked_ai_pcg/LSI/data/config/experiment`.

An experiment config file contains the following required fields:
```
visualize (bool): to visualize the evaluations or not
num_cores (int): number of processes that runs the evaluations
num_simulations (int): total number of evaluations/simulations to run
algorithm_config (string): file name of the algorithm config file
elite_map_config (string): file name of the elite map config file
```

The experiment config files are the entry points each LSI experiments.

##### `algorithm` config files
They are under `overcooked_ai_pcg/LSI/data/config/algorithms`.

An algorithm config file contains the following required fields:
```
name (string): name of the algorithm used for deciding which
               algorithm instance to intialize at run time.
```

It also contains hyper params of the algorithm to run. For example, for MAP-Elites, they are initial population and mutation power.

##### `elite_map` config files
They are under `overcooked_ai_pcg/LSI/data/config/elite_map`.

An elite map config file contains an array of behavior characteristics (bc). Each bc contains the following required fields:
```
name (string): name of the bc
low (int/double): lower bound of the bc
high (int/double): upper bound of the bc
resolution (int): resolution (how many sections to divide) for the bc
```

Note that the name should match the name of the function to calculate the bc in `overcooked_ai_pcg/LSI/bc_calculate.py`

### Making More GAN Training Data

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

Note: These are also the constraints that the MILP solver is trying to satisfy.

### Overcooked-AI-PCG Code structure Overview

`overcooked_ai_pcg/` contains:

- `milp_repair.py`: Mixed Integer Linear Programming solver to fix levels generated by GAN.

- `gen_lvl.py`: Script that generates a level from trained GAN and repair that level using MILP solver.

- `helper.py`: helper functions

- `GAN_training/`:
  - `dcgan.py`: Deep Convolutional Generative Adversarial Network Code
  - `train_gan.py`: GAN training script

- `LSI/`:
  - `bc_calculate.py`: Relevant functions to calculate behavior characteristics
  - `qd_algorithms.py`: Implementations of QD algorithms
  - `run_search.py`: Script to run LSI search
  - `evaluator.py`: Overcooked game evaluator
  - `logger.py`: LSI experiment data loggers
  - `data/`: config and log data of LSI experiment
