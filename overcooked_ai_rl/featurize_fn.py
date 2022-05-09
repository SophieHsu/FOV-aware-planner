import numpy as np


def low_level_featurize(mdp, state):
    """
    Featurize overcooked state as a low dimensional vector with the position
    of the players and key items.

    Args:
        mdp (OvercookedGridworld): mdp of the environment.
        state (OvercookedState): state of the environment.
    """
    pass

    print(np.array(mdp.lossless_state_encoding(state, debug=True)[0]).shape)


def lossless_state_featurize(mdp, state):
    return mdp.lossless_state_encoding(state)