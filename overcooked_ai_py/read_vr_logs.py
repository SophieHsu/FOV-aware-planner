import json
import os

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

overcooked_states = []

if __name__ == "__main__":
    # f = open(os.path.join(os.getcwd(), "overcooked_ai_py/data/logs/vr_study_logs/mike_d/test.json"))
    f = open(os.path.join(os.getcwd(), "overcooked_ai_py/data/logs/vr_study_logs/12/mid_aware.json"))
    log = json.load(f)

    for i in range(1, log['i']):
        try:
            state_dict = log[str(i)]['overcooked_state_sent']

            for player in state_dict['players']:
                if player['held_object'] is not None:
                    if player['position'] != player['held_object']['position']:
                        player['held_object']['position'] = player['position']
            
            state_obj = OvercookedState.from_dict(state_dict)
            overcooked_states.append(state_obj.deepcopy())
        except:
            pass

    pass