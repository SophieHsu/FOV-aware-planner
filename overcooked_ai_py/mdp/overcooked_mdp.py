import itertools, copy
import numpy as np
import time
from functools import reduce
from collections import defaultdict
from overcooked_ai_py.utils import pos_distance, load_from_json
from overcooked_ai_py import read_layout_dict
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.graphics import *


# class ObjectState(object):
#     """
#     State of an object in OvercookedGridworld.
#     """

#     SOUP_TYPES = ['onion', 'tomato']

#     def __init__(self, name, position, state=None):
#         """
#         name (str): The name of the object
#         position (int, int): Tuple for the current location of the object.
#         state (tuple or None):
#             Extra information about the object. Is None for all objects 
#             except soups, for which `state` is a tuple:
#             (soup_type, num_items, cook_time)
#             where cook_time is how long the soup has been cooking for.
#         """
#         self.id = None
#         self.name = name
#         self.position = tuple(position)
#         if name == 'soup':
#             assert len(state) == 3
#         if state is None:
#             self.state = None 
#         elif type(state) is int:
#             self.state = state
#         else:
#             self.state = tuple(state)

#     def is_valid(self):
#         if self.name in ['onion', 'tomato', 'dish']:
#             return self.state is None
#         elif self.name == 'soup':
#             soup_type, num_items, cook_time = self.state
#             valid_soup_type = soup_type in self.SOUP_TYPES
#             valid_item_num = (1 <= num_items <= 3)
#             valid_cook_time = (0 <= cook_time)
#             return valid_soup_type and valid_item_num and valid_cook_time
#         elif self.name in ['steak', 'garnish', 'hot_plate']:
#             prep_time = self.state
#             valid_prep_time = (0 <= prep_time)
#             return valid_prep_time
#         # Unrecognized object
#         return False

#     def deepcopy(self):
#         return ObjectState(self.name, self.position, self.state)

#     def __eq__(self, other):
#         return isinstance(other, ObjectState) and \
#             self.name == other.name and \
#             self.position == other.position and \
#             self.state == other.state

#     def __hash__(self):
#         return hash((self.name, self.position, self.state))

#     def __repr__(self):
#         if self.state is None:
#             return '{}@{}'.format(self.name, self.position)
#         return '{}@{} with state {}'.format(self.name, self.position,
#                                             str(self.state))

#     def to_dict(self):
#         return {
#             "name": self.name,
#             "position": self.position,
#             "state": self.state
#         }

#     @staticmethod
#     def from_dict(obj_dict):
#         obj_dict = copy.deepcopy(obj_dict)
#         return ObjectState(**obj_dict)

class ObjectState(object):
    """
    State of an object in OvercookedGridworld.
    """

    SOUP_TYPES = ['onion', 'tomato']

    def __init__(self, id, name, position, state=None):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        state (tuple or None):
            Extra information about the object. Is None for all objects 
            except soups, for which `state` is a tuple:
            (soup_type, num_items, cook_time)
            where cook_time is how long the soup has been cooking for.
        """
        self.id = id
        self.name = name
        self.position = tuple(position)
        if name == 'soup':
            assert len(state) == 3
        if state is None:
            self.state = None 
        elif type(state) is int:
            self.state = state
        else:
            self.state = tuple(state)

    def is_valid(self):
        if self.name in ['onion', 'tomato', 'dish']:
            return self.state is None
        elif self.name == 'soup':
            soup_type, num_items, cook_time = self.state
            valid_soup_type = soup_type in self.SOUP_TYPES
            valid_item_num = (1 <= num_items <= 3)
            valid_cook_time = (0 <= cook_time)
            return valid_soup_type and valid_item_num and valid_cook_time
        elif self.name in ['steak', 'garnish', 'hot_plate']:
            prep_time = self.state
            valid_prep_time = (0 <= prep_time)
            return valid_prep_time
        # Unrecognized object
        return False

    def deepcopy(self):
        return ObjectState(self.id, self.name, self.position, self.state)

    def __eq__(self, other):
        return isinstance(other, ObjectState) and \
            self.id == other.id and \
            self.name == other.name and \
            self.position == other.position and \
            self.state == other.state

    def __hash__(self):
        return hash((self.id, self.name, self.position, self.state))

    def __repr__(self):
        if self.state is None:
            return '{}-{}@{}'.format(self.id, self.name, self.position)
        return '{}-{}@{} with state {}'.format(self.id, self.name, self.position,
                                            str(self.state))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            "state": self.state
        }

    @staticmethod
    def from_dict(obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return ObjectState(**obj_dict)


class PlayerState(object):
    """
    State of a player in OvercookedGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
                 None if there is no such object.
    num_ingre_held (int): Number of times the player has held an ingredient object
                         (onion or tomato).
    num_plate_held (int): Number of times the player has held a plate
    num_served (int): Number of times the player has served food
    """
    def __init__(self,
                 position,
                 orientation,
                 held_object=None,
                 num_ingre_held=0,
                 num_plate_held=0,
                 num_served=0,
                 active_log=[],
                 stuck_log=[],
                 subtask_log=[]):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object
        self.num_ingre_held = num_ingre_held
        self.num_plate_held = num_plate_held
        self.num_served = num_served
        self.active_log = active_log.copy()
        self.stuck_log = stuck_log.copy()
        self.subtask_log = subtask_log.copy()

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def get_pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj

    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj

    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy(
        )
        return PlayerState(self.position, self.orientation, new_obj,
                           self.num_ingre_held, self.num_plate_held,
                           self.num_served, self.active_log, self.stuck_log, self.subtask_log)

    def __eq__(self, other):
        return isinstance(other, PlayerState) and \
            self.position == other.position and \
            self.orientation == other.orientation and \
            self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(self.position,
                                                self.orientation,
                                                str(self.held_object))

    def to_dict(self):
        return {
            "position":
            self.position,
            "orientation":
            self.orientation,
            "held_object":
            self.held_object.to_dict()
            if self.held_object is not None else None
        }

    def get_workload(self, ):
        return {
            "num_ingre_held": self.num_ingre_held,
            "num_plate_held": self.num_plate_held,
            "num_served": self.num_served,
        }

    def print_workload(self, ):
        print("Number of ingredients held: {}".format(self.num_ingre_held))
        print("Number of plates held: {}".format(self.num_plate_held))
        print("Number of soup served: {}".format(self.num_served))

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class OvercookedState(object):
    """A state in OvercookedGridworld."""
    def __init__(self, players, objects, order_list, obj_count=0):
        """
        players: List of PlayerStates (order corresponds to player indices).
        objects: Dictionary mapping positions (x, y) to ObjectStates. 
                 NOTE: Does NOT include objects held by players (they are in 
                 the PlayerState objects).
        order_list: Current orders to be delivered

        NOTE: Does not contain time left, which is handled from the environment side.
        """
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        # if order_list is not None:
            # assert all(
                # [o in OvercookedGridworld.ORDER_TYPES for o in order_list])
        self.order_list = order_list
        self.obj_count = obj_count

    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def unowned_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, NOT including
        ones held by players.
        """
        objects_by_type = defaultdict(list)
        for pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects held by players.
        """
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    @property
    def all_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, including
        ones held by players.
        """
        all_objs_by_type = self.unowned_objects_by_type.copy()
        for obj_type, player_objs in self.player_objects_by_type.items():
            all_objs_by_type[obj_type].extend(player_objs)
        return all_objs_by_type

    @property
    def all_objects_list(self):
        all_objects_lists = list(self.all_objects_by_type.values()) + [[], []]
        return reduce(lambda x, y: x + y, all_objects_lists)

    @property
    def curr_order(self):
        return "any" if self.order_list is None else self.order_list[0]

    @property
    def next_order(self):
        return "any" if self.order_list is None else self.order_list[1]

    @property
    def num_orders_remaining(self):
        return np.Inf if self.order_list is None else len(self.order_list)

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    @staticmethod
    def from_players_pos_and_or(players_pos_and_or, order_list):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return OvercookedState([
            PlayerState(*player_pos_and_or)
            for player_pos_and_or in players_pos_and_or
        ],
                               objects={},
                               order_list=order_list)

    @staticmethod
    def from_player_positions(player_positions, order_list):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.SOUTH) for pos in player_positions]
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or,
                                                       order_list)

    def deepcopy(self):
        return OvercookedState(
            [player.deepcopy() for player in self.players],
            {pos: obj.deepcopy()
             for pos, obj in self.objects.items()},
            None if self.order_list is None else list(self.order_list), obj_count=self.obj_count)

    def __eq__(self, other):
        order_list_equal = type(self.order_list) == type(other.order_list) and \
            ((self.order_list is None and other.order_list is None) or \
            (type(self.order_list) is list and np.array_equal(self.order_list, other.order_list)))

        return isinstance(other, OvercookedState) and \
            self.players == other.players and \
            set(self.objects.items()) == set(other.objects.items()) and \
            order_list_equal

    def __hash__(self):
        order_list_hash = tuple(
            self.order_list) if self.order_list is not None else None
        return hash(
            (self.players, tuple(self.objects.values()), order_list_hash))

    def __str__(self):
        return 'Players: {}, Objects: {}, Order list: {}'.format(
            str(self.players), str(list(self.objects.values())),
            str(self.order_list))

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "order_list": self.order_list
        }

    @staticmethod
    def from_dict(state_dict, obj_count=0):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [
            PlayerState.from_dict(p) for p in state_dict["players"]
        ]
        object_list = [ObjectState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return OvercookedState(**state_dict, obj_count=obj_count)

    @staticmethod
    def from_json(filename):
        return load_from_json(filename)

    def print_player_workload(self, ):
        for idx, player in enumerate(self.players):
            print("Player {}".format(idx + 1))
            player.print_workload()
            print()

    def get_player_workload(self, ):
        workloads = []
        for idx, player in enumerate(self.players):
            workloads.append(player.get_workload())
        return workloads

    def cal_concurrent_active_sum(self, ):
        concurrent_active_log = self.cal_concurrent_active_log()
        res = np.sum(concurrent_active_log)
        # res = 0
        # if len(concurrent_active_log) > 0:
        #     res = round(np.sum(concurrent_active_log)/len(concurrent_active_log)*100)

        return res

    def cal_concurrent_active_log(self, ):
        active_logs = self.get_player_active_log()
        if len(active_logs[0]) == 0:
            return []

        # from IPython import embed
        # embed()

        return np.array(active_logs[0]) & np.array(active_logs[1])

    def get_player_active_log(self, ):
        active_log = []
        for idx, player in enumerate(self.players):
            active_log.append(player.active_log)
        return active_log

    def cal_total_stuck_time(self, ):
        stuck_logs = self.get_player_stuck_log()
        res = sum(stuck_logs[0])
        # res = 0
        # if len(stuck_logs[0]) > 0:
        #     res = round(sum(stuck_logs[0])/len(stuck_logs[0])*100)
        return res

    def get_player_stuck_log(self, ):
        stuck_log = []
        for idx, player in enumerate(self.players):
            stuck_log.append(player.stuck_log)
        return stuck_log


NO_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
    "COOKING_STEAK_REW": 0,
}

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "COOKING_STEAK_REW": 3,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}

EVENT_TYPES = [
    # Onion events
    'onion_pickup',
    'useful_onion_pickup',
    'onion_drop',
    'useful_onion_drop',
    'potting_onion',
    'onion_potting',

    # Tomato events
    'tomato_pickup',
    'useful_tomato_pickup',
    'tomato_drop',
    'useful_tomato_drop',
    'potting_tomato',

    # Dish events
    'dish_pickup',
    'useful_dish_pickup',
    'dish_drop',
    'useful_dish_drop',
    'dish_delivery',

    # Soup events
    'soup_pickup',
    'soup_delivery',
    'soup_drop',

    # Meat events
    'meat_pickup',
    'meat_cooking',
    'steak_cooking',
    'steak_pickup',
    'steak_drop',

    # Plate events
    'plate_pickup',
    'plate_drop',
    'plate_washing',
    'plate_heating',

    # Hot plate events
    'hot_plate_pickup',

    # Garnish events
    'onion_chopping',
    'garnish_pickup',
    'garnish_drop',
]

SIMPLE_EVENT_TYPES = [
    # Onion events
    'onion_pickup',
    'onion_drop',
    'onion_potting',

    # Tomato events
    'tomato_pickup',
    'tomato_drop',
    'tomato_potting',

    # Dish events
    'dish_pickup',
    'dish_drop',
    'dish_delivery',

    # Soup events
    'soup_pickup',
    'soup_delivery',
    'soup_drop',

    # Meat events
    'meat_pickup',
    'meat_cooking',
    'steak_cooking',

    # Plate events
    'plate_pickup',
    'plate_washing',
    'plate_heating',

    # Hot plate events
    'hot_plate_pickup',

    # Garnish events
    'onion_chopping',
    'garnish_pickup',
    'garnish_drop',
]


class OvercookedGridworld(object):
    """
    An MDP grid world based off of the Overcooked game.
    TODO: clean the organization of this class further.
    """
    ORDER_TYPES = ObjectState.SOUP_TYPES + ['any']

    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(self,
                 terrain,
                 start_player_positions,
                 start_order_list=None,
                 cook_time=20,
                 num_items_for_soup=3,
                 delivery_reward=20,
                 rew_shaping_params=None,
                 layout_name="unnamed_layout"):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        start_order_list: either a tuple of orders or None if there is not specific list
        cook_time: amount of timesteps required for a soup to cook
        delivery_reward: amount of reward given per delivery
        rew_shaping_params: reward given for completion of specific subgoals
        """
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.start_order_list = start_order_list
        self.soup_cooking_time = cook_time
        self.num_items_for_soup = num_items_for_soup
        self.delivery_reward = delivery_reward
        self.reward_shaping_params = NO_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        self.layout_name = layout_name
        self.viewer = None  # for visualization
        self.pre_players_pos = None  # players position of the previous state, for visualization
        self.pre_objects_pos = None

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params['grid']
        del base_layout_params['grid']
        base_layout_params['layout_name'] = layout_name

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld.from_grid(grid, base_layout_params,
                                             params_to_overwrite)

    @staticmethod
    def from_grid(layout_grid,
                  base_layout_params={},
                  params_to_overwrite={},
                  debug=False):
        """
        Returns instance of OvercookedGridworld with terrain and starting 
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = base_layout_params.copy()

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    layout_grid[y][x] = ' '

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[
                        int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config[k]
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".
                    format(k, curr_val, v))
            mdp_config[k] = v

        return OvercookedGridworld(**mdp_config)

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                self.start_order_list == other.start_order_list and \
                self.soup_cooking_time == other.soup_cooking_time and \
                self.num_items_for_soup == other.num_items_for_soup and \
                self.delivery_reward == other.delivery_reward and \
                self.reward_shaping_params == other.reward_shaping_params and \
                self.layout_name == other.layout_name

    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_order_list=None
            if self.start_order_list is None else list(self.start_order_list),
            cook_time=self.soup_cooking_time,
            num_items_for_soup=self.num_items_for_soup,
            delivery_reward=self.delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name)

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_order_list": self.start_order_list,
            "cook_time": self.soup_cooking_time,
            "num_items_for_soup": self.num_items_for_soup,
            "delivery_reward": self.delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params)
        }

    ##############
    # GAME LOGIC #
    ##############

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [
            self._get_player_actions(state, i)
            for i in range(len(state.players))
        ]

    def _get_player_actions(self, state, player_num):
        """All actions are allowed to all players in all states."""
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action,
                                             self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError('Invalid action')

    def get_standard_start_state(self):
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, order_list=self.start_order_list)
        return start_state

    def get_random_start_state_fn(self,
                                  random_start_pos=False,
                                  rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(
                    len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(
                start_pos, order_list=self.start_order_list)

            if rnd_obj_prob_thresh == 0:
                return start_state

            # Arbitrary hard-coding for randomization of objects
            # For each pot, add a random amount of onions with prob rnd_obj_prob_thresh
            pots = self.get_pot_states(start_state)["empty"]
            for pot_loc in pots:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    n = int(np.random.randint(low=1, high=4))
                    start_state.objects[pot_loc] = ObjectState(
                        "soup", pot_loc, ('onion', n, 0))

            # For each player, add a random object with prob rnd_obj_prob_thresh
            for player in start_state.players:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Different objects have different probabilities
                    obj = np.random.choice(["dish", "onion", "soup"],
                                           p=[0.2, 0.6, 0.2])
                    if obj == "soup":
                        player.set_object(
                            ObjectState(obj, player.position,
                                        ('onion', self.num_items_for_soup,
                                         self.soup_cooking_time)))
                    else:
                        player.set_object(ObjectState(obj, player.position))
            return start_state

        return start_state_fn

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        if state.order_list is None:
            return False
        return len(state.order_list) <= 1

    def get_state_transition(self, state, joint_action):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        events_infos = {
            event: [False] * self.num_players
            for event in EVENT_TYPES
        }

        assert not self.is_terminal(
            state), "Trying to find successor of a terminal state: {}".format(
                state)
        # for action, action_set in zip(joint_action, self.get_actions(state)):
        #     if action not in action_set:
        #         raise ValueError("Illegal action %s in state %s" %
        #                          (action, state))

        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward, shaped_reward = self.resolve_interacts(
            new_state, joint_action, events_infos, rollout=False)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)

        return new_state, sparse_reward, shaped_reward, events_infos

    def resolve_interacts(self, new_state, joint_action, events_infos, rollout=True):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0] * self.num_players, [
            0
        ] * self.num_players

        for player_idx, (player, action) in enumerate(
                zip(new_state.players, joint_action)):

            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == 'X':

                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(events_infos, new_state, obj_name,
                                         pot_states, player_idx)

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)

                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name
                    self.log_object_pickup(events_infos, new_state, obj_name,
                                           pot_states, player_idx)

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

            elif terrain_type == 'O' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "onion",
                                       pot_states, player_idx)

                # Onion pickup from dispenser
                player.set_object(ObjectState('onion', pos))
                player.num_ingre_held += 1

            elif terrain_type == 'T' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "tomato",
                                       pot_states, player_idx)

                # Tomato pickup from dispenser
                player.set_object(ObjectState('tomato', pos))
                player.num_ingre_held += 1

            elif terrain_type == 'D' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "dish",
                                       pot_states, player_idx)
                player.num_plate_held += 1

                # Give shaped reward if pickup is useful
                if self.is_dish_pickup_useful(new_state, pot_states):
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "DISH_PICKUP_REWARD"]

                # Perform dish pickup from dispenser
                obj = ObjectState('dish', pos)
                player.set_object(obj)

            elif terrain_type == 'P' and player.has_object():

                if player.get_object(
                ).name == 'dish' and self.soup_ready_at_location(
                        new_state, i_pos):
                    self.log_object_pickup(events_infos, new_state, "soup",
                                           pot_states, player_idx)

                    # Pick up soup
                    player.remove_object()  # Remove the dish
                    obj = new_state.remove_object(i_pos)  # Get soup
                    player.set_object(obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "SOUP_PICKUP_REWARD"]

                elif player.get_object().name in ['onion', 'tomato']:
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add onion to it
                        player.remove_object()
                        new_state.add_object(
                            ObjectState('soup', i_pos, (item_type, 1, 0)),
                            i_pos)
                        shaped_reward[
                            player_idx] += self.reward_shaping_params[
                                "PLACEMENT_IN_POT_REW"]

                        # Log onion potting
                        events_infos['potting_onion'][player_idx] = True

                    else:
                        # Pot has already items in it, add if not full and of same type
                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'soup', 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        if num_items < self.num_items_for_soup and soup_type == item_type:
                            player.remove_object()
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward[
                                player_idx] += self.reward_shaping_params[
                                    "PLACEMENT_IN_POT_REW"]

                            # Log onion potting
                            events_infos['potting_onion'][player_idx] = True

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':

                    new_state, delivery_rew = self.deliver_soup(
                        new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew
                    player.num_served += 1

                    # Log soup delivery
                    events_infos['soup_delivery'][player_idx] = True

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(
                            new_state.order_list) == 0:
                        break

        return sparse_reward, shaped_reward

    def deliver_soup(self, state, player, soup_obj):
        """
        Deliver the soup, and get reward if there is no order list
        or if the type of the delivered soup matches the next order.
        """
        soup_type, num_items, cook_time = soup_obj.state
        assert soup_type in ObjectState.SOUP_TYPES
        assert num_items == self.num_items_for_soup
        assert cook_time >= self.soup_cooking_time, "Cook time {} mdp cook time {}".format(
            cook_time, self.soup_cooking_time)
        player.remove_object()

        if state.order_list is None:
            return state, self.delivery_reward

        # If the delivered soup is the one currently required
        assert not self.is_terminal(state)
        current_order = state.order_list[0]
        if current_order == 'any' or soup_type == current_order:
            state.order_list = state.order_list[1:]
            return state, self.delivery_reward

        return state, 0

    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations = self.compute_new_positions_and_orientations(
            state.players, joint_action)
        for player_state, new_pos, new_o in zip(state.players, new_positions,
                                                new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states,
                                               joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(zip(*[
            self._move_if_direction(p.position, p.orientation, a) \
            for p, a in zip(old_player_states, joint_action)]))
        old_positions = tuple(p.position for p in old_player_states)
        new_positions = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations

    def is_transition_collision(self, old_positions, new_positions):
        # Checking for any players ending in same square
        if self.is_joint_position_collision(new_positions):
            return True
        # Check if any two players crossed paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p1_old, p2_old = old_positions[idx0], old_positions[idx1]
            p1_new, p2_new = new_positions[idx0], new_positions[idx1]
            if p1_new == p2_old and p1_old == p2_new:
                return True
        return False

    def is_joint_position_collision(self, joint_position):
        return any(pos0 == pos1
                   for pos0, pos1 in itertools.combinations(joint_position, 2))

    def step_environment_effects(self, state):
        for obj in state.objects.values():
            if obj.name == 'soup':
                x, y = obj.position
                soup_type, num_items, cook_time = obj.state
                # NOTE: cook_time is capped at self.soup_cooking_time
                if self.terrain_mtx[y][x] == 'P' and \
                    num_items == self.num_items_for_soup and \
                    cook_time < self.soup_cooking_time:
                    obj.state = soup_type, num_items, cook_time + 1

    def _handle_collisions(self, old_positions, new_positions):
        """If agents collide, they stay at their old locations"""
        if self.is_transition_collision(old_positions, new_positions):
            return old_positions
        return new_positions

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns position and orientation that would 
        be obtained after executing action"""
        if action not in Action.MOTION_ACTIONS:
            return position, orientation
        new_pos = Action.move_in_direction(position, action)
        new_orientation = orientation if action == Action.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation

    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_valid_player_positions(self):
        return self.terrain_pos_dict[' ']

    def get_valid_joint_player_positions(self):
        """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
        valid_positions = self.get_valid_player_positions()
        all_joint_positions = list(
            itertools.product(valid_positions, repeat=self.num_players))
        valid_joint_positions = [
            j_pos for j_pos in all_joint_positions
            if not self.is_joint_position_collision(j_pos)
        ]
        return valid_joint_positions

    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations(
        )

        valid_joint_player_states = []
        for players_pos_and_orientations in itertools.product(
                valid_player_states, repeat=self.num_players):
            joint_position = [
                plyer_pos_and_or[0]
                for plyer_pos_and_or in players_pos_and_orientations
            ]
            if not self.is_joint_position_collision(joint_position):
                valid_joint_player_states.append(players_pos_and_orientations)

        return valid_joint_player_states

    def get_adjacent_features(self, player):
        adj_feats = []
        pos = player.position
        for d in Direction.ALL_DIRECTIONS:
            adj_pos = Action.move_in_direction(pos, d)
            adj_feats.append((adj_pos, self.get_terrain_type_at_pos(adj_pos)))
        return adj_feats

    def get_terrain_type_at_pos(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])
    
    def get_plate_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    def get_onion_dispenser_locations(self):
        return list(self.terrain_pos_dict['O'])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['P'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])

    def get_key_objects_locations(self):
        return self.get_dish_dispenser_locations() + self.get_dish_dispenser_locations() + self.get_onion_dispenser_locations() + self.get_tomato_dispenser_locations() + self.get_serving_locations() + self.get_pot_locations()

    @property
    def num_pots(self):
        return len(self.get_pot_locations())

    def get_pot_states(self, state, pots_states_dict=None, valid_pos=None):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        if pots_states_dict is None:
            pots_states_dict = {}
            pots_states_dict['empty'] = []
            pots_states_dict['onion'] = defaultdict(list)
            pots_states_dict['tomato'] = defaultdict(list)

        get_pot_info = []
        if valid_pos is not None:
            for pot_pos in self.get_pot_locations():
                if pot_pos in valid_pos:
                    get_pot_info.append(pot_pos)
        else:
            get_pot_info = self.get_pot_locations()

        for pot_pos in get_pot_info:
            # reset
            for k, v in pots_states_dict.items():
                if type(v) == list:
                    if pot_pos in v:
                        v.remove(pot_pos)
                else:
                    for d_k, d_v in v.items():
                        if pot_pos in d_v:
                            d_v.remove(pot_pos)

            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)

            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['{}_items'.format(num_items)].append(pot_pos)
                elif num_items == self.num_items_for_soup:
                    assert cook_time <= self.soup_cooking_time
                    if cook_time == self.soup_cooking_time:
                        pots_states_dict[soup_type]['ready'].append(pot_pos)
                    else:
                        pots_states_dict[soup_type]['cooking'].append(pot_pos)
                else:
                    raise ValueError("Pot with more than {} items".format(
                        self.num_items_for_soup))

                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['partially_full'].append(pot_pos)

        return pots_states_dict

    def get_pot_state(self, state, pot_pos):
        """Returns pot status based on the position of the pot"""
        pot_status = [None, None, None]

        if not state.has_object(pot_pos):
            pot_status = [None, 'empty', 0] # soup type, cooking status, num of items
        else:
            soup_obj = state.get_object(pot_pos)
            soup_type, num_items, cook_time = soup_obj.state
            if 0 < num_items < self.num_items_for_soup:
                pot_status = [soup_type, None, num_items]
            elif num_items == self.num_items_for_soup:
                assert cook_time <= self.soup_cooking_time
                if cook_time == self.soup_cooking_time:
                    pot_status = [soup_type, 'ready', num_items]
                else:
                    pot_status = [soup_type, 'cooking', num_items]
            else:
                raise ValueError("Pot with more than {} items".format(
                    self.num_items_for_soup))

            if 0 < num_items < self.num_items_for_soup:
                    pot_status = [soup_type, 'partially_full', num_items]

        return pot_status

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        counters_considered = self.terrain_pos_dict[
            'X'] if counter_subset is None else counter_subset
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counter_locations(self, state):
        counter_locations = self.get_counter_locations()
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def get_empty_pots(self, pot_states):
        return pot_states["empty"]

    def get_ready_pots(self, pot_states):
        return pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]

    def get_cooking_pots(self, pot_states):
        return pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]

    def get_full_pots(self, pot_states):
        return self.get_cooking_pots(pot_states) + self.get_ready_pots(
            pot_states)

    def get_partially_full_pots(self, pot_states):
        return pot_states["tomato"]["partially_full"] + pot_states["onion"][
            "partially_full"]

    def soup_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name in ['soup', 'steak'], 'Object in pot was not soup'
        _, num_items, cook_time = obj.state
        return num_items == self.num_items_for_soup and cook_time >= self.soup_cooking_time

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert player_state.held_object.position == player_state.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != ' '

        # Check that players and non-held objects don't overlap
        all_pos = [player_state.position for player_state in state.players]
        all_pos += [obj_state.position for obj_state in state.objects.values()]
        assert len(all_pos) == len(
            set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            assert obj_state.is_valid()

    def find_free_counters_valid_for_both_players(self, state, mlp):
        """Finds all empty counter locations that are accessible to both players"""
        one_player, other_player = state.players
        free_counters = self.get_empty_counter_locations(state)
        free_counters_valid_for_both = []
        for free_counter in free_counters:
            goals = mlp.mp.motion_goals_for_pos[free_counter]
            if any([mlp.mp.is_valid_motion_start_goal_pair(one_player.pos_and_or, goal) for goal in goals]) and \
            any([mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal) for goal in goals]):
                free_counters_valid_for_both.append(free_counter)
        return free_counters_valid_for_both

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in 'XOPDSTWMB'

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players +
                                           1)), "Some players were missing"

        assert all(c in 'XOPDSTBWM123456789 '
                   for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count(
            'D') >= 1, "'D' must be present at least once"
        assert all_elements.count(
            'S') >= 1, "'S' must be present at least once"
        assert all_elements.count(
            'P') >= 1, "'P' must be present at least once"
        assert all_elements.count('O') >= 1 or all_elements.count(
            'T') >= 1, "'O' or 'T' must be present at least once"

    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    def log_object_pickup(self, events_infos, state, obj_name, pot_states,
                          player_index):
        """Player picked an object up from a counter or a dispenser"""
        obj_pickup_key = obj_name + "_pickup"
        if obj_pickup_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_pickup_key))
        events_infos[obj_pickup_key][player_index] = True

        USEFUL_PICKUP_FNS = {
            "onion": self.is_onion_pickup_useful,
            "dish": self.is_dish_pickup_useful
        }
        if obj_name in USEFUL_PICKUP_FNS:
            if USEFUL_PICKUP_FNS[obj_name](state, pot_states, player_index):
                obj_useful_key = "useful_" + obj_name + "_pickup"
                events_infos[obj_useful_key][player_index] = True

    def log_object_drop(self, events_infos, state, obj_name, pot_states,
                        player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        if obj_drop_key not in events_infos:
            # TODO: add support for tomato event logging
            if obj_name == "tomato":
                return
            raise ValueError("Unknown event {}".format(obj_drop_key))
        events_infos[obj_drop_key][player_index] = True

        USEFUL_DROP_FNS = {
            "onion": self.is_onion_drop_useful,
            "dish": self.is_dish_drop_useful
        }
        if obj_name in USEFUL_DROP_FNS:
            if USEFUL_DROP_FNS[obj_name](state, pot_states, player_index):
                obj_useful_key = "useful_" + obj_name + "_drop"
                events_infos[obj_useful_key][player_index] = True

    def is_dish_pickup_useful(self, state, pot_states, player_index=None):
        """
        NOTE: this only works if self.num_players == 2
        Useful if:
        - Pot is ready/cooking and there is no player with a dish               \ 
        - 2 pots are ready/cooking and there is one player with a dish          | -> number of dishes in players hands < number of ready/cooking/partially full soups
        - Partially full pot is ok if the other player is on course to fill it  /

        We also want to prevent picking up and dropping dishes, so add the condition
        that there must be no dishes on counters
        """
        if self.num_players != 2: return False

        # This next line is to prevent reward hacking (this logic is also used by reward shaping)
        dishes_on_counters = self.get_counter_objects_dict(state)["dish"]
        no_dishes_on_counters = len(dishes_on_counters) == 0

        num_player_dishes = len(state.player_objects_by_type['dish'])
        non_empty_pots = len(
            self.get_ready_pots(pot_states) +
            self.get_cooking_pots(pot_states) +
            self.get_partially_full_pots(pot_states))
        return no_dishes_on_counters and num_player_dishes < non_empty_pots

    def is_dish_drop_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Useful if:
        - Onion is needed (all pots are non-full)
        - Nobody is holding onions
        """
        if self.num_players != 2: return False
        all_non_full = len(self.get_full_pots(pot_states)) == 0
        other_player = state.players[1 - player_index]
        other_player_holding_onion = other_player.has_object(
        ) and other_player.get_object().name == "onion"
        return all_non_full and not other_player_holding_onion

    def is_onion_pickup_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Always useful unless:
        - All pots are full & other agent is not holding a dish
        """
        if self.num_players != 2: return False
        all_pots_full = self.num_pots == len(self.get_full_pots(pot_states))
        other_player = state.players[1 - player_index]
        other_player_has_dish = other_player.has_object(
        ) and other_player.get_object().name == "dish"
        return not (all_pots_full and not other_player_has_dish)

    def is_onion_drop_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Useful if:
        - Dish is needed (all pots are full)
        - Nobody is holding a dish
        """
        if self.num_players != 2: return False
        all_pots_full = len(self.get_full_pots(pot_states)) == self.num_pots
        other_player = state.players[1 - player_index]
        other_player_holding_dish = other_player.has_object(
        ) and other_player.get_object().name == "dish"
        return all_pots_full and not other_player_holding_dish

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    player_idx_lst = [
                        i for i, p in enumerate(state.players)
                        if p.position == player.position
                    ]
                    assert len(player_idx_lst) == 1, players_dict

                    grid_string += Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                        grid_string += str(player_idx_lst[0])
                    else:
                        grid_string += str(player_idx_lst[0])
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        if soup_type == "onion":
                            grid_string += "ø"
                        elif soup_type == "tomato":
                            grid_string += "†"
                        else:
                            raise ValueError()

                        if num_items == self.num_items_for_soup:
                            grid_string += str(cook_time)

                        # NOTE: do not currently have terminal graphics
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string += "="
                        else:
                            grid_string += "-"
                    else:
                        grid_string += element + " "

            grid_string += "\n"

        if state.order_list is not None:
            grid_string += "Current orders: {}/{} are any's\n".format(
                len(state.order_list),
                len([order == "any" for order in state.order_list]))
        return grid_string

    ###################
    # RENDER FUNCTION #
    ###################

    def check_viewpoint(self, player_pos, player_ori, x, y, view_angle=145):
        ori = Direction.DIRECTION_TO_INDEX[player_ori]
        # center_pt = np.array(player_pos)
        # if ori == 0: # north
        #     center_pt[1] += 1
        # elif ori == 2: # east
        #     center_pt[0] -= 1
        # elif ori == 1: # south
        #     center_pt[1] -= 1
        # elif ori == 3: # west
        #     center_pt[0] += 1
        
        item_ang = np.arctan2((y-player_pos[1]), (x-player_pos[0]))*180/np.pi

        # if x == center_pt[0] and y == center_pt[1]:
        #     return False
        if (x,y) == player_pos:
            return True
        if ori == 1: # north
            if item_ang <= (90+(view_angle/2)) and (item_ang >= (90-(view_angle/2))):
                return True
            if player_pos[1] == y and ((player_pos[0] == x-1) or (player_pos[0] == x+1)) and view_angle >= 120:
                return True
        elif ori == 2: # east
            if item_ang <= (0+(view_angle/2)) and (item_ang >= (0-(view_angle/2))):
                return True
            if player_pos[0] == x and ((player_pos[1] == y-1) or (player_pos[1] == y+1)) and view_angle >= 120:
                return True
        elif ori == 0: # south
            if item_ang <= (-90+(view_angle/2)) and (item_ang >= (-90-(view_angle/2))):
                return True
            if player_pos[1] == y and ((player_pos[0] == x-1) or (player_pos[0] == x+1)) and view_angle >= 120:
                return True
        elif ori == 3: # west
            if item_ang <= (-180+(view_angle/2)) or (item_ang >= (180-(view_angle/2))):
                return True
            if player_pos[0] == x and ((player_pos[1] == y-1) or (player_pos[1] == y+1)) and view_angle >= 120:
                return True
        return False

    def render(self, state, mode, time_step_left=None, time_passed=None, selected_action_count=0, view_angle=120):
        """
        Function that renders the game

        Args:
            state(OvercookedState): state to render
            mode (string): mode of rendering
                           For "human", render the game without a info panel
                           For "blur", render the game with trajectory of both
                               players
                           For "full", render the game with a info panel
            time_step_left(int): timestep left for the game
        """
        players_dict = {player.position: player for player in state.players}
        objects_pos = []  # list of positions of the objects

        # set window size; SPRITE_LENGTH is the length of each squared sprite, which could be tuned in graphics.py
        if mode == "full":
            window_size = self.width*SPRITE_LENGTH,\
                          self.height*SPRITE_LENGTH + INFO_PANEL_HEIGHT
        else:
            window_size = self.width*SPRITE_LENGTH, self.height*SPRITE_LENGTH

        if self.viewer == None:
            # create viewer
            self.viewer = pygame.display.set_mode(window_size)
            self.viewer.fill((255, 255, 255)) # white background
            # render the terrain
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain in enumerate(terrain_row):
                blit_terrain(x, y, self.terrain_mtx, self.viewer, mode)

        # remove the objects on the counters and pots
        if self.pre_objects_pos is not None:
            for pos in self.pre_objects_pos:
                x, y = pos
                blit_terrain(x, y, self.terrain_mtx, self.viewer, mode)

        # render objects at the new locations
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain in enumerate(terrain_row):
                curr_pos = get_curr_pos(x, y, mode)
                # there is object on a counter
                if terrain == "X" and state.has_object((x, y)):
                    state_obj = state.get_object((x, y))
                    obj_pgobj = get_object_sprite(state_obj)
                    self.viewer.blit(obj_pgobj, curr_pos)
                    objects_pos.append((x, y))
                # there is soup on a pot
                elif terrain == "P" and state.has_object((x, y)):
                    soup_obj = state.get_object((x, y))
                    soup_type, num_items, cook_time = soup_obj.state
                    # if soup is ready
                    if self.soup_ready_at_location(state, (x, y)):
                        soup_pgobj = load_image(
                            os.path.join(ASSETS_DIR, OBJECT_DIR,
                                         'soup-%s-cooked.png' % soup_type))
                    # if soup is not ready
                    else:
                        soup_pgobj = get_object_sprite(soup_obj, on_pot=True)
                    
                    self.viewer.blit(soup_pgobj, curr_pos)
                    objects_pos.append((x, y))

                    # render cook time for current soup if cooking starts
                    if num_items == self.num_items_for_soup:
                        cook_time_text_surface = get_text_sprite(
                            str(cook_time))
                        text_pos = cook_time_text_surface.get_rect()
                        # align midbottom of the textbox to midbottom of
                        # current terrain
                        text_pos.midbottom = curr_pos.midbottom
                        # if not self.check_viewpoint(state.players[1].position, state.players[1].orientation, text_pos.midbottom[0], text_pos.midbottom[1]):
                        #     cook_time_text_surface.set_alpha(1)
                        self.viewer.blit(cook_time_text_surface, text_pos)

        if mode == "human" or mode == "full" or mode == "fog":
            # remove chefs from last state
            if self.pre_players_pos is not None:
                for pos in self.pre_players_pos:
                    x, y = pos
                    blit_terrain(x, y, self.terrain_mtx, self.viewer, mode)

        # render the chefs at new location
        for pos, player in players_dict.items():
            x, y = pos
            curr_pos = get_curr_pos(x, y, mode)

            # check player position conflicts
            player_idx_lst = [
                i for i, p in enumerate(state.players)
                if p.position == player.position
            ]
            assert len(player_idx_lst) == 1

            player_pgobj, player_hat_pgobj = get_player_sprite(
                player, player_idx_lst[0])

            if mode == "blur":
                transparent = (time_step_left+10)*5
                print(transparent)
                player_pgobj.set_alpha(transparent)
                player_hat_pgobj.set_alpha(transparent)
                # draw_arrow(self.viewer, player, player_idx_lst[0], curr_pos,
                           # time_step_left)

            self.viewer.blit(player_pgobj, curr_pos)
            self.viewer.blit(player_hat_pgobj, curr_pos)                

        # update previous players and objects positions
        self.pre_players_pos = players_dict.keys()
        if len(objects_pos) > 0:
            self.pre_objects_pos = objects_pos
        else:
            self.pre_objects_pos = None

        # render the game info panel
        if mode == "full":
            render_game_info_panel(
                self.viewer,
                (self.width * SPRITE_LENGTH, self.height * SPRITE_LENGTH),
                state.num_orders_remaining, time_passed)

        if mode == "fog": # fog the terrain
            for y, terrain_row in enumerate(self.terrain_mtx):
                for x, terrain in enumerate(terrain_row):
                    in_view = self.check_viewpoint(state.players[1].position, state.players[1].orientation, x, y, view_angle=view_angle)
                    if not in_view:
                        curr_pos = get_curr_pos(x, y, mode)
                        blit_terrain(x, y, self.terrain_mtx, self.viewer, mode, in_view)
                        fog_pgobj = pygame.Surface((50,50))
                        fog_pgobj.fill((0,0,0))
                        # load_image(os.path.join(ASSETS_DIR, TERRAIN_DIR, 'counter.png'))
                        fog_pgobj.set_alpha(100)
                        self.viewer.blit(fog_pgobj, curr_pos)

        # update display
        pygame.display.update()

    ###################
    # STATE ENCODINGS #
    ###################

    def lossless_state_encoding(self, overcooked_state, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert self.num_players == 2, "Functionality has to be added to support encondings for > 2 players"
        assert type(debug) is bool
        base_map_features = [
            "pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc",
            "serve_loc"
        ]
        variable_map_features = [
            "onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes",
            "onions"
        ]

        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
            layer = np.zeros(self.shape)
            layer[position] = value
            return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = ["player_{}_loc".format(primary_agent_idx), "player_{}_loc".format(other_agent_idx)] + \
                        ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                        for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.ALL_DIRECTIONS)]

            LAYERS = ordered_player_features + base_map_features + variable_map_features
            state_mask_dict = {k: np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_dish_dispenser_locations():
                state_mask_dict["dish_disp_loc"][loc] = 1

            for loc in self.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[
                    player.orientation]
                state_mask_dict["player_{}_loc".format(i)] = make_layer(
                    player.position, 1)
                state_mask_dict["player_{}_orientation_{}".format(
                    i,
                    player_orientation_idx)] = make_layer(player.position, 1)

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "soup":
                    soup_type, num_onions, cook_time = obj.state
                    if soup_type == "onion":
                        if obj.position in self.get_pot_locations():
                            soup_type, num_onions, cook_time = obj.state
                            state_mask_dict["onions_in_pot"] += make_layer(
                                obj.position, num_onions)
                            state_mask_dict["onions_cook_time"] += make_layer(
                                obj.position, cook_time)
                        else:
                            # If player soup is not in a pot, put it in separate mask
                            state_mask_dict["onion_soup_loc"] += make_layer(
                                obj.position, 1)
                    else:
                        raise ValueError("Unrecognized soup")

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array(
                [state_mask_dict[layer_id] for layer_id in LAYERS])
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        final_obs_for_players = tuple(
            process_for_player(i) for i in range(num_players))
        return final_obs_for_players

    def featurize_state(self, overcooked_state, mlp):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(
                idx, name)] = self.get_deltas_to_closest_location(
                    player, locations, mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(
                4)[orientation_idx]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(
                    len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(
                    i, "onion",
                    self.get_onion_dispenser_locations() +
                    counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot",
                                 pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot",
                                 pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot",
                                 pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(
                    i, "dish",
                    self.get_dish_dispenser_locations() +
                    counter_objects["dish"])

            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            for direction, pos_and_feat in enumerate(
                    self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat

                if direction == player.orientation:
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos
                                      not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    # NOTE: Really, this feature should have been "closest empty counter"
                    all_features["p{}_facing_empty_counter".format(
                        i)] = facing_counter_feature

                all_features["p{}_wall_{}".format(
                    i, direction)] = [0] if feat == ' ' else [1]

        features_np = {k: np.array(v) for k, v in all_features.items()}

        p0, p1 = overcooked_state.players
        p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(
            np.concatenate(
                [p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p1.position)
        ordered_features_p1 = np.squeeze(
            np.concatenate(
                [p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1

    def get_deltas_to_closest_location(self, player, locations, mlp):
        _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or,
                                                    locations,
                                                    with_argmin=True)
        if closest_loc is None:
            # "any object that does not exist or I am carrying is going to show up as a (0,0)
            # but I can disambiguate the two possibilities by looking at the features
            # for what kind of object I'm carrying"
            return (0, 0)
        dy_loc, dx_loc = pos_distance(closest_loc, player.position)
        return dy_loc, dx_loc

    ##############
    # DEPRECATED #
    ##############

    def calculate_distance_based_shaped_reward(self, state, new_state):
        """
        Adding reward shaping based on distance to certain features.
        """
        distance_based_shaped_reward = 0

        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"][
            "ready"]
        cooking_pots = ready_pots + pot_states["tomato"][
            "cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"][
            "partially_full"] + pot_states["onion"]["partially_full"]
        dishes_in_play = len(new_state.player_objects_by_type['dish'])
        for player_old, player_new in zip(state.players, new_state.players):
            # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
            max_dist = 8

            if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(
                    nearly_ready_pots) >= dishes_in_play:
                min_dist_to_pot_new = np.inf
                min_dist_to_pot_old = np.inf
                for pot in nearly_ready_pots:
                    new_dist = np.linalg.norm(
                        np.array(pot) - np.array(player_new.position))
                    old_dist = np.linalg.norm(
                        np.array(pot) - np.array(player_old.position))
                    if new_dist < min_dist_to_pot_new:
                        min_dist_to_pot_new = new_dist
                    if old_dist < min_dist_to_pot_old:
                        min_dist_to_pot_old = old_dist
                if min_dist_to_pot_old > min_dist_to_pot_new:
                    distance_based_shaped_reward += self.reward_shaping_params[
                        "POT_DISTANCE_REW"] * (
                            1 - min(min_dist_to_pot_new / max_dist, 1))

            if player_new.held_object is None and len(
                    cooking_pots) > 0 and dishes_in_play == 0:
                min_dist_to_d_new = np.inf
                min_dist_to_d_old = np.inf
                for serving_loc in self.terrain_pos_dict['D']:
                    new_dist = np.linalg.norm(
                        np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(
                        np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_d_new:
                        min_dist_to_d_new = new_dist
                    if old_dist < min_dist_to_d_old:
                        min_dist_to_d_old = old_dist

                if min_dist_to_d_old > min_dist_to_d_new:
                    distance_based_shaped_reward += self.reward_shaping_params[
                        "DISH_DISP_DISTANCE_REW"] * (
                            1 - min(min_dist_to_d_new / max_dist, 1))

            if player_new.held_object is not None and player_new.held_object.name == 'soup':
                min_dist_to_s_new = np.inf
                min_dist_to_s_old = np.inf
                for serving_loc in self.terrain_pos_dict['S']:
                    new_dist = np.linalg.norm(
                        np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(
                        np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_s_new:
                        min_dist_to_s_new = new_dist

                    if old_dist < min_dist_to_s_old:
                        min_dist_to_s_old = old_dist

                if min_dist_to_s_old > min_dist_to_s_new:
                    distance_based_shaped_reward += self.reward_shaping_params[
                        "SOUP_DISTANCE_REW"] * (
                            1 - min(min_dist_to_s_new / max_dist, 1))

        return distance_based_shaped_reward

class SteakHouseGridworld(OvercookedGridworld):
    """
    An MDP grid world extension of the Overcooked game.
    TODO: clean the organization of this class further.
    """
    ORDER_TYPES = ObjectState.SOUP_TYPES + ['steak','any']

    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(self, terrain, start_player_positions, start_order_list=None, cook_time=20, num_items_for_steak=1, chop_time=3, wash_time=3, delivery_reward=20, rew_shaping_params=None, layout_name="unnamed_layout", object_id_dict={}):
        super().__init__(terrain, start_player_positions, start_order_list, cook_time, num_items_for_steak, delivery_reward, rew_shaping_params, layout_name)
        self.steak_cooking_time = cook_time
        self.chopping_time = chop_time
        self.wash_time = wash_time
        self.num_items_for_steak = num_items_for_steak
        self.object_id_dict = object_id_dict


    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a SteakHouseGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params['grid']
        del base_layout_params['grid']
        base_layout_params['layout_name'] = layout_name

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return SteakHouseGridworld.from_grid(grid, base_layout_params,
                                             params_to_overwrite)
    
    @staticmethod
    def from_grid(layout_grid,
                base_layout_params={},
                params_to_overwrite={},
                debug=False):
        """
        Returns instance of SteakHouseGridworld with terrain and starting 
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = base_layout_params.copy()

        layout_grid = [[c for c in row] for row in layout_grid]
        SteakHouseGridworld._assert_valid_grid(layout_grid)

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    layout_grid[y][x] = ' '

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[
                        int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config[k]
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".
                    format(k, curr_val, v))
            mdp_config[k] = v

        return SteakHouseGridworld(**mdp_config)
    
    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                self.start_order_list == other.start_order_list and \
                self.steak_cooking_time == other.steak_cooking_time and \
                self.chopping_time == other.chopping_time and \
                self.wash_time == other.wash_time and \
                self.delivery_reward == other.delivery_reward and \
                self.reward_shaping_params == other.reward_shaping_params and \
                self.layout_name == other.layout_name
    
    def copy(self):
        return SteakHouseGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_order_list=None
            if self.start_order_list is None else list(self.start_order_list),
            cook_time=self.steak_cooking_time,
            chop_time=self.chopping_time,
            wash_time=self.wash_time,
            delivery_reward=self.delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
            object_id_dict=copy.deepcopy(self.object_id_dict))
    
    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_order_list": self.start_order_list,
            "cook_time": self.soup_cooking_time,
            "chop_time": self.chopping_time,
            "wash_time": self.wash_time,
            "delivery_reward": self.delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params)
        }
    
    ##############
    # GAME LOGIC #
    ##############

    def resolve_interacts(self, new_state, joint_action, events_infos, rollout=True):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0] * self.num_players, [
            0
        ] * self.num_players

        for player_idx, (player, action) in enumerate(
                zip(new_state.players, joint_action)):

            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)
            if not rollout:
                obj_count = len(self.object_id_dict)
            else:
                obj_count = new_state.obj_count
            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == 'X':

                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(events_infos, new_state, obj_name,
                                         pot_states, player_idx)

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)

                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name
                    self.log_object_pickup(events_infos, new_state, obj_name,
                                           pot_states, player_idx)

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)
                
                elif player.has_object() and new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    player_obj = player.remove_object()

                    # Pick up object from counter
                    self.log_object_pickup(events_infos, new_state, obj_name,
                                           pot_states, player_idx)
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

                    # Drop object on counter
                    self.log_object_drop(events_infos, new_state, obj_name,
                                         pot_states, player_idx)
                    new_state.add_object(player_obj, i_pos)


            elif terrain_type == 'O' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "onion",
                                       pot_states, player_idx)

                # Onion pickup from dispenser
                new_o_id = obj_count
                o = ObjectState(new_o_id, 'onion', pos)
                if not rollout: self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                player.num_ingre_held += 1

            elif terrain_type == 'M' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "meat",
                                       pot_states, player_idx)

                # Tomato pickup from dispenser
                new_o_id = obj_count
                o = ObjectState(new_o_id, 'meat', pos)
                if not rollout: self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                player.num_ingre_held += 1

            elif terrain_type == 'D' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "plate",
                                       pot_states, player_idx)
                player.num_plate_held += 1

                # Give shaped reward if pickup is useful
                # if self.is_plate_pickup_useful(new_state, pot_states):
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    #     "PLATE_PICKUP_REWARD"]

                # Perform dish pickup from dispenser
                new_o_id = obj_count
                o = ObjectState(new_o_id, 'plate', pos)
                if not rollout: self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)

            elif terrain_type == 'W':
                if player.held_object is None:
                    # pick up clean plates
                    if self.plate_hot_at_location(new_state, i_pos):
                        self.log_object_pickup(events_infos, new_state, "hot_plate",
                                        pot_states, player_idx)
                        obj = new_state.remove_object(i_pos)
                        player.set_object(obj)
                        # Give shaped reward if pickup is useful
                        # if self.is_plate_pickup_useful(new_state, pot_states):
                            # shaped_reward[player_idx] += self.reward_shaping_params["PLATE_PICKUP_REWARD"]
                        
                        player.num_plate_held += 1
                    
                    # wash plates
                    else:
                        if new_state.has_object(i_pos):
                            obj = new_state.get_object(i_pos)
                            wash_time = obj.state
                            if wash_time < self.wash_time:
                                obj.state = wash_time + 1
                                # shaped_reward[
                                #     player_idx] += self.reward_shaping_params[
                                #         "WASH_PLATE_REW"]

                                # Log onion potting
                                events_infos['plate_heating'][player_idx] = True

                else: # sink is empty and put plate
                    if player.get_object().name == "plate" and not new_state.has_object(i_pos):  
                        obj_name = player.get_object().name
                        self.log_object_drop(events_infos, new_state, obj_name,
                                            pot_states, player_idx)

                    
                        # Drop object on counter
                        obj = player.remove_object()
                        new_o_id = obj_count
                        new_obj = ObjectState(new_o_id, 'hot_plate', i_pos, 0)
                        if not rollout: self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_state.add_object(new_obj, i_pos) # wash time = 0

            elif terrain_type == 'P' and player.has_object():

                if player.get_object(
                ).name == 'hot_plate' and self.steak_ready_at_location(
                        new_state, i_pos):
                    self.log_object_pickup(events_infos, new_state, "steak",
                                           pot_states, player_idx)

                    # Pick up steak
                    player.remove_object()  # Remove the hot plate
                    obj = new_state.remove_object(i_pos)  # Get steak
                    player.set_object(obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                        # "STEAK_PICKUP_REWARD"]
                
                elif player.get_object().name == 'meat':
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add meat to it
                        player.remove_object()
                        new_o_id = obj_count
                        new_obj = ObjectState(new_o_id, 'steak', i_pos, ('steak', 1, 0))
                        if not rollout: self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_state.add_object(new_obj, i_pos)
                        shaped_reward[
                            player_idx] += self.reward_shaping_params[
                                "PLACEMENT_IN_POT_REW"]

                        # Log meat cooking
                        events_infos['steak_cooking'][player_idx] = True

                    else:
                        # Pot has already meat in it
                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'steak', 'Object in pot was not steak'
                        food_type, num_items, cook_time = obj.state
                        if cook_time < self.steak_cooking_time:
                            obj.state = food_type, num_items, cook_time + 1
                            shaped_reward[
                                player_idx] += self.reward_shaping_params[
                                    "COOKING_STEAK_REW"]

                            # Log onion potting
                            events_infos['steak_cooking'][player_idx] = True

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'dish':
                    new_state, delivery_rew = self.deliver_dish(
                        new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew
                    player.num_served += 1

                    # Log dish delivery
                    events_infos['dish_delivery'][player_idx] = True

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(
                            new_state.order_list) == 0:
                        break

            elif terrain_type == 'B':
                if player.held_object is None:
                    if new_state.has_object(i_pos):
                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'garnish', 'Object on chopping board was not garnish'
                        chop_time = obj.state
                        if chop_time < self.chopping_time:
                            obj.state = chop_time + 1
                            # shaped_reward[
                            #     player_idx] += self.reward_shaping_params[
                            #         "CHOPPING_ONION_REW"]

                            # Log onion chopping
                            events_infos['onion_chopping'][player_idx] = True

                elif player.get_object().name == 'onion' and not new_state.has_object(i_pos):
                    # Chopping board was empty, add onion to it
                    player.remove_object()
                    new_o_id = obj_count
                    new_obj = ObjectState(new_o_id, 'garnish', i_pos, 0)
                    if not rollout: self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    new_state.add_object(new_obj, i_pos)
                    # shaped_reward[
                        # player_idx] += self.reward_shaping_params[
                            # "PLACEMENT_ON_BOARD_REW"]

                    # Log onion potting
                    events_infos['onion_chopping'][player_idx] = True

                # Pick up garnish
                elif player.get_object().name == 'steak' and self.garnish_ready_at_location(new_state, i_pos):
                    player.remove_object() # Remove the hot plate
                    self.log_object_pickup(events_infos, new_state, "dish",
                                           pot_states, player_idx)

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = ObjectState(new_o_id, 'dish', pos)
                    if not rollout: self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    #     "GARNISH_STEAK_REWARD"]
            
            new_state.obj_count = obj_count
        return sparse_reward, shaped_reward
    
    def deliver_dish(self, state, player, dish_obj):
        """
        Deliver the steak, and get reward if there is no order list
        or if the type of the delivered steak matches the next order.
        """
        player.remove_object()

        if state.order_list is None:
            return state, self.delivery_reward

        # If the delivered soup is the one currently required
        # assert not self.is_terminal(state)
        current_order = state.order_list[0]
        if dish_obj.name == 'dish' and 'steak' == current_order:
            state.order_list = state.order_list[1:]
            return state, self.delivery_reward

        return state, 0
    
    def step_environment_effects(self, state):
        for obj in state.objects.values():
            if obj.name == 'soup':
                x, y = obj.position
                soup_type, num_items, cook_time = obj.state
                # NOTE: cook_time is capped at self.soup_cooking_time
                if self.terrain_mtx[y][x] == 'P' and \
                    num_items == self.num_items_for_soup and \
                    cook_time < self.soup_cooking_time:
                    obj.state = soup_type, num_items, cook_time + 1

            elif obj.name == 'steak':
                x, y = obj.position
                food_type, num_items, cook_time = obj.state
                # NOTE: cook_time is capped at self.soup_cooking_time
                if self.terrain_mtx[y][x] == 'P' and \
                    cook_time < self.steak_cooking_time:
                    obj.state = food_type, num_items, cook_time + 1
    
    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_chopping_board_locations(self):
        return list(self.terrain_pos_dict['B'])

    def get_meat_dispenser_locations(self):
        return list(self.terrain_pos_dict['M'])

    def get_sink_locations(self):
        return list(self.terrain_pos_dict['W'])
    
    def get_key_objects_locations(self):
        return self.get_dish_dispenser_locations() + self.get_dish_dispenser_locations() + self.get_onion_dispenser_locations() + self.get_tomato_dispenser_locations() + self.get_serving_locations() + self.get_pot_locations() + self.get_chopping_board_locations() + self.get_meat_dispenser_locations() + self.get_sink_locations()
    
    def get_pot_states(self, state, pots_states_dict=None, valid_pos=None):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        if pots_states_dict is None:
            pots_states_dict = {}
            pots_states_dict['empty'] = []
            pots_states_dict['onion'] = defaultdict(list)
            pots_states_dict['tomato'] = defaultdict(list)
            pots_states_dict['steak'] = defaultdict(list)

        get_pot_info = []
        if valid_pos is not None:
            for pot_pos in self.get_pot_locations():
                if pot_pos in valid_pos:
                    get_pot_info.append(pot_pos)
        else:
            get_pot_info = self.get_pot_locations()

        for pot_pos in get_pot_info:
            # reset
            for k, v in pots_states_dict.items():
                if type(v) == list:
                    if pot_pos in v:
                        v.remove(pot_pos)
                else:
                    for d_k, d_v in v.items():
                        if pot_pos in d_v:
                            d_v.remove(pot_pos)

            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)

            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['{}_items'.format(num_items)].append(pot_pos)
                elif num_items == self.num_items_for_soup:
                    assert cook_time <= self.soup_cooking_time
                    if cook_time == self.soup_cooking_time:
                        pots_states_dict[soup_type]['ready'].append(pot_pos)
                    else:
                        pots_states_dict[soup_type]['cooking'].append(pot_pos)
                else:
                    raise ValueError("Pot with more than {} items".format(
                        self.num_items_for_soup))

                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['partially_full'].append(pot_pos)

        return pots_states_dict
    
    def get_ready_pots(self, pot_states):
        return pot_states["steak"]["ready"] + pot_states["onion"]["ready"]

    def get_cooking_pots(self, pot_states):
        return pot_states["steak"]["cooking"] + pot_states["onion"]["cooking"]
    
    def get_partially_full_pots(self, pot_states):
        return pot_states["steak"]["partially_full"] + pot_states["onion"][
            "partially_full"]
    
    def get_sink_status(self, state):
        empty_sink = []
        full_sink = []
        ready_sink = []
        sink_locations = self.get_sink_locations()
        for loc in sink_locations:
            if not state.has_object(loc): # board is empty
                empty_sink.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.state >= self.wash_time:
                    ready_sink.append(loc)
                else:
                    full_sink.append(loc)
        return {'empty': empty_sink, 'full': full_sink, 'ready': ready_sink}

    def get_chopping_board_status(self, state):
        empty_board = []
        full_board = []
        ready_board = []
        board_locations = self.get_chopping_board_locations()
        for loc in board_locations:
            if not state.has_object(loc): # board is empty
                empty_board.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.state >= self.chopping_time:
                    ready_board.append(loc)
                else:
                    full_board.append(loc)
        return {'empty': empty_board, 'full': full_board, 'ready': ready_board}

    def steak_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == 'steak', 'Object in pan was not steak'
        _, _, cook_time = obj.state
        return cook_time >= self.steak_cooking_time
    
    def plate_hot_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == 'hot_plate', 'Object in sink was not hot plate'
        prep_time = obj.state
        return prep_time >= self.wash_time
    
    def garnish_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == 'garnish', 'Object on chopping board was not garnish'
        prep_time = obj.state
        return prep_time >= self.chopping_time
    
    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in 'XOPDSTMWB'

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players +
                                           1)), "Some players were missing"

        assert all(c in 'XOPDSTWMB123456789 '
                   for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count(
            'D') >= 1, "'D' must be present at least once"
        assert all_elements.count(
            'S') >= 1, "'S' must be present at least once"
        assert all_elements.count(
            'P') >= 1, "'P' must be present at least once"
        assert all_elements.count('O') >= 1 or all_elements.count(
            'T') >= 1, "'O' or 'T' must be present at least once"
    
    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    def log_object_drop(self, events_infos, state, obj_name, pot_states,
                        player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        if obj_drop_key not in events_infos:
            # TODO: add support for tomato event logging
            if obj_name in ["meat", "hot_plate", "steak", "garnish"]:
                return
            raise ValueError("Unknown event {}".format(obj_drop_key))

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        # TODO
        return ''

    ###################
    # RENDER FUNCTION #
    ###################

    def render(self, state, mode, time_step_left=None, time_passed=None, view_angle=120, selected_action_count=0, info=None):
        """
        Function that renders the game

        Args:
            state(OvercookedState): state to render
            mode (string): mode of rendering
                           For "human", render the game without a info panel
                           For "blur", render the game with trajectory of both
                               players
                           For "full", render the game with a info panel
            time_step_left(int): timestep left for the game
        """
        players_dict = {player.position: player for player in state.players}
        objects_pos = []  # list of positions of the objects

        # set window size; SPRITE_LENGTH is the length of each squared sprite, which could be tuned in graphics.py
        if mode == "full":
            window_size = self.width*SPRITE_LENGTH,\
                          self.height*SPRITE_LENGTH + INFO_PANEL_HEIGHT
        elif mode == "right_panel" or mode == "right_panel_init":
            window_size = self.width*SPRITE_LENGTH + INFO_PANEL_WIDTH,\
                          self.height*SPRITE_LENGTH + INFO_PANEL_HEIGHT
        else:
            window_size = self.width*SPRITE_LENGTH, self.height*SPRITE_LENGTH

        if self.viewer == None:
            # create viewer
            self.viewer = pygame.display.set_mode(window_size)
            self.viewer.fill((255, 255, 255)) # white background
            # render the terrain
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain in enumerate(terrain_row):
                blit_terrain(x, y, self.terrain_mtx, self.viewer, mode)

        # remove the objects on the counters and pots
        if self.pre_objects_pos is not None:
            for pos in self.pre_objects_pos:
                x, y = pos
                blit_terrain(x, y, self.terrain_mtx, self.viewer, mode)

        # render objects at the new locations
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain in enumerate(terrain_row):
                curr_pos = get_curr_pos(x, y, mode)
                # there is object on a counter
                if terrain == "X" and state.has_object((x, y)):
                    state_obj = state.get_object((x, y))
                    obj_pgobj = get_object_sprite(state_obj)
                    self.viewer.blit(obj_pgobj, curr_pos)
                    objects_pos.append((x, y))
                # there is soup on a pot
                elif terrain == "P" and state.has_object((x, y)):
                    soup_obj = state.get_object((x, y))
                    soup_type, num_items, cook_time = soup_obj.state
                    # if soup is ready
                    if self.soup_ready_at_location(state, (x, y)):
                        soup_pgobj = load_image(
                            os.path.join(ASSETS_DIR, OBJECT_DIR,
                                         'soup-%s-cooked.png' % soup_type))
                    # if soup is not ready
                    else:
                        soup_pgobj = get_object_sprite(soup_obj, on_pot=True)
                    
                    self.viewer.blit(soup_pgobj, curr_pos)
                    objects_pos.append((x, y))

                    # render cook time for current soup if cooking starts
                    if num_items == self.num_items_for_soup:
                        cook_time_text_surface = get_text_sprite(
                            str(cook_time))
                        text_pos = cook_time_text_surface.get_rect()
                        # align midbottom of the textbox to midbottom of
                        # current terrain
                        text_pos.midbottom = curr_pos.midbottom
                        # if not self.check_viewpoint(state.players[1].position, state.players[1].orientation, text_pos.midbottom[0], text_pos.midbottom[1]):
                        #     cook_time_text_surface.set_alpha(1)
                        self.viewer.blit(cook_time_text_surface, text_pos)

                elif terrain == "B" and state.has_object((x, y)):
                    chop_obj = state.get_object((x, y))
                    chop_time = chop_obj.state
                    # if soup is ready
                    if self.garnish_ready_at_location(state, (x, y)):
                        board_pgobj = load_image(
                            os.path.join(ASSETS_DIR, OBJECT_DIR,
                                         'chopped.png'))
                    # if soup is not ready
                    else:
                        board_pgobj = get_object_sprite(chop_obj, on="board")
                    
                    self.viewer.blit(board_pgobj, curr_pos)
                    objects_pos.append((x, y))

                    # render cook time for current soup if cooking starts
                    cook_time_text_surface = get_text_sprite(
                        str(chop_time))
                    text_pos = cook_time_text_surface.get_rect()
                    # align midbottom of the textbox to midbottom of
                    # current terrain
                    text_pos.midbottom = curr_pos.midbottom
                    # if not self.check_viewpoint(state.players[1].position, state.players[1].orientation, text_pos.midbottom[0], text_pos.midbottom[1]):
                    #     cook_time_text_surface.set_alpha(1)
                    self.viewer.blit(cook_time_text_surface, text_pos)

                elif terrain == "W" and state.has_object((x, y)):
                    sink_obj = state.get_object((x, y))
                    wash_time = sink_obj.state
                    sink_pgobj = get_object_sprite(sink_obj, on="sink")
                    
                    self.viewer.blit(sink_pgobj, curr_pos)
                    objects_pos.append((x, y))

                    # render cook time for current soup if cooking starts
                    cook_time_text_surface = get_text_sprite(
                        str(wash_time))
                    text_pos = cook_time_text_surface.get_rect()
                    # align midbottom of the textbox to midbottom of
                    # current terrain
                    text_pos.midbottom = curr_pos.midbottom
                    # if not self.check_viewpoint(state.players[1].position, state.players[1].orientation, text_pos.midbottom[0], text_pos.midbottom[1]):
                    #     cook_time_text_surface.set_alpha(1)
                    self.viewer.blit(cook_time_text_surface, text_pos)

        if mode == "human" or mode == "full" or mode == "fog" or mode == "right_panel" or mode == "right_panel_init" or mode=="not_aware":
            # remove chefs from last state
            if self.pre_players_pos is not None:
                for pos in self.pre_players_pos:
                    x, y = pos
                    blit_terrain(x, y, self.terrain_mtx, self.viewer, mode)

        if mode == "not_aware":
            player_hat_color = {
                0: 'purplehat',
                1: 'bluehat'
            }
        else:
            player_hat_color = {
                0: 'greenhat',
                1: 'bluehat'
            }
        # render the chefs at new location
        for pos, player in players_dict.items():
            x, y = pos
            curr_pos = get_curr_pos(x, y, mode)

            # check player position conflicts
            player_idx_lst = [
                i for i, p in enumerate(state.players)
                if p.position == player.position
            ]
            assert len(player_idx_lst) == 1

            player_pgobj, player_hat_pgobj = get_player_sprite(
                player, player_idx_lst[0], hat_color=player_hat_color)

            if mode == "blur":
                transparent = (time_step_left+10)*5
                print(transparent)
                player_pgobj.set_alpha(transparent)
                player_hat_pgobj.set_alpha(transparent)
                # draw_arrow(self.viewer, player, player_idx_lst[0], curr_pos,
                           # time_step_left)

            self.viewer.blit(player_pgobj, curr_pos)
            self.viewer.blit(player_hat_pgobj, curr_pos)                

        # update previous players and objects positions
        self.pre_players_pos = players_dict.keys()
        if len(objects_pos) > 0:
            self.pre_objects_pos = objects_pos
        else:
            self.pre_objects_pos = None

        # render the game info panel
        if mode == "full":
            render_game_info_panel(
                self.viewer,
                (self.width * SPRITE_LENGTH, self.height * SPRITE_LENGTH),
                state.num_orders_remaining, time_passed)
            
        if mode == "right_panel_init":
            self.rend_boxes = render_game_info_panel(
                self.viewer,
                (self.width * SPRITE_LENGTH, self.height * SPRITE_LENGTH),
                state.num_orders_remaining, time_passed, selected_action_count, init=True)
            
        elif mode == "right_panel" or mode=="not_aware":
            render_game_info_panel(
                self.viewer,
                (self.width * SPRITE_LENGTH, self.height * SPRITE_LENGTH),
                state.num_orders_remaining, time_passed, selected_action_count, curr_s = info)
            for box in self.rend_boxes:
                box.render_checkbox()
    

        if (mode == "fog" or mode == "right_panel" or mode=="not_aware") and (view_angle != 0): # fog the terrain
            for y, terrain_row in enumerate(self.terrain_mtx):
                for x, terrain in enumerate(terrain_row):
                    in_view = self.check_viewpoint(state.players[1].position, state.players[1].orientation, x, y, view_angle=view_angle)
                    if not in_view:
                        curr_pos = get_curr_pos(x, y, mode)
                        blit_terrain(x, y, self.terrain_mtx, self.viewer, mode, in_view)
                        fog_pgobj = pygame.Surface((50,50))
                        fog_pgobj.fill((0,0,0))
                        # load_image(os.path.join(ASSETS_DIR, TERRAIN_DIR, 'counter.png'))
                        fog_pgobj.set_alpha(100)
                        self.viewer.blit(fog_pgobj, curr_pos)

            # render_game_info_panel(
            #     self.viewer,
            #     (self.width * SPRITE_LENGTH, self.height * SPRITE_LENGTH),
            #     state.num_orders_remaining, time_passed)

        # update display
        pygame.display.update()

    ###################
    # STATE ENCODINGS #
    ###################
    #
    # TODO

    ##############
    # DEPRECATED #
    ##############
    #
    # TODO
    