import itertools, os
from turtle import position
import numpy as np
import pickle, time, random, copy, json
import math
from overcooked_ai_py.utils import pos_distance, manhattan_distance
from overcooked_ai_py.planning.search import SearchTree, Graph
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, ObjectState, OvercookedGridworld, EVENT_TYPES, SIMPLE_EVENT_TYPES
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.data.planners import load_saved_action_manager, PLANNERS_DIR
import pprint
import pandas as pd

# from deepdiff import DeepDiff

# to measure exec time 
from timeit import default_timer as timer

# Run planning logic with additional checks and
# computation to prevent or identify possible minor errors
SAFE_RUN = False
LOGUNIT = 500
TRAINNINGUNIT = 5000
SOFTMAX_T = 10 # higher cause the actions to be nearly equiprobable
MAX_NUM_STATES = 19000

NO_COUNTERS_PARAMS = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

NO_COUNTERS_START_OR_PARAMS = {
    'start_orientations': True,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

class MotionPlanner(object):
    """A planner that computes optimal plans for a single agent to 
    arrive at goal positions and orientations in an OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
        counter_goals (list): list of positions of counters we will consider
                              as valid motion goals
    """

    def __init__(self, mdp, counter_goals=[]):
        self.mdp = mdp

        # If positions facing counters should be 
        # allowed as motion goals
        self.counter_goals = counter_goals

        # Graph problem that solves shortest path problem
        # between any position & orientation start-goal pair
        self.graph_problem = self._graph_from_grid()
        self.motion_goals_for_pos = self._get_goal_dict()

    def get_plan(self, start_pos_and_or, goal_pos_and_or):
        """
        Returns pre-computed plan from initial agent position
        and orientation to a goal position and orientation.
        
        Args:
            start_pos_and_or (tuple): starting (pos, or) tuple
            goal_pos_and_or (tuple): goal (pos, or) tuple
        """
        action_plan, pos_and_or_path, plan_cost = self._compute_plan(start_pos_and_or, goal_pos_and_or)
        return action_plan, pos_and_or_path, plan_cost
    
    def get_gridworld_distance(self, start_pos_and_or, goal_pos_and_or):
        """Number of actions necessary to go from starting position
        and orientations to goal position and orientation (not including
        interaction action)"""
        assert self.is_valid_motion_start_goal_pair(start_pos_and_or, goal_pos_and_or), \
            "Goal position and orientation were not a valid motion goal"
        # Removing interaction cost
        return self.graph_problem.dist(start_pos_and_or, goal_pos_and_or) - 1

    def get_gridworld_pos_distance(self, pos1, pos2):
        """Minimum (over possible orientations) number of actions necessary 
        to go from starting position to goal position (not including 
        interaction action)."""
        # NOTE: currently unused, pretty bad code. If used in future, clean up
        min_cost = np.Inf
        for d1, d2 in itertools.product(Direction.ALL_DIRECTIONS, repeat=2):
            start = (pos1, d1)
            end = (pos2, d2)
            if self.is_valid_motion_start_goal_pair(start, end):
                plan_cost = self.get_gridworld_distance(start, end)
                if plan_cost < min_cost:
                    min_cost = plan_cost
        return min_cost

    def is_valid_motion_start_goal_pair(self, start_pos_and_or, goal_pos_and_or, debug=False):
        if not self.is_valid_motion_goal(goal_pos_and_or):
            return False
        if not self.positions_are_connected(start_pos_and_or, goal_pos_and_or):
            return False
        return True

    def is_valid_motion_goal(self, goal_pos_and_or):
        """Checks that desired single-agent goal state (position and orientation) 
        is reachable and is facing a terrain feature"""
        goal_position, goal_orientation = goal_pos_and_or
        if goal_position not in self.mdp.get_valid_player_positions():
            return False

        ## temp commit since actions should not be limited to only sub-goals that complete a task, but should include actions such as wait and put down item and switch sub-goals, which do not always face a terrain with features.
        # # Restricting goals to be facing a terrain feature
        # pos_of_facing_terrain = Action.move_in_direction(goal_position, goal_orientation)
        # facing_terrain_type = self.mdp.get_terrain_type_at_pos(pos_of_facing_terrain)
        # if facing_terrain_type == ' ' or (facing_terrain_type == 'X' and pos_of_facing_terrain not in self.counter_goals):
        #     return False
        return True

    def _compute_plan(self, start_motion_state, goal_motion_state):
        """Computes optimal action plan for single agent movement
        
        Args:
            start_motion_state (tuple): starting positions and orientations
            positions_plan (list): positions path followed by agent
            goal_motion_state (tuple): goal positions and orientations
        """
        assert self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state)
        positions_plan = self._get_position_plan_from_graph(start_motion_state, goal_motion_state)
        action_plan, pos_and_or_path, plan_length = self.action_plan_from_positions(positions_plan, start_motion_state, goal_motion_state)
        return action_plan, pos_and_or_path, plan_length

    def positions_are_connected(self, start_pos_and_or, goal_pos_and_or):
        return self.graph_problem.are_in_same_cc(start_pos_and_or, goal_pos_and_or)

    def _get_position_plan_from_graph(self, start_node, end_node):
        """Recovers positions to be reached by agent after the start node to reach the end node"""
        node_path = self.graph_problem.get_node_path(start_node, end_node)
        assert node_path[0] == start_node and node_path[-1] == end_node
        positions_plan = [state_node[0] for state_node in node_path[1:]]
        return positions_plan

    def action_plan_from_positions(self, position_list, start_motion_state, goal_motion_state):
        """
        Recovers an action plan reaches the goal motion position and orientation, and executes
        and interact action.
        
        Args:
            position_list (list): list of positions to be reached after the starting position
                                  (does not include starting position, but includes ending position)
            start_motion_state (tuple): starting position and orientation
            goal_motion_state (tuple): goal position and orientation

        Returns:
            action_plan (list): list of actions to reach goal state
            pos_and_or_path (list): list of (pos, or) pairs visited during plan execution
                                    (not including start, but including goal)
        """
        goal_position, goal_orientation = goal_motion_state
        action_plan, pos_and_or_path = [], []
        position_to_go = list(position_list)
        curr_pos, curr_or = start_motion_state

        # Get agent to goal position
        while position_to_go and curr_pos != goal_position:
            next_pos = position_to_go.pop(0)
            action = Action.determine_action_for_change_in_pos(curr_pos, next_pos)
            action_plan.append(action)
            curr_or = action if action != Action.STAY else curr_or
            pos_and_or_path.append((next_pos, curr_or))
            curr_pos = next_pos
        
        # Fix agent orientation if necessary
        if curr_or != goal_orientation:
            new_pos, _ = self.mdp._move_if_direction(curr_pos, curr_or, goal_orientation)
            # print(curr_pos, curr_or, goal_position, goal_orientation)
            # assert new_pos == goal_position
            action_plan.append(goal_orientation)
            pos_and_or_path.append((goal_position, goal_orientation))

        # Add interact action
        action_plan.append(Action.INTERACT)
        pos_and_or_path.append((goal_position, goal_orientation))

        return action_plan, pos_and_or_path, len(action_plan)

    def _graph_from_grid(self):
        """Creates a graph adjacency matrix from an Overcooked MDP class."""
        state_decoder = {}
        for state_index, motion_state in enumerate(self.mdp.get_valid_player_positions_and_orientations()):
            state_decoder[state_index] = motion_state

        pos_encoder = {motion_state:state_index for state_index, motion_state in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for state_index, start_motion_state in state_decoder.items():
            for action, successor_motion_state in self._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                adjacency_matrix[state_index][adj_pos_index] = self._graph_action_cost(action)

        return Graph(adjacency_matrix, pos_encoder, state_decoder)

    def _graph_action_cost(self, action):
        """Returns cost of a single-agent action"""
        assert action in Action.ALL_ACTIONS
        
        # # penalize plans that have the action stay to avoid stopping and waiting in joint plans
        # if action == Action.STAY:
        #     return 2

        return 1

    def _get_valid_successor_motion_states(self, start_motion_state):
        """Get valid motion states one action away from the starting motion state."""
        start_position, start_orientation = start_motion_state
        return [(action, self.mdp._move_if_direction(start_position, start_orientation, action)) for action in Action.ALL_ACTIONS]

    def min_cost_between_features(self, pos_list1, pos_list2, manhattan_if_fail=False):
        """
        Determines the minimum number of timesteps necessary for a player to go from any 
        terrain feature in list1 to any feature in list2 and perform an interact action
        """
        min_dist = np.Inf
        min_manhattan = np.Inf
        for pos1, pos2 in itertools.product(pos_list1, pos_list2):
            for mg1, mg2 in itertools.product(self.motion_goals_for_pos[pos1], self.motion_goals_for_pos[pos2]):
                if not self.is_valid_motion_start_goal_pair(mg1, mg2):
                    if manhattan_if_fail:
                        pos0, pos1 = mg1[0], mg2[0]
                        curr_man_dist = manhattan_distance(pos0, pos1)
                        if curr_man_dist < min_manhattan:
                            min_manhattan = curr_man_dist
                    continue
                curr_dist = self.get_gridworld_distance(mg1, mg2)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                
        # +1 to account for interaction action
        if manhattan_if_fail and min_dist == np.Inf:
            min_dist = min_manhattan
        min_cost = min_dist + 1
        return min_cost

    def min_cost_to_feature(self, start_pos_and_or, feature_pos_list, with_argmin=False, with_motion_goal=False, debug=False):
        """
        Determines the minimum number of timesteps necessary for a player to go from the starting
        position and orientation to any feature in feature_pos_list and perform an interact action
        """
        start_pos = start_pos_and_or[0]
        assert self.mdp.get_terrain_type_at_pos(start_pos) != 'X'
        min_dist = np.Inf
        best_feature = None
        best_motion_goal = None
        for feature_pos in feature_pos_list:
            for feature_goal in self.motion_goals_for_pos[feature_pos]:
                if not self.is_valid_motion_start_goal_pair(start_pos_and_or, feature_goal, debug=debug):
                    print('{} and {} is not a valid motion start goal pair'.format(start_pos_and_or, feature_goal))
                    continue
                curr_dist = self.get_gridworld_distance(start_pos_and_or, feature_goal)
                if curr_dist < min_dist:
                    best_feature = feature_pos
                    best_motion_goal = feature_goal
                    min_dist = curr_dist
        # +1 to account for interaction action
        min_cost = min_dist + 1
        if with_motion_goal:
            return min_cost, best_motion_goal
        if with_argmin:
            # assert best_feature is not None, "{} vs {}".format(start_pos_and_or, feature_pos_list)
            return min_cost, best_feature
        return min_cost

    def _get_goal_dict(self):
        """Creates a dictionary of all possible goal states for all possible
        terrain features that the agent might want to interact with."""
        terrain_feature_locations = []
        for terrain_type, pos_list in self.mdp.terrain_pos_dict.items():
            if terrain_type != ' ':
                terrain_feature_locations += pos_list
        return {feature_pos:self._get_possible_motion_goals_for_feature(feature_pos) for feature_pos in terrain_feature_locations}

    def _get_possible_motion_goals_for_feature(self, goal_pos):
        """Returns a list of possible goal positions (and orientations)
        that could be used for motion planning to get to goal_pos"""
        goals = []
        valid_positions = self.mdp.get_valid_player_positions()
        for d in Direction.ALL_DIRECTIONS:
            adjacent_pos = Action.move_in_direction(goal_pos, d)
            if adjacent_pos in valid_positions:
                goal_orientation = Direction.OPPOSITE_DIRECTIONS[d]
                motion_goal = (adjacent_pos, goal_orientation)
                goals.append(motion_goal)
        return goals


class JointMotionPlanner(object):
    """A planner that computes optimal plans for a two agents to 
    arrive at goal positions and orientations in a OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
    """

    def __init__(self, mdp, params, debug=False):
        self.mdp = mdp

        # Whether starting orientations should be accounted for
        # when solving all motion problems 
        # (increases number of plans by a factor of 4)
        # but removes additional fudge factor <= 1 for each
        # joint motion plan
        self.start_orientations = params["start_orientations"]

        # Enable both agents to have the same motion goal
        self.same_motion_goals = params["same_motion_goals"]
        
        # Single agent motion planner
        self.motion_planner = MotionPlanner(mdp, counter_goals=params["counter_goals"])

        # Graph problem that returns optimal paths from 
        # starting positions to goal positions (without
        # accounting for orientations)
        self.joint_graph_problem = self._joint_graph_from_grid()

    def get_low_level_action_plan(self, start_jm_state, goal_jm_state, merge_one=False):
        """
        Returns pre-computed plan from initial joint motion state
        to a goal joint motion state.
        
        Args:
            start_jm_state (tuple): starting pos & orients ((pos1, or1), (pos2, or2))
            goal_jm_state (tuple): goal pos & orients ((pos1, or1), (pos2, or2))
        
        Returns:
            joint_action_plan (list): joint actions to be executed to reach end_jm_state
            end_jm_state (tuple): the pair of (pos, or) tuples corresponding
                to the ending timestep (this will usually be different from 
                goal_jm_state, as one agent will end before other).
            plan_lengths (tuple): lengths for each agent's plan
        """
        assert self.is_valid_joint_motion_pair(start_jm_state, goal_jm_state), \
            "start: {} \t end: {} was not a valid motion goal pair".format(start_jm_state, goal_jm_state)
        
        if self.start_orientations:
            plan_key = (start_jm_state, goal_jm_state)
        else:
            starting_positions = tuple(player_pos_and_or[0] for player_pos_and_or in start_jm_state)
            goal_positions = tuple(player_pos_and_or[0] for player_pos_and_or in goal_jm_state)
            # If beginning positions are equal to end positions, the pre-stored
            # plan (not dependent on initial positions) will likely return a 
            # wrong answer, so we compute it from scratch.
            # 
            # This is because we only compute plans with starting orientations
            # (North, North), so if one of the two agents starts at location X
            # with orientation East it's goal is to get to location X with
            # orientation North. The precomputed plan will just tell that agent
            # that it is already at the goal, so no actions (or just 'interact')
            # are necessary.
            #
            # We also compute the plan for any shared motion goal with SAFE_RUN,
            # as there are some minor edge cases that could not be accounted for
            # but I expect should not make a difference in nearly all scenarios
            if any([s == g for s, g in zip(starting_positions, goal_positions)]) or (SAFE_RUN and goal_positions[0] == goal_positions[1]):
                return self._obtain_plan(start_jm_state, goal_jm_state)

            dummy_orientation = Direction.NORTH
            dummy_start_jm_state = tuple((pos, dummy_orientation) for pos in starting_positions)
            plan_key = (dummy_start_jm_state, goal_jm_state)

        joint_action_plan, end_jm_state, plan_lengths = self._obtain_plan(plan_key[0], plan_key[1], merge_one=merge_one)
        return joint_action_plan, end_jm_state, plan_lengths

    def is_valid_jm_start_goal_pair(self, joint_start_state, joint_goal_state):
        """Checks if the combination of joint start state and joint goal state is valid"""
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        check_valid_fn = self.motion_planner.is_valid_motion_start_goal_pair
        return all([check_valid_fn(joint_start_state[i], joint_goal_state[i]) for i in range(2)])

    def _obtain_plan(self, joint_start_state, joint_goal_state, merge_one=False):
        """Either use motion planner or actually compute a joint plan"""
        # Try using MotionPlanner plans and join them together
        action_plans, pos_and_or_paths, plan_lengths = self._get_plans_from_single_planner(joint_start_state, joint_goal_state)
        # Check if individual plans conflict
        # print('action_plans =', action_plans)
        have_conflict = self.plans_have_conflict(joint_start_state, joint_goal_state, pos_and_or_paths, plan_lengths)
        # If there is no conflict, the joint plan computed by joining single agent MotionPlanner plans is optimal
        # print('have_conflict =', have_conflict)
        if not have_conflict:
            joint_action_plan, end_pos_and_orientations = self._join_single_agent_action_plans(
                joint_start_state, action_plans, pos_and_or_paths, min(plan_lengths)
            )
            return joint_action_plan, end_pos_and_orientations, plan_lengths
        
        # If there is a conflict in the single motion plan and the agents have the same goal,
        # the graph problem can't be used either as it can't handle same goal state: we compute
        # manually what the best way to handle the conflict is
        elif self._agents_are_in_same_position(joint_goal_state):
            joint_action_plan, end_pos_and_orientations, plan_lengths = self._handle_path_conflict_with_same_goal(
                joint_start_state, joint_goal_state, action_plans, pos_and_or_paths
            )
            return joint_action_plan, end_pos_and_orientations, plan_lengths

        # If there is a conflict, and the agents have different goals, we can use solve the joint graph problem
        # return self._compute_plan_from_joint_graph(joint_start_state, joint_goal_state)

        try:
            return self._get_joint_plan_from_merging_ind_paths(
                pos_and_or_paths, joint_start_state, joint_goal_state, merge_one=merge_one
            )
        except ValueError:
            return self._compute_plan_from_joint_graph(
                joint_start_state, joint_goal_state
            )

    def merge_paths_dp(self, pos_and_or_paths, joint_start_state):
        """
        DP solver that merges two paths such that they do not have conflicts.
        Note that this solver can only deal with paths that does not share
        the same start point and end point.

        Args:
            pos_and_or_paths (list): list of tuple(position, orientation)

        Returns:
            position_list1 (list), position_list2 (list)
        """

        s1, s2 = self.extract_ind_pos_list(pos_and_or_paths, joint_start_state)

        if s1[-1] == s2[-1] or s1[0] == s2[0]:
            return None, None
        oo = np.inf
        table = np.full((len(s1)+1, len(s2)+1), oo)
        table[0, 0] = 0
        choice = np.full((len(s1)+1, len(s2)+1), -1)
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    table[i][j] = oo
                    continue
                ncost = table[i, j]+(1 if j >= i else 0)
                if ncost < table[i, j+1]:
                    table[i,j+1] = ncost
                    choice[i,j+1] = 0
                ncost = table[i, j]+(1 if i >= j else 0)
                if ncost < table[i+1,j]:
                    table[i+1,j] = ncost
                    choice[i+1,j] = 1
                ncost = table[i,j]
                if ncost < table[i+1,j+1]:
                    table[i+1,j+1] = ncost
                    choice[i+1,j+1] = 2
        # Use the choice matrix to build back the path
        i = len(s1)-1
        j = len(s2)-1
        ans1 = []
        ans2 = []
        while 0 < i or 0 < j:
            ans1.append(s1[i])
            ans2.append(s2[j])
            if choice[i,j] == 0:
                j -= 1
            elif choice[i,j] == 1:
                i -= 1
            elif choice[i,j] == 2:
                i -= 1
                j -= 1
            else:
                raise ValueError("Static agent blocking the way: No solution!")
        ans1.append(s1[0])
        ans2.append(s2[0])
        ans1.reverse()
        ans2.reverse()

        # paths are invalid if they crash into each other
        for idx in range(min(len(ans1), len(ans2)) - 1):
            if ans1[idx] == ans2[idx+1] and ans1[idx+1] == ans2[idx]:
                raise ValueError("Two paths crached: Solution not valid!")

        return ans1[1:], ans2[1:]

    def merge_one_path_into_other_dp(self, pos_and_or_paths, joint_start_state):
        """
        DP solver that merges one path to another by only changing one 
        path's pos and or such that they do not have conflicts.
        Note that this solver can only deal with paths that does not share
        the same start point and end point.

        Args:
            pos_and_or_paths (list): list of tuple(position, orientation)

        Returns:
            position_list1 (list), position_list2 (list)
        """
        s1, s2 = self.extract_ind_pos_list(pos_and_or_paths, joint_start_state)
        if s1[-1] == s2[-1] or s1[0] == s2[0]:
            return None, None
        oo = np.inf
        table = np.full((len(s1)+1, len(s2)+1), oo)
        table[0, 0] = 0
        choice = np.full((len(s1)+1, len(s2)+1), -1)
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    table[i][j] = oo
                    continue
                ncost = table[i, j]+(1 if j >= i else 0)
                if ncost < table[i, j+1]:
                    table[i,j+1] = ncost
                    choice[i,j+1] = 0
                ncost = table[i, j]+(1 if i >= j else 0)
                if ncost < table[i+1,j]:
                    table[i+1,j] = ncost
                    choice[i+1,j] = 1
                ncost = table[i,j]
                if ncost < table[i+1,j+1]:
                    table[i+1,j+1] = ncost
                    choice[i+1,j+1] = 2
        # Use the choice matrix to build back the path
        i = len(s1)-1
        j = len(s2)-1
        ans1 = []
        ans2 = []
        while 0 < i or 0 < j:
            ans1.append(s1[i])
            ans2.append(s2[j])
            if choice[i,j] == 0:
                j -= 1
            elif choice[i,j] == 1:
                i -= 1
            elif choice[i,j] == 2:
                i -= 1
                j -= 1
            else:
                raise ValueError("Static agent blocking the way: No solution!")
        ans1.append(s1[0])
        ans2.append(s2[0])
        ans1.reverse()
        ans2.reverse()

        # paths are invalid if they crash into each other
        for idx in range(min(len(ans1), len(ans2)) - 1):
            if ans1[idx] == ans2[idx+1] and ans1[idx+1] == ans2[idx]:
                raise ValueError("Two paths crached: Solution not valid!")
        return ans1[1:], ans2[1:]

    def extract_ind_pos_list(self, pos_and_or_paths, joint_start_state):
        pos_and_or_path1, pos_and_or_path2 = pos_and_or_paths
        pos_list1 = [row[0] for row in pos_and_or_path1]
        pos_list2 = [row[0] for row in pos_and_or_path2]
        start1, start2 = joint_start_state
        pos_list1.insert(0, start1[0])
        pos_list2.insert(0, start2[0])
        return pos_list1, pos_list2

    def _get_joint_plan_from_merging_ind_paths(self, pos_and_or_paths, joint_start_state, joint_goal_state, merge_one=False):
        """
        Get joint motion plan by using the DP solver to resolve conflicts
        in the individual motion paths

        Args:
            pos_and_or_path (list): list of (pos, or) pairs visited during
                                    plan execution
                                    (not including start, but including goal)
            joint_start_state (list(tuple)): list of starting position and
                                             orientation
            joint_goal_state (list(tuple)): list of goal position and
                                            orientation
        """
        # resolve conflict in the individual paths
        if merge_one: path_lists = self.merge_one_path_into_other_dp(pos_and_or_paths, joint_start_state)
        else: path_lists = self.merge_paths_dp(pos_and_or_paths, joint_start_state)

        # obtain action_plans from paths
        action_plans, pos_and_or_paths, plan_lengths = [], [], []
        for path_list, start, goal in zip(path_lists, joint_start_state, joint_goal_state):
            action_plan, pos_and_or_path, plan_length = \
                self.motion_planner.action_plan_from_positions(
                    path_list, start, goal
                )
            action_plans.append(action_plan)
            pos_and_or_paths.append(pos_and_or_path)
            plan_lengths.append(plan_length)

        # joint the action plans
        joint_action_plan, end_pos_and_orientations = \
            self._join_single_agent_action_plans(
                joint_start_state,
                action_plans,
                pos_and_or_paths,
                 min(plan_lengths)
            )
        return joint_action_plan, end_pos_and_orientations, plan_lengths

    def _get_plans_from_single_planner(self, joint_start_state, joint_goal_state):
        """
        Get individual action plans for each agent from the MotionPlanner to get each agent
        independently to their goal state. NOTE: these plans might conflict
        """
        single_agent_motion_plans = [self.motion_planner.get_plan(start, goal) for start, goal in zip(joint_start_state, joint_goal_state)]
        action_plans, pos_and_or_paths = [], []
        for action_plan, pos_and_or_path, _ in single_agent_motion_plans:
            action_plans.append(action_plan)
            pos_and_or_paths.append(pos_and_or_path)
        plan_lengths = tuple(len(p) for p in action_plans)
        assert all([plan_lengths[i] == len(pos_and_or_paths[i]) for i in range(2)])
        return action_plans, pos_and_or_paths, plan_lengths

    def plans_have_conflict(self, joint_start_state, joint_goal_state, pos_and_or_paths, plan_lengths):
        """Check if the sequence of pos_and_or_paths for the two agents conflict"""
        min_length = min(plan_lengths)
        prev_positions = tuple(s[0] for s in joint_start_state)
        for t in range(min_length):
            curr_pos_or0, curr_pos_or1 = pos_and_or_paths[0][t], pos_and_or_paths[1][t]
            curr_positions = (curr_pos_or0[0], curr_pos_or1[0])
            if self.mdp.is_transition_collision(prev_positions, curr_positions):
                return True
            prev_positions = curr_positions
        return False

    def _join_single_agent_action_plans(self, joint_start_state, action_plans, pos_and_or_paths, finishing_time):
        """Returns the joint action plan and end joint state obtained by joining the individual action plans"""
        assert finishing_time > 0
        end_joint_state = (pos_and_or_paths[0][finishing_time - 1], pos_and_or_paths[1][finishing_time - 1])
        joint_action_plan = list(zip(*[action_plans[0][:finishing_time], action_plans[1][:finishing_time]]))
        return joint_action_plan, end_joint_state

    def _handle_path_conflict_with_same_goal(self, joint_start_state, joint_goal_state, action_plans, pos_and_or_paths):
        """Assumes that optimal path in case two agents have the same goal and their paths conflict 
        is for one of the agents to wait. Checks resulting plans if either agent waits, and selects the 
        shortest cost among the two."""

        joint_plan0, end_pos_and_or0, plan_lengths0 = self._handle_conflict_with_same_goal_idx(
            joint_start_state, joint_goal_state, action_plans, pos_and_or_paths, wait_agent_idx=0
        )

        joint_plan1, end_pos_and_or1, plan_lengths1 = self._handle_conflict_with_same_goal_idx(
            joint_start_state, joint_goal_state, action_plans, pos_and_or_paths, wait_agent_idx=1
        )

        assert any([joint_plan0 is not None, joint_plan1 is not None])

        best_plan_idx = np.argmin([min(plan_lengths0), min(plan_lengths1)])
        solutions = [(joint_plan0, end_pos_and_or0, plan_lengths0), (joint_plan1, end_pos_and_or1, plan_lengths1)]
        return solutions[best_plan_idx]

    def _handle_conflict_with_same_goal_idx(self, joint_start_state, joint_goal_state, action_plans, pos_and_or_paths, wait_agent_idx):
        """
        Determines what is the best joint plan if whenether there is a conflict between the two agents' actions,
        the agent with index `wait_agent_idx` waits one turn.
        
        If the agent that is assigned to wait is "in front" of the non-waiting agent, this could result
        in an endless conflict. In this case, we return infinite finishing times.
        """
        idx0, idx1 = 0, 0
        prev_positions = [start_pos_and_or[0] for start_pos_and_or in joint_start_state]
        curr_pos_or0, curr_pos_or1 = joint_start_state

        agent0_plan_original, agent1_plan_original = action_plans

        joint_plan = []
        # While either agent hasn't finished their plan
        while idx0 != len(agent0_plan_original) and idx1 != len(agent1_plan_original): 
            next_pos_or0, next_pos_or1 = pos_and_or_paths[0][idx0], pos_and_or_paths[1][idx1]
            next_positions = (next_pos_or0[0], next_pos_or1[0])

            # If agents collide, let the waiting agent wait and the non-waiting
            # agent take a step
            if self.mdp.is_transition_collision(prev_positions, next_positions):
                if wait_agent_idx == 0:
                    curr_pos_or0 = curr_pos_or0 # Agent 0 will wait, stays the same
                    curr_pos_or1 = next_pos_or1
                    curr_joint_action = [Action.STAY, agent1_plan_original[idx1]]
                    idx1 += 1
                elif wait_agent_idx == 1:
                    curr_pos_or0 = next_pos_or0
                    curr_pos_or1 = curr_pos_or1 # Agent 1 will wait, stays the same
                    curr_joint_action = [agent0_plan_original[idx0], Action.STAY]
                    idx0 += 1

                curr_positions = (curr_pos_or0[0], curr_pos_or1[0])
                
                # If one agent waiting causes other to crash into it, return None
                if self._agents_are_in_same_position((curr_pos_or0, curr_pos_or1)):
                    return None, None, [np.Inf, np.Inf]
                
            else:
                curr_pos_or0, curr_pos_or1 = next_pos_or0, next_pos_or1
                curr_positions = next_positions
                curr_joint_action = [agent0_plan_original[idx0], agent1_plan_original[idx1]]
                idx0 += 1
                idx1 += 1
                
            joint_plan.append(curr_joint_action)
            prev_positions = curr_positions
        
        assert idx0 != idx1, "No conflict found"

        end_pos_and_or = (curr_pos_or0, curr_pos_or1)
        finishing_times = (np.Inf, idx1) if wait_agent_idx == 0 else (idx0, np.Inf)
        return joint_plan, end_pos_and_or, finishing_times

    def is_valid_joint_motion_goal(self, joint_goal_state):
        """Checks whether the goal joint positions and orientations are a valid goal"""
        if not self.same_motion_goals and self._agents_are_in_same_position(joint_goal_state):
            return False
        multi_cc_map = len(self.motion_planner.graph_problem.connected_components) > 1
        players_in_same_cc = self.motion_planner.graph_problem.are_in_same_cc(joint_goal_state[0], joint_goal_state[1])
        if multi_cc_map and players_in_same_cc:
            return False
        return all([self.motion_planner.is_valid_motion_goal(player_state) for player_state in joint_goal_state])

    def is_valid_joint_motion_pair(self, joint_start_state, joint_goal_state):
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        return all([ \
            self.motion_planner.is_valid_motion_start_goal_pair(joint_start_state[i], joint_goal_state[i]) for i in range(2)])

    def _agents_are_in_same_position(self, joint_motion_state):
        agent_positions = [player_pos_and_or[0] for player_pos_and_or in joint_motion_state]
        return len(agent_positions) != len(set(agent_positions))

    def _compute_plan_from_joint_graph(self, joint_start_state, joint_goal_state):
        """Compute joint action plan for two agents to achieve a 
        certain position and orientation with the joint motion graph
        
        Args:
            joint_start_state: pair of start (pos, or)
            goal_statuses: pair of goal (pos, or)
        """
        assert self.is_valid_joint_motion_pair(joint_start_state, joint_goal_state), joint_goal_state
        # Solve shortest-path graph problem
        start_positions = list(zip(*joint_start_state))[0]
        goal_positions = list(zip(*joint_goal_state))[0]
        joint_positions_node_path = self.joint_graph_problem.get_node_path(start_positions, goal_positions)[1:]
        # print('joint_positions_node_path =', joint_positions_node_path)
        joint_actions_list, end_pos_and_orientations, finishing_times = self.joint_action_plan_from_positions(joint_positions_node_path, joint_start_state, joint_goal_state)
        return joint_actions_list, end_pos_and_orientations, finishing_times

    def joint_action_plan_from_positions(self, joint_positions, joint_start_state, joint_goal_state):
        """
        Finds an action plan and it's cost, such that at least one of the agent goal states is achieved

        Args:
            joint_positions (list): list of joint positions to be reached after the starting position
                                    (does not include starting position, but includes ending position)
            joint_start_state (tuple): pair of starting positions and orientations
            joint_goal_state (tuple): pair of goal positions and orientations
        """
        action_plans = []
        for i in range(2):
            agent_position_sequence = [joint_position[i] for joint_position in joint_positions]
            action_plan, _, _ = self.motion_planner.action_plan_from_positions(
                                    agent_position_sequence, joint_start_state[i], joint_goal_state[i])
            action_plans.append(action_plan)

        finishing_times = tuple(len(plan) for plan in action_plans)
        trimmed_action_plans = self._fix_plan_lengths(action_plans)
        joint_action_plan = list(zip(*trimmed_action_plans))
        end_pos_and_orientations = self._rollout_end_pos_and_or(joint_start_state, joint_action_plan)
        return joint_action_plan, end_pos_and_orientations, finishing_times

    def _fix_plan_lengths(self, plans):
        """Truncates the longer plan when shorter plan ends"""
        plans = list(plans)
        finishing_times = [len(p) for p in plans]
        delta_length = max(finishing_times) - min(finishing_times)
        if delta_length != 0:
            index_long_plan = np.argmax(finishing_times)
            long_plan = plans[index_long_plan]
            long_plan = long_plan[:min(finishing_times)]
        return plans

    def _rollout_end_pos_and_or(self, joint_start_state, joint_action_plan):
        """Execute plan in environment to determine ending positions and orientations"""
        # Assumes that final pos and orientations only depend on initial ones
        # (not on objects and other aspects of state).
        # Also assumes can't deliver more than two orders in one motion goal
        # (otherwise Environment will terminate)
        dummy_state = OvercookedState.from_players_pos_and_or(joint_start_state, order_list=['any', 'any'])
        env = OvercookedEnv.from_mdp(self.mdp, horizon=200) # Plans should be shorter than 200 timesteps, or something is likely wrong

        # remove action interact as the roll out passes a dummy state that some interacts will have world states that are important
        if ('interact' in joint_action_plan[-1]):
            successor_state, is_done = env.execute_plan(dummy_state, joint_action_plan[:-1])
        else:
            successor_state, is_done = env.execute_plan(dummy_state, joint_action_plan)
        assert not is_done
        return successor_state.players_pos_and_or

    def _joint_graph_from_grid(self):
        """Creates a graph instance from the mdp instance. Each graph node encodes a pair of positions"""
        state_decoder = {}
        # Valid positions pairs, not including ones with both players in same spot
        valid_joint_positions = self.mdp.get_valid_joint_player_positions()
        for state_index, joint_pos in enumerate(valid_joint_positions):
            state_decoder[state_index] = joint_pos

        state_encoder = {v:k for k, v in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for start_state_index, start_joint_positions in state_decoder.items():
            for joint_action, successor_jm_state in self._get_valid_successor_joint_positions(start_joint_positions).items():
                successor_node_index = state_encoder[successor_jm_state]

                # this_action_cost = self._graph_joint_action_cost(joint_action)
                # Below counts stay actions as cost of COST_OF_STAY. Above (original overcooked ai implementation) function, stay action costs nothing.
                this_action_cost = self._graph_joint_action_cost_include_stay(joint_action)
                current_cost = adjacency_matrix[start_state_index][successor_node_index]

                if current_cost == 0 or this_action_cost < current_cost:
                    adjacency_matrix[start_state_index][successor_node_index] = this_action_cost

        return Graph(adjacency_matrix, state_encoder, state_decoder)

    def _graph_joint_action_cost(self, joint_action):
        """The cost used in the graph shortest-path problem for a certain joint-action"""
        num_of_non_stay_actions = len([a for a in joint_action if a != Action.STAY])
        # NOTE: Removing the possibility of having 0 cost joint_actions
        if num_of_non_stay_actions == 0:
            return 1
        return num_of_non_stay_actions

    def _graph_joint_action_cost_include_stay(self, joint_action, COST_OF_STAY=1):
        """The cost used in the graph shortest-path problem for a certain joint-action"""
        num_of_non_stay_actions = len([a for a in joint_action if a != Action.STAY])
        num_of_stay_actions = len([a for a in joint_action if a == Action.STAY])
        # NOTE: Removing the possibility of having 0 cost joint_actions
        if (num_of_stay_actions + num_of_non_stay_actions) == 0:
            return 1

        total_cost_of_actions = num_of_non_stay_actions + num_of_stay_actions*COST_OF_STAY
        return total_cost_of_actions

    def _get_valid_successor_joint_positions(self, starting_positions):
        """Get all joint positions that can be reached by a joint action.
        NOTE: this DOES NOT include joint positions with superimposed agents."""
        successor_joint_positions = {}
        joint_motion_actions = itertools.product(Action.MOTION_ACTIONS, Action.MOTION_ACTIONS)
        
        # Under assumption that orientation doesn't matter
        dummy_orientation = Direction.NORTH
        dummy_player_states = [PlayerState(pos, dummy_orientation) for pos in starting_positions]
        for joint_action in joint_motion_actions:
            new_positions, _ = self.mdp.compute_new_positions_and_orientations(dummy_player_states, joint_action)
            successor_joint_positions[joint_action] = new_positions
        return successor_joint_positions

    def derive_state(self, start_state, end_pos_and_ors, action_plans):
        """
        Given a start state, end position and orientations, and an action plan, recovers
        the resulting state without executing the entire plan.
        """
        if len(action_plans) == 0:
            return start_state

        end_state = start_state.deepcopy()
        end_players = []
        for player, end_pos_and_or in zip(end_state.players, end_pos_and_ors):
            new_player = player.deepcopy()
            position, orientation = end_pos_and_or
            new_player.update_pos_and_or(position, orientation)
            end_players.append(new_player)
        
        end_state.players = tuple(end_players)

        # Resolve environment effects for t - 1 turns
        plan_length = len(action_plans)
        assert plan_length > 0
        for _ in range(plan_length - 1):
            self.mdp.step_environment_effects(end_state)

        # Interacts
        last_joint_action = tuple(a if a == Action.INTERACT else Action.STAY for a in action_plans[-1])

        events_dict = {k : [ [] for _ in range(self.mdp.num_players) ] for k in EVENT_TYPES}
        self.mdp.resolve_interacts(end_state, last_joint_action, events_dict)
        self.mdp.resolve_movement(end_state, last_joint_action)
        self.mdp.step_environment_effects(end_state)
        return end_state


class MediumLevelActionManager(object):
    """
    Manager for medium level actions (specific joint motion goals). 
    Determines available medium level actions for each state.
    
    Args:
        mdp (OvercookedGridWorld): gridworld of interest
        start_orientations (bool): whether the JointMotionPlanner should store plans for 
                                   all starting positions & orientations or just for unique 
                                   starting positions
    """

    def __init__(self, mdp, params):
        start_time = time.time()
        self.mdp = mdp
        
        self.params = params
        self.wait_allowed = params['wait_allowed']
        self.counter_drop = params["counter_drop"]
        self.counter_pickup = params["counter_pickup"]
        
        self.joint_motion_planner = JointMotionPlanner(mdp, params)
        self.motion_planner = self.joint_motion_planner.motion_planner
        print("It took {} seconds to create MediumLevelActionManager".format(time.time() - start_time))

    def save_to_file(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def joint_ml_actions(self, state):
        """Determine all possible joint medium level actions for a certain state"""
        agent1_actions, agent2_actions = tuple(self.get_medium_level_actions(state, player) for player in state.players)
        joint_ml_actions = list(itertools.product(agent1_actions, agent2_actions))
        
        # ml actions are nothing but specific joint motion goals
        valid_joint_ml_actions = list(filter(lambda a: self.is_valid_ml_action(state, a), joint_ml_actions))

        # HACK: Could cause things to break.
        # Necessary to prevent states without successors (due to no counters being allowed and no wait actions)
        # causing A* to not find a solution
        if len(valid_joint_ml_actions) == 0:
            agent1_actions, agent2_actions = tuple(self.get_medium_level_actions(state, player, waiting_substitute=True) for player in state.players)
            joint_ml_actions = list(itertools.product(agent1_actions, agent2_actions))
            valid_joint_ml_actions = list(filter(lambda a: self.is_valid_ml_action(state, a), joint_ml_actions))
            if len(valid_joint_ml_actions) == 0:
                print("WARNING: Found state without valid actions even after adding waiting substitute actions. State: {}".format(state))
        return valid_joint_ml_actions

    def is_valid_ml_action(self, state, ml_action):
        return self.joint_motion_planner.is_valid_jm_start_goal_pair(state.players_pos_and_or, ml_action)

    def get_medium_level_actions(self, state, player, waiting_substitute=False):
        """
        Determine valid medium level actions for a player.
        
        Args:
            state (OvercookedState): current state
            waiting_substitute (bool): add a substitute action that takes the place of 
                                       a waiting action (going to closest feature)
        
        Returns:
            player_actions (list): possible motion goals (pairs of goal positions and orientations)
        """
        player_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
        if not player.has_object():
            onion_pickup = self.pickup_onion_actions(counter_pickup_objects)
            tomato_pickup = self.pickup_tomato_actions(counter_pickup_objects)
            dish_pickup = self.pickup_dish_actions(counter_pickup_objects)
            soup_pickup = self.pickup_counter_soup_actions(counter_pickup_objects)
            player_actions.extend(onion_pickup + tomato_pickup + dish_pickup + soup_pickup)

        else:
            player_object = player.get_object()
            pot_states_dict = self.mdp.get_pot_states(state)

            # No matter the object, we can place it on a counter
            if len(self.counter_drop) > 0:
                player_actions.extend(self.place_obj_on_counter_actions(state))

            if player_object.name == 'soup':
                player_actions.extend(self.deliver_soup_actions())
            elif player_object.name == 'onion':
                player_actions.extend(self.put_onion_in_pot_actions(pot_states_dict))
            elif player_object.name == 'tomato':
                player_actions.extend(self.put_tomato_in_pot_actions(pot_states_dict))
            elif player_object.name == 'dish':
                # Not considering all pots (only ones close to ready) to reduce computation
                # NOTE: could try to calculate which pots are eligible, but would probably take
                # a lot of compute
                player_actions.extend(self.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=False))
            else:
                raise ValueError("Unrecognized object")

        if self.wait_allowed:
            player_actions.extend(self.wait_actions(player))

        if waiting_substitute:
            # Trying to mimic a "WAIT" action by adding the closest allowed feature to the avaliable actions
            # This is because motion plans that aren't facing terrain features (non counter, non empty spots)
            # are not considered valid
            player_actions.extend(self.go_to_closest_feature_actions(player))

        is_valid_goal_given_start = lambda goal: self.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, goal)    
        player_actions = list(filter(is_valid_goal_given_start, player_actions))
        return player_actions

    def pickup_onion_actions(self, counter_objects, only_use_dispensers=False, knowledge_base=None):
        """If only_use_dispensers is True, then only take onions from the dispensers"""
        onion_pickup_locations = self.mdp.get_onion_dispenser_locations()
        if not only_use_dispensers:
            onion_pickup_locations += counter_objects['onion']
        if knowledge_base is not None:
            for pos, obj in knowledge_base.items():
                if (obj is not None) and pos not in ['pot_states', 'chop_states', 'sink_states', 'other_player']:
                    if obj.name == 'onion':
                        onion_pickup_locations += [pos]
        return self._get_ml_actions_for_positions(onion_pickup_locations)

    def pickup_tomato_actions(self, counter_objects):
        tomato_dispenser_locations = self.mdp.get_tomato_dispenser_locations()
        tomato_pickup_locations = tomato_dispenser_locations + counter_objects['tomato']
        return self._get_ml_actions_for_positions(tomato_pickup_locations)
    
    def pickup_meat_actions(self, counter_objects, knowledge_base=None):
        meat_dispenser_locations = self.mdp.get_meat_dispenser_locations()
        meat_pickup_locations = meat_dispenser_locations + counter_objects['meat']
        if knowledge_base is not None:
            for pos, obj in knowledge_base.items():
                if (obj is not None) and pos not in ['pot_states', 'chop_states', 'sink_states', 'other_player']:
                    if obj.name in 'meat':
                        meat_pickup_locations += [pos]
        return self._get_ml_actions_for_positions(meat_pickup_locations)
    
    def pickup_dish_actions(self, counter_objects, only_use_dispensers=False, knowledge_base=None):
        """If only_use_dispensers is True, then only take dishes from the dispensers"""
        dish_pickup_locations = self.mdp.get_dish_dispenser_locations()
        if not only_use_dispensers:
            dish_pickup_locations += counter_objects['dish']
            if knowledge_base is not None:
                for pos, obj in knowledge_base.items():
                    if (obj is not None) and pos not in ['pot_states', 'chop_states', 'sink_states', 'other_player']:
                        if obj.name == 'dish':
                            dish_pickup_locations += [pos]
        return self._get_ml_actions_for_positions(dish_pickup_locations)
    
    def pickup_plate_actions(self, counter_objects, only_use_dispensers=False, knowledge_base=None):
        """If only_use_dispensers is True, then only take dishes from the dispensers"""
        plate_pickup_locations = self.mdp.get_plate_dispenser_locations()
        if not only_use_dispensers:
            plate_pickup_locations += counter_objects['plate']
            if knowledge_base is not None:
                for pos, obj in knowledge_base.items():
                    if (obj is not None) and pos not in ['pot_states', 'chop_states', 'sink_states', 'other_player']:
                        if obj.name == 'plate':
                            plate_pickup_locations += [pos]
        return self._get_ml_actions_for_positions(plate_pickup_locations)

    def pickup_counter_soup_actions(self, counter_objects):
        soup_pickup_locations = counter_objects['soup']
        return self._get_ml_actions_for_positions(soup_pickup_locations)

    def place_obj_on_counter_actions(self, state):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters]
        return self._get_ml_actions_for_positions(valid_empty_counters)

    def deliver_soup_actions(self):
        serving_locations = self.mdp.get_serving_locations()
        return self._get_ml_actions_for_positions(serving_locations)
    
    def deliver_dish_actions(self): # dish here is a plate with fully garnished steak on it
        serving_locations = self.mdp.get_serving_locations()
        return self._get_ml_actions_for_positions(serving_locations)

    def put_onion_in_pot_actions(self, pot_states_dict):
        partially_full_onion_pots = pot_states_dict['onion']['partially_full']
        fillable_pots = partially_full_onion_pots + pot_states_dict['empty']
        return self._get_ml_actions_for_positions(fillable_pots)

    def put_tomato_in_pot_actions(self, pot_states_dict):
        partially_full_tomato_pots = pot_states_dict['tomato']['partially_full']
        fillable_pots = partially_full_tomato_pots + pot_states_dict['empty']
        return self._get_ml_actions_for_positions(fillable_pots)
    
    def put_meat_in_pot_actions(self, pot_states_dict):
        partially_full_steak_pots = pot_states_dict['steak']['partially_full']
        fillable_pots = partially_full_steak_pots + pot_states_dict['empty']
        return self._get_ml_actions_for_positions(fillable_pots)
    
    def put_onion_on_board_actions(self, state, knowledge_base=None):
        empty_boards = []
        if knowledge_base is not None:
            empty_boards += knowledge_base['chop_states']['empty']
        else:
            board_locations = self.mdp.get_chopping_board_locations()
            for loc in board_locations:
                if not state.has_object(loc): # board is empty
                    empty_boards.append(loc)
        return self._get_ml_actions_for_positions(empty_boards)
    
    def chop_onion_on_board_actions(self, state, knowledge_base=None):
        full_boards = []
        if knowledge_base is not None:
            full_boards += knowledge_base['chop_states']['full']
        else:
            board_locations = self.mdp.get_chopping_board_locations()
            for loc in board_locations:
                if state.has_object(loc): # board is with onion
                    full_boards.append(loc)
        return self._get_ml_actions_for_positions(full_boards)
    
    def put_plate_in_sink_actions(self, counter_objects, state, knowledge_base=None):
        empty_sink = []
        plate_on_counter = []
        if knowledge_base is not None:
            empty_sink = knowledge_base['sink_states']['empty']
        else:
            sink_locations = self.mdp.get_sink_locations()
            for loc in sink_locations:
                if not state.has_object(loc): # board is empty
                    empty_sink.append(loc)
            plate_on_counter = counter_objects['plate']
        return self._get_ml_actions_for_positions(empty_sink) 
    
    def heat_plate_in_sink_actions(self, state, knowledge_base=None):
        heat_needed_loc = []
        if knowledge_base is not None:
            heat_needed_loc = knowledge_base['sink_states']['full']
        else:
            sink_locations = self.mdp.get_sink_locations()
            for loc in sink_locations:
                if state.has_object(loc):
                    if state.get_object(loc).state < self.mdp.wash_time:
                        heat_needed_loc.append(loc)
        return self._get_ml_actions_for_positions(heat_needed_loc)

    def add_garnish_to_steak_actions(self, state, knowledge_base=None):
        garnish_chopped_loc = []
        if knowledge_base is not None:
            garnish_chopped_loc = knowledge_base['chop_states']['ready']
            if len(garnish_chopped_loc) == 0:
                garnish_chopped_loc = knowledge_base['chop_states']['full']
            if len(garnish_chopped_loc) == 0:
                robot_obj = knowledge_base['other_player'].held_object.name if knowledge_base['other_player'].held_object is not None else 'None'
                if robot_obj == 'onion':
                    garnish_chopped_loc = knowledge_base['chop_states']['empty']
        else:
            board_locations = self.mdp.get_chopping_board_locations()
            for loc in board_locations:
                if state.has_object(loc):
                    chop_time = state.get_object(loc).state
                    if chop_time >= self.mdp.chopping_time:
                        garnish_chopped_loc.append(loc)
        return self._get_ml_actions_for_positions(garnish_chopped_loc)
    
    def pickup_hot_plate_from_sink_actions(self, counter_objects, state, knowledge_base=None):
        hot_plate_loc = []
        hot_plate_on_counter = []
        if knowledge_base is not None:
            hot_plate_loc = knowledge_base['sink_states']['ready']
            for pos, obj in knowledge_base.items():
                if (obj is not None) and pos not in ['pot_states', 'chop_states', 'sink_states', 'other_player']:
                    if obj.name == 'hot_plate':
                        hot_plate_on_counter += [pos]
        else:
            sink_locations = self.mdp.get_sink_locations()
            for loc in sink_locations:
                if state.has_object(loc):
                    wash_time = state.get_object(loc).state
                    if wash_time >= self.mdp.wash_time:
                        hot_plate_loc.append(loc)
            hot_plate_on_counter = counter_objects['hot_plate']
        return self._get_ml_actions_for_positions(hot_plate_loc + hot_plate_on_counter)

    def pickup_soup_with_dish_actions(self, pot_states_dict, only_nearly_ready=False):
        ready_pot_locations = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        nearly_ready_pot_locations = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        if not only_nearly_ready:
            partially_full_pots = pot_states_dict['tomato']['partially_full'] + pot_states_dict['onion']['partially_full']
            nearly_ready_pot_locations = nearly_ready_pot_locations + pot_states_dict['empty'] + partially_full_pots
        return self._get_ml_actions_for_positions(ready_pot_locations + nearly_ready_pot_locations)
    
    def pickup_steak_with_hot_plate_actions(self, pot_states_dict, only_nearly_ready=False):
        ready_pot_locations = pot_states_dict['steak']['ready']
        nearly_ready_pot_locations = pot_states_dict['steak']['cooking']
        if not only_nearly_ready:
            partially_full_pots = pot_states_dict['steak']['partially_full']
            nearly_ready_pot_locations = nearly_ready_pot_locations + pot_states_dict['empty'] + partially_full_pots
        return self._get_ml_actions_for_positions(ready_pot_locations + nearly_ready_pot_locations)

    def go_to_closest_feature_actions(self, player):
        feature_locations = self.mdp.get_onion_dispenser_locations() + self.mdp.get_tomato_dispenser_locations() + \
                            self.mdp.get_pot_locations() + self.mdp.get_dish_dispenser_locations()
        closest_feature_pos = self.motion_planner.min_cost_to_feature(player.pos_and_or, feature_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_feature_pos])
    
    def get_closest_counter(self, state, player):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters]
        closest_empty_counter = self.motion_planner.min_cost_to_feature(player.pos_and_or, valid_empty_counters, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_empty_counter])

    def go_to_closest_feature_or_counter_to_goal(self, goal_pos_and_or, goal_location):
        """Instead of going to goal_pos_and_or, go to the closest feature or counter to this goal, that ISN'T the goal itself"""
        valid_locations = self.mdp.get_onion_dispenser_locations() + \
                                    self.mdp.get_tomato_dispenser_locations() + self.mdp.get_pot_locations() + \
                                    self.mdp.get_dish_dispenser_locations() + self.counter_drop
        valid_locations.remove(goal_location)
        closest_non_goal_feature_pos = self.motion_planner.min_cost_to_feature(
                                            goal_pos_and_or, valid_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_non_goal_feature_pos])

    def wait_actions(self, player):
        waiting_motion_goal = (player.position, player.orientation)
        return [waiting_motion_goal]

    def _get_ml_actions_for_positions(self, positions_list):
        """Determine what are the ml actions (joint motion goals) for a list of positions
        
        Args:
            positions_list (list): list of target terrain feature positions
        """
        possible_motion_goals = []
        for pos in positions_list:
            # All possible ways to reach the target feature
            for motion_goal in self.joint_motion_planner.motion_planner.motion_goals_for_pos[pos]:
                possible_motion_goals.append(motion_goal)
        return possible_motion_goals


class MediumLevelPlanner(object):
    """
    A planner that computes optimal plans for two agents to deliver a certain number of dishes 
    in an OvercookedGridworld using medium level actions (single motion goals) in the corresponding
    A* search problem.
    """

    def __init__(self, mdp, mlp_params, ml_action_manager=None):
        self.mdp = mdp
        self.params = mlp_params
        self.ml_action_manager = ml_action_manager if ml_action_manager else MediumLevelActionManager(mdp, mlp_params)
        self.jmp = self.ml_action_manager.joint_motion_planner
        self.mp = self.jmp.motion_planner

    @staticmethod
    def from_action_manager_file(filename):
        mlp_action_manager = load_saved_action_manager(filename)
        mdp = mlp_action_manager.mdp
        params = mlp_action_manager.params
        return MediumLevelPlanner(mdp, params, mlp_action_manager)
    
    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute=False, info=True):
        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + "_am.pkl"

        if force_compute:
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)
        
        try:
            mlp = MediumLevelPlanner.from_action_manager_file(filename)

            if mlp.ml_action_manager.params != mlp_params or mlp.mdp != mdp:
                print("Mlp with different params or mdp found, computing from scratch")
                return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        if info:
            print("Loaded MediumLevelPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))
        return mlp

    @staticmethod
    def compute_mlp(filename, mdp, mlp_params):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        print("Computing MediumLevelPlanner to be saved in {}".format(final_filepath))
        start_time = time.time()
        mlp = MediumLevelPlanner(mdp, mlp_params=mlp_params)
        print("It took {} seconds to create mlp".format(time.time() - start_time))
        mlp.ml_action_manager.save_to_file(final_filepath)
        return mlp

    def get_low_level_action_plan(self, start_state, h_fn, delivery_horizon=4, debug=False, goal_info=False):
        """
        Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries
        
        Args:
            state (OvercookedState): starting state

        Returns:
            full_joint_action_plan (list): joint actions to reach goal
        """
        start_state = start_state.deepcopy()
        ml_plan, cost = self.get_ml_plan(start_state, h_fn, delivery_horizon=delivery_horizon, debug=debug)

        if start_state.order_list is None:
            start_state.order_list = ['any'] * delivery_horizon
            
        full_joint_action_plan = self.get_low_level_plan_from_ml_plan(
            start_state, ml_plan, h_fn, debug=debug, goal_info=goal_info
        )
        assert cost == len(full_joint_action_plan), "A* cost {} but full joint action plan cost {}".format(cost, len(full_joint_action_plan))
        if debug: print("Found plan with cost {}".format(cost))
        return full_joint_action_plan

    def get_low_level_plan_from_ml_plan(self, start_state, ml_plan, heuristic_fn, debug=False, goal_info=False):
        t = 0
        full_joint_action_plan = []
        curr_state = start_state
        curr_motion_state = start_state.players_pos_and_or
        prev_h = heuristic_fn(start_state, t, debug=False)
        
        if len(ml_plan) > 0 and goal_info:
            print("First motion goal: ", ml_plan[0][0])

        if debug:
            print("Start state")
            OvercookedEnv.print_state(self.mdp, start_state)

        for joint_motion_goal, goal_state in ml_plan:
            joint_action_plan, end_motion_state, plan_costs = \
                self.ml_action_manager.joint_motion_planner.get_low_level_action_plan(curr_motion_state, joint_motion_goal)
            curr_plan_cost = min(plan_costs)
            full_joint_action_plan.extend(joint_action_plan)
            t += 1

            if debug:
                print(t)
                OvercookedEnv.print_state(self.mdp, goal_state)
            
            if SAFE_RUN:
                env = OvercookedEnv.from_mdp(self.mdp, info_level = 0, horizon = 100)
                s_prime, _ = OvercookedEnv.execute_plan(env, curr_state, joint_action_plan)
                assert s_prime == goal_state

            curr_h = heuristic_fn(goal_state, t, debug=False)
            self.check_heuristic_consistency(curr_h, prev_h, curr_plan_cost)
            curr_motion_state, prev_h, curr_state = end_motion_state, curr_h, goal_state
        return full_joint_action_plan

    def check_heuristic_consistency(self, curr_heuristic_val, prev_heuristic_val, actual_edge_cost):
        delta_h = curr_heuristic_val - prev_heuristic_val
        assert actual_edge_cost >= delta_h, \
            "Heuristic was not consistent. \n Prev h: {}, Curr h: {}, Actual cost: {}, Δh: {}" \
            .format(prev_heuristic_val, curr_heuristic_val, actual_edge_cost, delta_h)

    def get_ml_plan(self, start_state, h_fn, delivery_horizon=4, debug=False):
        """
        Solves A* Search problem to find optimal sequence of medium level actions
        to reach the goal number of deliveries

        Returns:
            ml_plan (list): plan not including starting state in form
                [(joint_action, successor_state), ..., (joint_action, goal_state)]
            cost (int): A* Search cost
        """
        start_state = start_state.deepcopy()
        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon
        else:
            start_state.order_list = start_state.order_list[:delivery_horizon]
        
        expand_fn = lambda state: self.get_successor_states(state)
        goal_fn = lambda state: state.num_orders_remaining == 0
        heuristic_fn = lambda state: h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
        ml_plan, cost = search_problem.A_star_graph_search(info=False)
        return ml_plan[1:], cost
    
    def get_successor_states(self, start_state):
        """Successor states for medium-level actions are defined as
        the first state in the corresponding motion plan in which 
        one of the two agents' subgoals is satisfied.
    
        Returns: list of
            joint_motion_goal: ((pos1, or1), (pos2, or2)) specifying the 
                                motion plan goal for both agents

            successor_state:   OvercookedState corresponding to state
                               arrived at after executing part of the motion plan
                               (until one of the agents arrives at his goal status)

            plan_length:       Time passed until arrival to the successor state
        """
        if self.mdp.is_terminal(start_state):
            return []

        start_jm_state = start_state.players_pos_and_or
        successor_states = []
        for goal_jm_state in self.ml_action_manager.joint_ml_actions(start_state):
            joint_motion_action_plans, end_pos_and_ors, plan_costs = self.jmp.get_low_level_action_plan(start_jm_state, goal_jm_state)
            end_state = self.jmp.derive_state(start_state, end_pos_and_ors, joint_motion_action_plans)

            if SAFE_RUN:
                assert end_pos_and_ors[0] == goal_jm_state[0] or end_pos_and_ors[1] == goal_jm_state[1]
                env = OvercookedEnv.from_mdp(self.mdp, info_level = 0, horizon = 100)
                s_prime, _ = OvercookedEnv.execute_plan(env, start_state, joint_motion_action_plans, display=False)
                assert end_state == s_prime

            successor_states.append((goal_jm_state, end_state, min(plan_costs)))
        return successor_states

    def get_successor_states_fixed_other(self, start_state, other_agent, other_agent_idx):
        """
        Get the successor states of a given start state, assuming that the other agent is fixed and will act according to the passed in model
        """
        if self.mdp.is_terminal(start_state):
            return []

        player = start_state.players[1 - other_agent_idx]
        ml_actions = self.ml_action_manager.get_medium_level_actions(start_state, player)

        if len(ml_actions) == 0:
            ml_actions = self.ml_action_manager.get_medium_level_actions(start_state, player, waiting_substitute=True)

        successor_high_level_states = []
        for ml_action in ml_actions:
            action_plan, end_state, cost = self.get_embedded_low_level_action_plan(start_state, ml_action, other_agent, other_agent_idx)
            
            if not self.mdp.is_terminal(end_state):
                # Adding interact action and deriving last state
                other_agent_action, _ = other_agent.action(end_state)
                last_joint_action = (Action.INTERACT, other_agent_action) if other_agent_idx == 1 else (other_agent_action, Action.INTERACT)
                action_plan = action_plan + (last_joint_action,)
                cost = cost + 1

                end_state, _ = self.embedded_mdp_step(end_state, Action.INTERACT, other_agent_action, other_agent.agent_index)

            successor_high_level_states.append((action_plan, end_state, cost))
        return successor_high_level_states

    def get_embedded_low_level_action_plan(self, state, goal_pos_and_or, other_agent, other_agent_idx):
        """Find action plan for a specific motion goal with A* considering the other agent"""
        other_agent.set_agent_index(other_agent_idx)
        agent_idx = 1 - other_agent_idx

        expand_fn = lambda state: self.embedded_mdp_succ_fn(state, other_agent)
        goal_fn = lambda state: state.players[agent_idx].pos_and_or == goal_pos_and_or or state.num_orders_remaining == 0
        heuristic_fn = lambda state: sum(pos_distance(state.players[agent_idx].position, goal_pos_and_or[0]))

        search_problem = SearchTree(state, goal_fn, expand_fn, heuristic_fn)
        state_action_plan, cost = search_problem.A_star_graph_search(info=False)
        action_plan, state_plan = zip(*state_action_plan)
        action_plan = action_plan[1:]
        end_state = state_plan[-1]
        return action_plan, end_state, cost

    def embedded_mdp_succ_fn(self, state, other_agent):
        other_agent_action, _ = other_agent.action(state)

        successors = []
        for a in Action.ALL_ACTIONS:
            successor_state, joint_action = self.embedded_mdp_step(state, a, other_agent_action, other_agent.agent_index)
            cost = 1
            successors.append((joint_action, successor_state, cost))
        return successors

    def embedded_mdp_step(self, state, action, other_agent_action, other_agent_index):
        if other_agent_index == 0:
            joint_action = (other_agent_action, action)
        else:
            joint_action = (action, other_agent_action)
        if not self.mdp.is_terminal(state):
            results, _, _, _ = self.mdp.get_state_transition(state, joint_action)
            successor_state = results
        else:
            print("Tried to find successor of terminal")
            assert False, "state {} \t action {}".format(state, action)
            successor_state = state
        return successor_state, joint_action


class HighLevelAction:
    """A high level action is given by a set of subsequent motion goals"""

    def __init__(self, motion_goals):
        self.motion_goals = motion_goals
    
    def _check_valid(self):
        for goal in self.motion_goals:
            assert len(goal) == 2
            pos, orient = goal
            assert orient in Direction.ALL_DIRECTIONS
            assert type(pos) is tuple
            assert len(pos) == 2

    def __getitem__(self, i):
        """Get ith motion goal of the HL Action"""
        return self.motion_goals[i]


class HighLevelActionManager(object):
    """
    Manager for high level actions. Determines available high level actions 
    for each state and player.
    """

    def __init__(self, medium_level_planner):
        self.mdp = medium_level_planner.mdp
        
        self.wait_allowed = medium_level_planner.params['wait_allowed']
        self.counter_drop = medium_level_planner.params["counter_drop"]
        self.counter_pickup = medium_level_planner.params["counter_pickup"]
        
        self.mlp = medium_level_planner
        self.ml_action_manager = medium_level_planner.ml_action_manager
        self.mp = medium_level_planner.mp

    def joint_hl_actions(self, state):
        hl_actions_a0, hl_actions_a1 = tuple(self.get_high_level_actions(state, player) for player in state.players)
        joint_hl_actions = list(itertools.product(hl_actions_a0, hl_actions_a1))

        assert self.mlp.params["same_motion_goals"]
        valid_joint_hl_actions = joint_hl_actions

        if len(valid_joint_hl_actions) == 0:
            print("WARNING: found a state without high level successors")
        return valid_joint_hl_actions

    def get_high_level_actions(self, state, player):
        player_hl_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
        if player.has_object():
            place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player)

            # HACK to prevent some states not having successors due to lack of waiting actions
            if len(place_obj_ml_actions) == 0:
                place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player, waiting_substitute=True)

            place_obj_hl_actions = [HighLevelAction([ml_action]) for ml_action in place_obj_ml_actions]
            player_hl_actions.extend(place_obj_hl_actions)
        else:
            pot_states_dict = self.mdp.get_pot_states(state)
            player_hl_actions.extend(self.get_onion_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
            player_hl_actions.extend(self.get_tomato_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
            player_hl_actions.extend(self.get_dish_and_soup_and_serve(state, counter_pickup_objects, pot_states_dict))
        return player_hl_actions

    def get_dish_and_soup_and_serve(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting a dish, 
        going to a pot and picking up a soup, and delivering the soup."""
        dish_pickup_actions = self.ml_action_manager.pickup_dish_actions(counter_objects)
        pickup_soup_actions = self.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict)
        deliver_soup_actions = self.ml_action_manager.deliver_soup_actions()
        hl_level_actions = list(itertools.product(dish_pickup_actions, pickup_soup_actions, deliver_soup_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]

    def get_onion_and_put_in_pot(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting an onion
        from a dispenser and placing it in a pot."""
        onion_pickup_actions = self.ml_action_manager.pickup_onion_actions(counter_objects)
        put_in_pot_actions = self.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)
        hl_level_actions = list(itertools.product(onion_pickup_actions, put_in_pot_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]

    def get_tomato_and_put_in_pot(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting an tomato
        from a dispenser and placing it in a pot."""
        tomato_pickup_actions = self.ml_action_manager.pickup_tomato_actions(counter_objects)
        put_in_pot_actions = self.ml_action_manager.put_tomato_in_pot_actions(pot_states_dict)
        hl_level_actions = list(itertools.product(tomato_pickup_actions, put_in_pot_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]


class HighLevelPlanner(object):
    """A planner that computes optimal plans for two agents to 
    deliver a certain number of dishes in an OvercookedGridworld
    using high level actions in the corresponding A* search problems
    """

    def __init__(self, hl_action_manager):
        self.hl_action_manager = hl_action_manager
        self.mlp = self.hl_action_manager.mlp
        self.jmp = self.mlp.ml_action_manager.joint_motion_planner
        self.mp = self.jmp.motion_planner
        self.mdp = self.mlp.mdp

    def get_successor_states(self, start_state):
        """Determines successor states for high-level actions"""
        successor_states = []

        if self.mdp.is_terminal(start_state):
            return successor_states

        for joint_hl_action in self.hl_action_manager.joint_hl_actions(start_state):
            _, end_state, hl_action_cost = self.perform_hl_action(joint_hl_action, start_state)

            successor_states.append((joint_hl_action, end_state, hl_action_cost))
        return successor_states

    def perform_hl_action(self, joint_hl_action, curr_state):
        """Determines the end state for a high level action, and the corresponding low level action plan and cost.
        Will return Nones if a pot exploded throughout the execution of the action"""
        full_plan = []
        motion_goal_indices = (0, 0)
        total_cost = 0
        while not self.at_least_one_finished_hl_action(joint_hl_action, motion_goal_indices):
            curr_jm_goal = tuple(joint_hl_action[i].motion_goals[motion_goal_indices[i]] for i in range(2))
            joint_motion_action_plans, end_pos_and_ors, plan_costs = \
                self.jmp.get_low_level_action_plan(curr_state.players_pos_and_or, curr_jm_goal)
            curr_state = self.jmp.derive_state(curr_state, end_pos_and_ors, joint_motion_action_plans)
            motion_goal_indices = self._advance_motion_goal_indices(motion_goal_indices, plan_costs)
            total_cost += min(plan_costs)
            full_plan.extend(joint_motion_action_plans)
        return full_plan, curr_state, total_cost

    def at_least_one_finished_hl_action(self, joint_hl_action, motion_goal_indices):
        """Returns whether either agent has reached the end of the motion goal list it was supposed
        to perform to finish it's high level action"""
        return any([len(joint_hl_action[i].motion_goals) == motion_goal_indices[i] for i in range(2)])

    def get_low_level_action_plan(self, start_state, h_fn, debug=False):
        """
        Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries
        by performaing an A* search in high-level action space
        
        Args:
            state (OvercookedState): starting state

        Returns:
            full_joint_action_plan (list): joint actions to reach goal
            cost (int): a cost in number of timesteps to reach the goal
        """
        full_joint_low_level_action_plan = []
        hl_plan, cost = self.get_hl_plan(start_state, h_fn)
        curr_state = start_state
        prev_h = h_fn(start_state, debug=False)
        total_cost = 0
        for joint_hl_action, curr_goal_state in hl_plan:
            assert all([type(a) is HighLevelAction for a in joint_hl_action])
            hl_action_plan, curr_state, hl_action_cost = self.perform_hl_action(joint_hl_action, curr_state)
            full_joint_low_level_action_plan.extend(hl_action_plan)
            total_cost += hl_action_cost
            assert curr_state == curr_goal_state

            curr_h = h_fn(curr_state, debug=False)
            self.mlp.check_heuristic_consistency(curr_h, prev_h, total_cost)
            prev_h = curr_h
        assert total_cost == cost == len(full_joint_low_level_action_plan), "{} vs {} vs {}"\
            .format(total_cost, cost, len(full_joint_low_level_action_plan))
        return full_joint_low_level_action_plan, cost
    
    def get_hl_plan(self, start_state, h_fn, debug=False):
        expand_fn = lambda state: self.get_successor_states(state)
        goal_fn = lambda state: state.num_orders_remaining == 0
        heuristic_fn = lambda state: h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
        hl_plan, cost = search_problem.A_star_graph_search(info=True)
        return hl_plan[1:], cost

    def _advance_motion_goal_indices(self, curr_plan_indices, plan_lengths):
        """Advance indices for agents current motion goals
        based on who finished their motion goal this round"""
        idx0, idx1 = curr_plan_indices
        if plan_lengths[0] == plan_lengths[1]:
            return idx0 + 1, idx1 + 1

        who_finished = np.argmin(plan_lengths)
        if who_finished == 0:
            return idx0 + 1, idx1
        elif who_finished == 1:
            return idx0, idx1 + 1


class Heuristic(object):

    def __init__(self, mp):
        self.motion_planner = mp
        self.mdp = mp.mdp
        self.heuristic_cost_dict = self._calculate_heuristic_costs()
    
    def hard_heuristic(self, state, goal_deliveries, time=0, debug=False):
        # NOTE: does not support tomatoes – currently deprecated as harder heuristic
        # does not seem worth the additional computational time

        """
        From a state, we can calculate exactly how many:
        - soup deliveries we need
        - dishes to pots we need
        - onion to pots we need

        We then determine if there are any soups/dishes/onions
        in transit (on counters or on players) than can be 
        brought to their destinations faster than starting off from
        a dispenser of the same type. If so, we consider fulfilling
        all demand from these positions. 

        After all in-transit objects are considered, we consider the
        costs required to fulfill all the rest of the demand, that is 
        given by:
        - pot-delivery trips
        - dish-pot trips
        - onion-pot trips
        
        The total cost is obtained by determining an optimistic time 
        cost for each of these trip types
        """
        forward_cost = 0

        # Obtaining useful quantities
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        min_pot_delivery_cost = self.heuristic_cost_dict['pot-delivery']
        min_dish_to_pot_cost = self.heuristic_cost_dict['dish-pot']
        min_onion_to_pot_cost = self.heuristic_cost_dict['onion-pot']

        pot_locations = self.mdp.get_pot_locations()
        full_soups_in_pots = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking'] \
                             + pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        partially_full_soups = pot_states_dict['onion']['partially_full'] + pot_states_dict['tomato']['partially_full']
        num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_soups])

        # Calculating costs
        num_deliveries_to_go = goal_deliveries - state.num_delivered

        # SOUP COSTS
        total_num_soups_needed = max([0, num_deliveries_to_go])
        
        soups_on_counters = [soup_obj for soup_obj in objects_dict['soup'] if soup_obj.position not in pot_locations]
        soups_in_transit = player_objects['soup'] + soups_on_counters
        soup_delivery_locations = self.mdp.get_serving_locations()
        
        num_soups_better_than_pot, total_better_than_pot_soup_cost = \
            self.get_costs_better_than_dispenser(soups_in_transit, soup_delivery_locations, min_pot_delivery_cost, total_num_soups_needed, state)
        
        min_pot_to_delivery_trips = max([0, total_num_soups_needed - num_soups_better_than_pot])
        pot_to_delivery_costs = min_pot_delivery_cost * min_pot_to_delivery_trips

        forward_cost += total_better_than_pot_soup_cost
        forward_cost += pot_to_delivery_costs

        # DISH COSTS
        total_num_dishes_needed = max([0, min_pot_to_delivery_trips])
        dishes_on_counters = objects_dict['dish']
        dishes_in_transit = player_objects['dish'] + dishes_on_counters
        
        num_dishes_better_than_disp, total_better_than_disp_dish_cost = \
            self.get_costs_better_than_dispenser(dishes_in_transit, pot_locations, min_dish_to_pot_cost, total_num_dishes_needed, state)

        min_dish_to_pot_trips = max([0, min_pot_to_delivery_trips - num_dishes_better_than_disp])
        dish_to_pot_costs = min_dish_to_pot_cost * min_dish_to_pot_trips

        forward_cost += total_better_than_disp_dish_cost
        forward_cost += dish_to_pot_costs

        # ONION COSTS
        num_pots_to_be_filled = min_pot_to_delivery_trips - len(full_soups_in_pots)
        total_num_onions_needed = num_pots_to_be_filled * 3 - num_onions_in_partially_full_pots
        onions_on_counters = objects_dict['onion']
        onions_in_transit = player_objects['onion'] + onions_on_counters

        num_onions_better_than_disp, total_better_than_disp_onion_cost = \
            self.get_costs_better_than_dispenser(onions_in_transit, pot_locations, min_onion_to_pot_cost, total_num_onions_needed, state)

        min_onion_to_pot_trips = max([0, total_num_onions_needed - num_onions_better_than_disp])
        onion_to_pot_costs = min_onion_to_pot_cost * min_onion_to_pot_trips
        
        forward_cost += total_better_than_disp_onion_cost
        forward_cost += onion_to_pot_costs

        # Going to closest feature costs
        # NOTE: as implemented makes heuristic inconsistent
        # for player in state.players:
        #     if not player.has_object():
        #         counter_objects = soups_on_counters + dishes_on_counters + onions_on_counters
        #         possible_features = counter_objects + pot_locations + self.mdp.get_dish_dispenser_locations() + self.mdp.get_onion_dispenser_locations()
        #         forward_cost += self.action_manager.min_cost_to_feature(player.pos_and_or, possible_features)

        heuristic_cost = forward_cost / 2
        
        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            print("\n" + "#"*35)
            print("Current state: (ml timestep {})\n".format(time))

            print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
            ))

            # NOTE Possible improvement: consider cost of dish delivery too when considering if a
            # transit soup is better than dispenser equivalent
            print("# better than disp: \t Soups {} \t Dishes {} \t Onions {}".format(
                num_soups_better_than_pot, num_dishes_better_than_disp, num_onions_better_than_disp
            ))

            print("# of trips: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                min_pot_to_delivery_trips, min_dish_to_pot_trips, min_onion_to_pot_trips
            ))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
            ))

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost

    def get_costs_better_than_dispenser(self, possible_objects, target_locations, baseline_cost, num_needed, state):
        """
        Computes the number of objects whose minimum cost to any of the target locations is smaller than
        the baseline cost (clipping it if greater than the number needed). It also calculates a lower
        bound on the cost of using such objects.
        """
        costs_from_transit_locations = []
        for obj in possible_objects:
            obj_pos = obj.position
            if obj_pos in state.player_positions:
                # If object is being carried by a player
                player = [p for p in state.players if p.position == obj_pos][0]
                # NOTE: not sure if this -1 is justified.
                # Made things work better in practice for greedy heuristic based agents.
                # For now this function is just used from there. Consider removing later if
                # greedy heuristic agents end up not being used.
                min_cost = self.motion_planner.min_cost_to_feature(player.pos_and_or, target_locations) - 1
            else:
                # If object is on a counter
                min_cost = self.motion_planner.min_cost_between_features([obj_pos], target_locations)
            costs_from_transit_locations.append(min_cost)
        
        costs_better_than_dispenser = [cost for cost in costs_from_transit_locations if cost <= baseline_cost]
        better_than_dispenser_total_cost = sum(np.sort(costs_better_than_dispenser)[:num_needed])
        return len(costs_better_than_dispenser), better_than_dispenser_total_cost

    def _calculate_heuristic_costs(self, debug=False):
        """Pre-computes the costs between common trip types for this mdp"""
        pot_locations = self.mdp.get_pot_locations()
        delivery_locations = self.mdp.get_serving_locations()
        dish_locations = self.mdp.get_dish_dispenser_locations()
        onion_locations = self.mdp.get_onion_dispenser_locations()
        tomato_locations = self.mdp.get_tomato_dispenser_locations()

        heuristic_cost_dict = {
            'pot-delivery': self.motion_planner.min_cost_between_features(pot_locations, delivery_locations, manhattan_if_fail=True),
            'dish-pot': self.motion_planner.min_cost_between_features(dish_locations, pot_locations, manhattan_if_fail=True)
        }

        onion_pot_cost = self.motion_planner.min_cost_between_features(onion_locations, pot_locations, manhattan_if_fail=True)
        tomato_pot_cost = self.motion_planner.min_cost_between_features(tomato_locations, pot_locations, manhattan_if_fail=True)

        if debug: print("Heuristic cost dict", heuristic_cost_dict)
        assert onion_pot_cost != np.inf or tomato_pot_cost != np.inf
        if onion_pot_cost != np.inf:
            heuristic_cost_dict['onion-pot'] = onion_pot_cost
        if tomato_pot_cost != np.inf:
            heuristic_cost_dict['tomato-pot'] = tomato_pot_cost
        
        return heuristic_cost_dict
    
    def simple_heuristic(self, state, time=0, debug=False):
        """Simpler heuristic that tends to run faster than current one"""
        # NOTE: State should be modified to have an order list w.r.t. which
        # one can calculate the heuristic
        assert state.order_list is not None
        
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        num_deliveries_to_go = state.num_orders_remaining
        
        full_soups_in_pots = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking'] \
                             + pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        partially_full_onion_soups = pot_states_dict['onion']['partially_full']
        partially_full_tomato_soups = pot_states_dict['tomato']['partially_full']
        num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_onion_soups])
        num_tomatoes_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_tomato_soups])

        soups_in_transit = player_objects['soup']
        dishes_in_transit = objects_dict['dish'] + player_objects['dish']
        onions_in_transit = objects_dict['onion'] + player_objects['onion']
        tomatoes_in_transit = objects_dict['tomato'] + player_objects['tomato']

        num_pot_to_delivery = max([0, num_deliveries_to_go - len(soups_in_transit)])
        num_dish_to_pot = max([0, num_pot_to_delivery - len(dishes_in_transit)])

        num_pots_to_be_filled = num_pot_to_delivery - len(full_soups_in_pots)
        num_onions_needed_for_pots = num_pots_to_be_filled * 3 - len(onions_in_transit) - num_onions_in_partially_full_pots
        num_tomatoes_needed_for_pots = num_pots_to_be_filled * 3 - len(tomatoes_in_transit) - num_tomatoes_in_partially_full_pots
        num_onion_to_pot = max([0, num_onions_needed_for_pots])
        num_tomato_to_pot = max([0, num_tomatoes_needed_for_pots])

        pot_to_delivery_costs = self.heuristic_cost_dict['pot-delivery'] * num_pot_to_delivery
        dish_to_pot_costs = self.heuristic_cost_dict['dish-pot'] * num_dish_to_pot

        items_to_pot_costs = []
        if 'onion-pot' in self.heuristic_cost_dict.keys():
            onion_to_pot_costs = self.heuristic_cost_dict['onion-pot'] * num_onion_to_pot
            items_to_pot_costs.append(onion_to_pot_costs)
        if 'tomato-pot' in self.heuristic_cost_dict.keys():
            tomato_to_pot_costs = self.heuristic_cost_dict['tomato-pot'] * num_tomato_to_pot
            items_to_pot_costs.append(tomato_to_pot_costs)

        # NOTE: doesn't take into account that a combination of the two might actually be more advantageous.
        # Might cause heuristic to be inadmissable in some edge cases.
        items_to_pot_cost = min(items_to_pot_costs)

        heuristic_cost = (pot_to_delivery_costs + dish_to_pot_costs + items_to_pot_cost) / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            print("\n" + "#" * 35)
            print("Current state: (ml timestep {})\n".format(time))

            print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
            ))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
            ))

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost

    
class Steak_Heuristic(Heuristic):
    def __init__(self, mp):
        super().__init__(mp)

    def _calculate_heuristic_costs(self, debug=False):
        """Pre-computes the costs between common trip types for this mdp"""
        pot_locations = self.mdp.get_pot_locations()
        delivery_locations = self.mdp.get_serving_locations()
        dish_locations = self.mdp.get_dish_dispenser_locations()
        onion_locations = self.mdp.get_onion_dispenser_locations()
        meat_locations = self.mdp.get_meat_dispenser_locations()
        board_locations = self.mdp.get_chopping_board_locations()
        sink_locations = self.mdp.get_sink_locations()

        heuristic_cost_dict = {
            'board-delivery': self.motion_planner.min_cost_between_features(board_locations, delivery_locations, manhattan_if_fail=True),
            'sink-pot': self.motion_planner.min_cost_between_features(sink_locations, pot_locations, manhattan_if_fail=True),
            'pot-board': self.motion_planner.min_cost_between_features(pot_locations, board_locations, manhattan_if_fail=True)
        }

        meat_pot_cost = self.motion_planner.min_cost_between_features(meat_locations, pot_locations, manhattan_if_fail=True)
        onion_board_cost = self.motion_planner.min_cost_between_features(onion_locations, board_locations, manhattan_if_fail=True)
        plate_sink_cost = self.motion_planner.min_cost_between_features(dish_locations, sink_locations, manhattan_if_fail=True)


        if debug: print("Heuristic cost dict", heuristic_cost_dict)
        assert meat_pot_cost != np.inf or onion_board_cost != np.inf or plate_sink_cost != np.inf
        if meat_pot_cost != np.inf:
            heuristic_cost_dict['meat-pot'] = meat_pot_cost
        if onion_board_cost != np.inf:
            heuristic_cost_dict['onion-board'] = onion_board_cost
        if plate_sink_cost != np.inf:
            heuristic_cost_dict['plate-sink'] = plate_sink_cost
        
        return heuristic_cost_dict
    
    def simple_heuristic(self, state, time=0, debug=False):
        """Simpler heuristic that tends to run faster than current one"""
        # NOTE: State should be modified to have an order list w.r.t. which
        # one can calculate the heuristic
        assert state.order_list is not None
        
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        board_states_dict = self.mdp.get_chopping_board_status(state)
        sink_states_dict = self.mdp.get_sink_status(state)
        num_deliveries_to_go = state.num_orders_remaining
        
        steak_in_pots = pot_states_dict['steak']['cooking'] + pot_states_dict['steak']['ready']
        onion_on_boards = board_states_dict['ready']
        plate_in_sinks = sink_states_dict['ready']

        partially_chopped_onion = board_states_dict['full']
        partially_heated_plate = sink_states_dict['full']
        num_chops_on_board = sum([state.get_object(loc).state for loc in partially_chopped_onion])
        num_heat_in_sink = sum([state.get_object(loc).state for loc in partially_heated_plate])

        dishes_in_transit = player_objects['dish']
        hot_plates_in_transit = objects_dict['hot_plate'] + player_objects['hot_plate']
        steak_in_transit = objects_dict['steak'] + player_objects['steak']
        onions_in_transit = objects_dict['onion'] + player_objects['onion']
        plates_in_transit = objects_dict['plate'] + player_objects['plate']
        meats_in_transit = objects_dict['meat'] + player_objects['meat']

        num_dish_to_delivery = max([0, num_deliveries_to_go - len(dishes_in_transit)])
        num_hot_plate_to_pot = max([0, num_dish_to_delivery - len(hot_plates_in_transit)])
        num_steak_to_board = max([0, num_dish_to_delivery - len(steak_in_transit)])

        num_pots_to_be_filled = num_dish_to_delivery - len(steak_in_pots)
        num_board_to_be_filled = num_dish_to_delivery - len(onion_on_boards)
        num_sink_to_be_filled = num_dish_to_delivery - len(plate_in_sinks)

        num_meat_needed = num_pots_to_be_filled - len(meats_in_transit)
        num_onion_needed = num_board_to_be_filled - len(onions_in_transit)
        num_plate_needed = num_sink_to_be_filled - len(plates_in_transit)
        
        num_chops_needed = num_board_to_be_filled * self.mdp.chopping_time - len(onions_in_transit) - num_chops_on_board
        num_heat_needed = num_sink_to_be_filled * self.mdp.wash_time - len(plates_in_transit) - num_heat_in_sink
        
        num_chops_to_garnish = max([0, num_chops_needed])
        num_heats_to_hot_plate = max([0, num_heat_needed])
        interaction_costs = num_chops_to_garnish + num_heats_to_hot_plate

        num_meat_to_pot = max([0, num_meat_needed])
        num_onion_to_board = max([0, num_onion_needed])
        num_plate_to_sink = max([0, num_plate_needed])

        pot_to_delivery_costs = self.heuristic_cost_dict['board-delivery'] * num_dish_to_delivery
        hot_plate_to_pot_costs = self.heuristic_cost_dict['sink-pot'] * num_hot_plate_to_pot
        steak_to_board_costs = self.heuristic_cost_dict['pot-board'] * num_steak_to_board
        prep_dish_costs = pot_to_delivery_costs + hot_plate_to_pot_costs + steak_to_board_costs

        items_to_pot_costs = []
        if 'onion-board' in self.heuristic_cost_dict.keys():
            onion_to_board_costs = self.heuristic_cost_dict['onion-board'] * num_onion_to_board
            items_to_pot_costs.append(onion_to_board_costs)
        if 'plate-sink' in self.heuristic_cost_dict.keys():
            plate_to_sink_costs = self.heuristic_cost_dict['plate-sink'] * num_plate_to_sink
            items_to_pot_costs.append(plate_to_sink_costs)
        if 'steak-pot' in self.heuristic_cost_dict.keys():
            meat_to_pot_costs = self.heuristic_cost_dict['meat-pot'] * num_meat_to_pot
            items_to_pot_costs.append(meat_to_pot_costs)

        items_to_pot_cost = sum(items_to_pot_costs)

        heuristic_cost = (interaction_costs + prep_dish_costs + items_to_pot_cost) / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            print("\n" + "#" * 35)
            print("Current state: (ml timestep {})\n".format(time))

            print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
            ))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
            ))

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost

class MediumLevelMdpPlanner(object):

    def __init__(self, mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8, jmp=None):

        self.mdp = mdp
        self.params = mlp_params
        self.jmp = JointMotionPlanner(mdp, mlp_params) if jmp is None else jmp
        self.mp = self.jmp.motion_planner

        self.state_idx_dict = state_idx_dict
        self.state_dict = state_dict
        self.action_dict = action_dict
        self.action_idx_dict = action_idx_dict
        # set states as 'player's object + medium level actions (get, place, deliver, put in pot)

        self.num_joint_action = (Action.NUM_ACTIONS)# * Action.NUM_ACTIONS)
        self.num_states = len(state_idx_dict)
        self.num_actions = len(action_idx_dict)
        self.num_rounds = num_rounds
        self.planner_name = 'mdp'
        self.agent_index = 0

        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.policy_matrix = policy_matrix
        self.value_matrix = value_matrix
        self.epsilon = epsilon
        self.discount = discount
        self.q = None

    @staticmethod
    def from_mdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            mdp_planner = pickle.load(f)
            mdp = mdp_planner[0]
            params = mdp_planner[1]

            state_idx_dict = mdp_planner[2]
            state_dict = mdp_planner[3]

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            policy_matrix = mdp_planner[4]
            # value_matrix = mdp_planner.value_matrix
            
            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds
            
            return MediumLevelMdpPlanner(mdp, params, mlp_action_manager, state_dict, state_idx_dict, policy_matrix=policy_matrix)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'medium_mdp' + '.pkl'

        if force_compute_all:
            mdp_planner = MediumLevelMdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner
        
        try:
            mdp_planner = MediumLevelMdpPlanner.from_mdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp_policy(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = MediumLevelMdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner

        if info:
            print("Loaded MediumMdpPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner

    def save_policy_to_file(self, filename):
        with open(filename, 'wb') as output:
            mdp_plan = [self.mdp, self.params, self.state_idx_dict, self.state_dict, self.action_idx_dict, self.action_dict, self.transition_matrix, self.policy_matrix]
            pickle.dump(mdp_plan, output, pickle.HIGHEST_PROTOCOL)

    def gen_state_dict_key(self, state, player, soup_finish, other_player=None):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)

        player_obj = None
        if player.held_object is not None:
            player_obj = player.held_object.name

        order_str = None if state.order_list is None else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + '_' + str(order)

        state_str = str(player_obj)+'_'+str(soup_finish)+'_'+ order_str

        return state_str

    def init_states(self, state_idx_dict=None, order_list=None):
        # print('In init_states()...')
        # player_obj, num_item_in_pot, order_list

        if state_idx_dict is None:
            objects = ['onion', 'soup', 'dish', 'None'] # 'tomato'
            # common_actions = ['pickup', 'drop']
            # addition_actions = [('soup','deliver'), ('soup', 'pickup'), ('dish', 'pickup'), ('None', 'None')]
            # obj_action_pair = list(itertools.product(objects, common_actions)) + addition_actions

            state_keys = []; state_obj = []; tmp_state_obj = []; tmp_state_obj_1 = []

            for obj in objects:
                tmp_state_obj.append(([obj]))

            # include number of item in soup
            objects_only_arr = [obj.copy() for obj in tmp_state_obj]
            for i in range(self.mdp.num_items_for_soup+1):
                tmp_keys = [val+'_'+str(i) for val in objects]
                for obj in tmp_state_obj:
                    obj.append(i)

                state_keys = state_keys + tmp_keys
                state_obj = state_obj + tmp_state_obj
                tmp_state_obj = [obj.copy() for obj in objects_only_arr]

            tmp_state_key = state_keys
            tmp_state_obj = [obj.copy() for obj in state_obj]

            # include order list items in state

            for order in order_list:
                prev_keys = tmp_state_key.copy()
                tmp_keys = [i+'_'+order for i in prev_keys]
                state_keys = state_keys + tmp_keys
                tmp_state_key = tmp_keys

                for obj in tmp_state_obj:
                    obj.append(order)
                state_obj = state_obj + [obj.copy() for obj in tmp_state_obj]

            # print(state_keys, state_obj)

            self.state_idx_dict = {k:i for i, k in enumerate(state_keys)}
            self.state_dict = {key:obj for key, obj in zip(state_keys, state_obj)} 

        else:
            self.state_idx_dict = state_idx_dict
            self.state_dict = state_dict

        # print('Initialize states:', self.state_idx_dict.items())
        return

    def init_actions(self, actions=None, action_dict=None, action_idx_dict=None):
        '''
        action_dict = {'pickup_onion': ['pickup', 'onion'], 'pickup_dish': ['pickup', 'dish'], 'drop_onion': ['drop', 'onion'], 'drop_dish': ['drop', 'dish'], 'deliver_soup': ['deliver', 'soup'], 'pickup_soup': ['pickup', 'soup']}
        '''
        # print('In init_actions()...')

        if actions is None:
            objects = ['onion', 'dish'] # 'tomato'
            common_actions = ['pickup', 'drop']
            addition_actions = [['deliver','soup'], ['pickup', 'soup']]

            common_action_obj_pair = list(itertools.product(common_actions, objects))
            common_action_obj_pair = [list(i) for i in common_action_obj_pair]
            actions = common_action_obj_pair + addition_actions
            self.action_dict = {action[0]+'_'+action[1]:action for action in actions}
            self.action_idx_dict = {action[0]+'_'+action[1]:i for i, action in enumerate(actions)}

        else:
            self.action_dict = action_dict
            self.action_idx_dict = action_idx_dict

        # print('Initialize actions:', self.action_dict)
        
        return

    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        game_logic_transition = self.transition_matrix.copy()
        distance_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]
                next_state_idx = state_idx
                next_action_idx = action_idx
        
                # define state and action game transition logic
                player_obj, soup_finish, orders = self.ml_state_to_objs(state_obj)
                next_actions, next_state_keys = self.state_action_nxt_state(player_obj, soup_finish, orders)

                if next_actions == action_key:
                    next_state_idx = self.state_idx_dict[next_state_keys]

                game_logic_transition[next_action_idx][state_idx][next_state_idx] += 1.0

            # print(state_key)
        # print(game_logic_transition[:, 25])
        # tmp = input()

        self.transition_matrix = game_logic_transition

    def ml_state_to_objs(self, state_obj):
        # state: obj + action + bool(soup nearly finish) + orders
        player_obj = state_obj[0]; soup_finish = state_obj[1];
        orders = []
        if len(state_obj) > 2:
            orders = state_obj[2:]

        return player_obj, soup_finish, orders
        
    def state_action_nxt_state(self, player_obj, soup_finish, orders, other_obj=''):
        # game logic
        actions = ''; next_obj = player_obj; next_soup_finish = soup_finish
        if player_obj == 'None':
            if (soup_finish == self.mdp.num_items_for_soup) and (other_obj != 'dish'):
                actions = 'pickup_dish'
                next_obj = 'dish'
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == 'onion':
                    actions = 'pickup_onion'
                    next_obj = 'onion'

                elif next_order == 'tomato':
                    actions = 'pickup_tomato' 
                    next_obj = 'tomato'

                else:
                    actions = 'pickup_onion'
                    next_obj = 'onion'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                next_soup_finish += 1

            elif player_obj == 'tomato':
                actions = 'drop_tomato'
                next_obj = 'None'
                next_soup_finish += 1

            elif (player_obj == 'dish') and (soup_finish == self.mdp.num_items_for_soup):
                actions = 'pickup_soup'
                next_obj = 'soup'
                next_soup_finish = 0

            elif (player_obj == 'dish') and (soup_finish != self.mdp.num_items_for_soup):
                actions = 'drop_dish'
                next_obj = 'None'

            elif player_obj == 'soup':
                actions = 'deliver_soup'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        next_state_keys = next_obj + '_' + str(next_soup_finish)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        return actions, next_state_keys

    def state_transition_by_distance(self, curr_state, next_state, action):
        action_taken = action[0]; action_obj = action[1]
        curr_state_obj = curr_state[0]; curr_state_action = curr_state[1]
        next_state_obj = next_state[0]; next_state_action = next_state[1]

        # location depends on the action and object in hand
        curr_location = self.map_action_to_location(curr_state_action, curr_state_obj)
        next_location = self.map_action_to_location(action_taken, action_obj)

        # calculate distance between locations
        min_distance = self.mp.min_cost_between_features(curr_location, next_location)

        return 1.0 / min_distance

    def drop_item(self, state):
        return self.mdp.get_empty_counter_locations(state)

    def map_action_to_location(self, world_state, state_obj, action, obj, p0_obj=None):
        """
        Get the next location the agent will be in based on current world state and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else self.state_dict[state_obj][0]
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []
        if action == 'pickup' and obj != 'soup':
            if p0_obj != 'None':
                location = self.drop_item(world_state)
            else:
                if obj == 'onion':
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == 'tomato':
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == 'dish':
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    print(p0_obj, action, obj)
                    ValueError()
        elif action == 'pickup' and obj == 'soup':
            if p0_obj != 'dish' and p0_obj != 'None':
                location = self.drop_item(world_state)
            elif p0_obj == 'None':
                location = self.mdp.get_dish_dispenser_locations()
            else:
                location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

        elif action == 'drop':
            if obj == 'onion' or obj == 'tomato':
                location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
            elif obj == 'dish':
                location = self.drop_item(world_state)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'deliver':
            if p0_obj != 'soup':
                location = self.mdp.get_empty_counter_locations(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location

    def map_action_to_state_location(self, state, state_str, action, obj, world_info):
        pots_states_dict = self.mdp.get_pot_states(world_info)
        location = []
        if action == 'pickup' and obj != 'soup':
            if not self._not_holding_object(state_str):
                location = self.drop_item(state)
            else:
                if obj == 'onion':
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == 'tomato':
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == 'dish':
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    ValueError()
        elif action == 'pickup' and obj == 'soup':
            if self.state_dict[state_str][0] != 'dish' and not self._not_holding_object(state_str):
                location = self.drop_item(state)
            elif self._not_holding_object(state_str):
                location = self.mdp.get_dish_dispenser_locations()
            else:
                location = self.mdp.get_ready_pots(self.mdp.get_pot_states(state)) + self.mdp.get_cooking_pots(self.mdp.get_pot_states(state)) + self.mdp.get_full_pots(self.mdp.get_pot_states(state))

        elif action == 'drop':
            if obj == 'onion' or obj == 'tomato':
                location = self.mdp.get_pot_locations()
            else:
                ValueError()

        elif action == 'deliver':
            if self.state_dict[state_str][0] != 'soup':
                location = self.mdp.get_empty_counter_locations(state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            ValueError()

        return location

    def _not_holding_object(self, state_obj):
        return self.state_dict[state_obj][0] == 'None'

    def init_reward(self, reward_matrix=None):
        # state: obj + action + bool(soup nearly finish) + orders

        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict)), dtype=float)

        # when deliver order, pickup onion. probabily checking the change in states to give out rewards: if action is correct, curr_state acts and changes to rewardable next state. Then, we reward.

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj = state_obj[0]; soup_finish = state_obj[1]
            orders = []
            if len(state_obj) > 2:
                orders = state_obj[2:]

            if player_obj == 'soup':
                self.reward_matrix[self.action_idx_dict['deliver_soup']][self.state_idx_dict[state_key]] += self.mdp.delivery_reward
        
            if len(orders) == 0:
                self.reward_matrix[:,self.state_idx_dict[state_key]] += self.mdp.delivery_reward

            # if soup_finish == self.mdp.num_items_for_soup and player_obj == 'dish':
            #     self.reward_matrix[self.action_idx_dict['pickup_soup'], self.state_idx_dict[state_key]] += self.mdp.delivery_reward/5.0

    def bellman_operator(self, V=None):

        if V is None:
            V = self.value_matrix

        Q = np.zeros((self.num_actions, self.num_states))
        for a in range(self.num_actions):
            # print(self.transition_matrix[a].dot(V))
            Q[a] = self.reward_matrix[a] + self.discount * self.transition_matrix[a].dot(V)

        return Q.max(axis=0), Q.argmax(axis=0)

    @staticmethod
    def get_span(arr):
        # print('in get span arr.max():', arr.max(), ' - arr.min():', arr.min(), ' = ', (arr.max()-arr.min()))
        return arr.max()-arr.min()

    def log_value_iter(self, iter_count):
        self.num_rounds = iter_count
        output_filename = self.mdp.layout_name+'_'+self.planner_name+'_'+str(self.num_rounds)+".pkl"
        output_mdp_path = os.path.join(PLANNERS_DIR, output_filename)
        self.save_policy_to_file(output_mdp_path)

        return

    def value_iteration(self, value_matrix=None):
        self.value_matrix = value_matrix if value_matrix is not None else np.zeros((self.num_states), dtype=float)
        self.policy_matrix = value_matrix if value_matrix is not None else np.zeros((self.num_states), dtype=float)

        # computation of threshold of variation for V for an epsilon-optimal policy
        if self.discount < 1.0:
            thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            thresh = self.epsilon

        iter_count = 0
        while True:
            V_prev = self.value_matrix.copy()

            self.value_matrix, self.policy_matrix = self.bellman_operator()

            variation = self.get_span(self.value_matrix-V_prev)
            # print(self.value_matrix)
            # print('Variation =',  variation, ', Threshold =', thresh)

            if variation < thresh:
                self.log_value_iter(iter_count)
                break
            elif iter_count % LOGUNIT == 0:
                self.log_value_iter(iter_count)
            else:
                pass
            
            iter_count += 1

        return

    def save_to_file(self, filename):
        print("In save_to_file")
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def init_mdp(self):
        self.init_states(order_list=self.mdp.start_order_list)
        self.init_actions()
        self.init_transition_matrix()
        self.init_reward()

    def compute_mdp_policy(self, filename):
        start_time = time.time()

        final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # print('Total states =', self.num_states, '; Total actions =', self.num_actions)

        self.value_iteration()

        # print("Policy Probability Distribution = ")
        # print(self.policy_matrix.tolist(), '\n')
        # print(self.policy_matrix.shape)

        # print("without GPU:", timer()-start)
        print("It took {} seconds to create MediumLevelMdpPlanner".format(time.time() - start_time))
        # self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return 


class SteakMediumLevelMDPPlanner(MediumLevelMdpPlanner):
    def __init__(self, mdp, mlp_params, state_dict={}, state_idx_dict={}, action_dict={}, action_idx_dict={}, transition_matrix=None, reward_matrix=None, policy_matrix=None, value_matrix=None, num_states=0, num_rounds=0, epsilon=0.01, discount=0.8, jmp=None):
        super().__init__(mdp, mlp_params, state_dict, state_idx_dict, action_dict, action_idx_dict, transition_matrix, reward_matrix, policy_matrix, value_matrix, num_states, num_rounds, epsilon, discount, jmp=jmp)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'steak_medium_mdp' + '.pkl'

        if force_compute_all:
            mdp_planner = SteakMediumLevelMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner
        
        try:
            mdp_planner = SteakMediumLevelMDPPlanner.from_mdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp_policy(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = SteakMediumLevelMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner

        if info:
            print("Loaded SteakMediumLevelMDPPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner
    
    def gen_state_dict_key(self, state, player, other_player=None, RETURN_OBJ=False):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, pot_state, chop_state, sink_state, len(order_list)

        player_obj = 'None'
        if player.held_object is not None:
            player_obj = player.held_object.name

        order_str = '' if len(state.order_list) == 0 else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + '_' + str(order)

        pot_state, chop_state, sink_state = 0, -1, -1

        for obj in state.objects.values():
            if obj.name == 'hot_plate' and obj.position in self.mdp.get_sink_locations():
                wash_time = obj.state
                if wash_time > sink_state:
                    sink_state = wash_time
            elif obj.name == 'steak' and obj.position in self.mdp.get_pot_locations():
                _, _, cook_time = obj.state
                if cook_time > 0:
                    pot_state = 1
            elif obj.name == 'garnish' and obj.position in self.mdp.get_chopping_board_locations():
                chop_time = obj.state
                if chop_time > chop_state:
                    chop_state = chop_time

        if chop_state < 0:
            chop_state = 'None'
        if sink_state < 0:
            sink_state = 'None'
            
        if RETURN_OBJ:
            return [player_obj, pot_state, chop_state, sink_state, state.order_list]

        state_str = str(player_obj)+'_'+str(pot_state)+'_'+str(chop_state)+'_'+str(sink_state)
        if order_str != '':
            state_str = state_str + '_' + order_str

        return state_str
    
    def init_states(self, state_idx_dict=None, order_list=None):
        # print('In init_states()...')
        # player_obj, num_items_for_steak, chop_time, sink_time, order_list

        if state_idx_dict is None:
            objects = ['meat', 'onion', 'plate', 'hot_plate', 'steak', 'dish', 'None']
            # common_actions = ['pickup', 'drop']
            # addition_actions = [('soup','deliver'), ('soup', 'pickup'), ('dish', 'pickup'), ('None', 'None')]
            # obj_action_pair = list(itertools.product(objects, common_actions)) + addition_actions

            state_keys = []; state_obj = []; tmp_state_obj = []; tmp_state_obj_1 = []

            for obj in objects:
                tmp_state_obj.append(([obj]))

            # include key object state 
            objects_only_arr = [obj.copy() for obj in tmp_state_obj]
            for i in range(self.mdp.num_items_for_steak+1):
                tmp_keys = [val+'_'+str(i) for val in objects]
                # for obj in tmp_state_obj:
                #     obj.append(i)

                state_keys = state_keys + tmp_keys
                tmp_state_obj = [obj.copy() for obj in objects_only_arr]
                for obj in tmp_state_obj:
                    obj.append(i)
                state_obj = state_obj + [obj for obj in tmp_state_obj]
                
            tmp_state_key = state_keys
            prev_state_obj = [obj.copy() for obj in state_obj]
            tmp_state_obj = [obj.copy() for obj in state_obj]
            prev_keys = tmp_state_key.copy()
            tmp_state_key = []
            state_obj = []

            for i in range(self.mdp.chopping_time+1):
                tmp_keys = [k+'_'+str(i) for k in prev_keys]
                tmp_state_key += tmp_keys

                for obj in tmp_state_obj:
                    obj.append(i)
                state_obj = state_obj + [obj for obj in tmp_state_obj]
                tmp_state_obj = [obj.copy() for obj in prev_state_obj]

            tmp_keys = [k+'_None' for k in prev_keys]
            tmp_state_key += tmp_keys
            # state_keys = tmp_state_key.copy()
            prev_keys = tmp_state_key.copy()

            for obj in tmp_state_obj:
                obj.append('None')
            state_obj = state_obj + [obj for obj in tmp_state_obj]

            prev_state_obj = [obj.copy() for obj in state_obj]
            tmp_state_obj = [obj.copy() for obj in state_obj]
            prev_keys = tmp_state_key.copy()
            tmp_state_key = []
            state_obj = []

            for i in range(self.mdp.wash_time+1):
                tmp_keys = [k+'_'+str(i) for k in prev_keys]
                # state_keys = state_keys + tmp_keys
                tmp_state_key += tmp_keys

                for obj in tmp_state_obj:
                    obj.append(i)
                state_obj = state_obj + [obj for obj in tmp_state_obj]
                tmp_state_obj = [obj.copy() for obj in prev_state_obj]

            tmp_keys = [k+'_None' for k in prev_keys]
            # state_keys = state_keys + tmp_keys
            tmp_state_key += tmp_keys
            # prev_keys = tmp_state_key.copy()
            # tmp_state_key = []
            for obj in tmp_state_obj:
                obj.append('None')
            state_obj = state_obj + [obj for obj in tmp_state_obj]
            # tmp_state_key = state_keys
            prev_state_obj = [obj.copy() for obj in state_obj]
            tmp_state_obj = [obj.copy() for obj in state_obj]
            state_obj = [] 

            # include order list items in state

            for order in order_list:
                prev_keys = tmp_state_key.copy()
                tmp_keys = [i+'_'+order for i in prev_keys]
                # state_keys = state_keys + tmp_keys
                tmp_state_key += tmp_keys

                for obj in tmp_state_obj:
                    obj.append(order)
                tmp_state_obj = prev_state_obj + [obj for obj in tmp_state_obj]
                prev_state_obj = [obj.copy() for obj in tmp_state_obj]

            # print(state_keys, state_obj)

            self.state_idx_dict = {k:i for i, k in enumerate(tmp_state_key)}
            self.state_dict = {key:obj for key, obj in zip(tmp_state_key, tmp_state_obj)} 

        else:
            self.state_idx_dict = state_idx_dict
            self.state_dict = state_dict

        # print('Initialize states:', self.state_idx_dict.items())
        return
    
    def init_actions(self, actions=None, action_dict=None, action_idx_dict=None):
        # print('In init_actions()...')

        if actions is None:
            objects = ['meat', 'onion', 'plate', 'hot_plate', 'steak']
            common_actions = ['pickup', 'drop']
            addition_actions = [['chop', 'onion'], ['heat', 'hot_plate'], ['pickup', 'garnish'], ['deliver','dish']]

            common_action_obj_pair = list(itertools.product(common_actions, objects))
            common_action_obj_pair = [list(i) for i in common_action_obj_pair]
            actions = common_action_obj_pair + addition_actions
            self.action_dict = {action[0]+'_'+action[1]:action for action in actions}
            self.action_idx_dict = {action[0]+'_'+action[1]:i for i, action in enumerate(actions)}

        else:
            self.action_dict = action_dict
            self.action_idx_dict = action_idx_dict

        # print('Initialize actions:', self.action_dict)
        
        return
    
    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        game_logic_transition = self.transition_matrix.copy()
        distance_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]
                next_state_idx = state_idx
                next_action_idx = action_idx
        
                # define state and action game transition logic
                player_obj, num_item_in_pot, chop_time, wash_time, orders = self.ml_state_to_objs(state_obj)
                next_actions, next_state_keys = self.state_action_nxt_state(player_obj, num_item_in_pot, chop_time, wash_time, orders)

                if next_actions == action_key:
                    next_state_idx = self.state_idx_dict[next_state_keys]

                game_logic_transition[next_action_idx][state_idx][next_state_idx] += 1.0

            # print(state_key)
        # print(game_logic_transition[:, 25])
        # tmp = input()

        self.transition_matrix = game_logic_transition

    def ml_state_to_objs(self, state_obj):
        # state: obj + action + bool(soup nearly finish) + orders
        player_obj = state_obj[0]; num_item_in_pot = state_obj[1]; chop_time = state_obj[2]; wash_time = state_obj[3];
        orders = []
        if len(state_obj) > 4:
            orders = state_obj[4:]

        return player_obj, num_item_in_pot, chop_time, wash_time, orders
        
    def state_action_nxt_state(self, player_obj, num_item_in_pot, chop_time, wash_time, orders, other_obj=''):
        # game logic
        actions = ''; next_obj = player_obj; next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time;
        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (other_obj != 'meat'):
                actions = 'pickup_meat'
                next_obj = 'meat'
            elif (chop_time < 0) and (other_obj != 'onion'):
                actions = 'pickup_onion'
                next_obj = 'onion'
            elif (chop_time > 0) and (chop_time < self.mdp.chopping_time) and (wash_time < self.mdp.wash_time):
                actions = 'chop_onion'
                next_obj = 'None'
            elif (wash_time < 0) and (other_obj != 'plate'):
                actions = 'pickup_plate'
                next_obj = 'plate'
            elif (wash_time > 0) and (wash_time < self.mdp.wash_time):
                actions = 'heat_hot_plate'
                next_obj = 'None'
            elif (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
                actions = 'pickup_hot_plate'
                next_obj = 'hot_plate'
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == 'steak':
                    actions = 'pickup_meat'
                    next_obj = 'meat'

                else:
                    actions = 'pickup_meat'
                    next_obj = 'meat'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                next_chop_time = 0

            elif player_obj == 'meat':
                actions = 'drop_meat'
                next_obj = 'None'
                next_num_item_in_pot = 1

            elif player_obj == 'plate':
                actions = 'drop_plate'
                next_obj = 'None'
                next_wash_time = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions = 'pickup_steak'
                next_obj = 'steak'
                next_num_item_in_pot = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions = 'drop_hot_plate'
                next_obj = 'None'
                next_wash_time = 'None'

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = 'None'

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                actions = 'drop_steak'
                next_obj = 'None'

            elif player_obj == 'dish':
                actions = 'deliver_dish'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj)
                raise ValueError()

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        return actions, next_state_keys
    
    def map_action_to_location(self, world_state, state_obj, action, obj, p0_obj=None):
        """
        Get the next location the agent will be in based on current world state and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else self.state_dict[state_obj][0]
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []
        if action == 'pickup':
            if obj == 'onion':
                location = self.mdp.get_onion_dispenser_locations()
            elif obj == 'plate':
                location = self.mdp.get_dish_dispens()
            elif obj == 'meat':
                location = self.mdp.get_meat_dispenser_locations()
            elif obj == 'hot_plate':
                location = self.mdp.get_sink_status(world_state)['full'] + self.mdp.get_sink_status(world_state)['ready']
            elif obj == 'garish':
                location = self.mdp.get_chopping_board_status(world_state)['full'] + self.mdp.get_chopping_board_status(world_state)['ready']
            elif obj == 'steak':
                location = self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'drop':
            if obj == 'meat':
                location = self.mdp.get_empty_pots(pots_states_dict)
            elif obj == 'onion':
                location = self.mdp.get_chopping_board_status(world_state)['empty']
            elif obj == 'plate':
                location = self.mdp.get_sink_status(world_state)['empty']
            elif obj == 'hot_plate' or obj == 'steak':
                location = self.drop_item(world_state)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'deliver':
            if p0_obj != 'dish':
                location = self.drop_item(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location
    
    def init_reward(self, reward_matrix=None):
        # state: obj + action + bool(soup nearly finish) + orders

        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict)), dtype=float)

        # when deliver order, pickup onion. probabily checking the change in states to give out rewards: if action is correct, curr_state acts and changes to rewardable next state. Then, we reward.

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj = state_obj[0]; soup_finish = state_obj[1]
            orders = []
            if len(state_obj) > 4:
                orders = state_obj[4:]

            if player_obj == 'soup':
                self.reward_matrix[self.action_idx_dict['deliver_dish']][self.state_idx_dict[state_key]] += self.mdp.delivery_reward
        
            if len(orders) == 0:
                self.reward_matrix[:,self.state_idx_dict[state_key]] += self.mdp.delivery_reward

            # if soup_finish == self.mdp.num_items_for_soup and player_obj == 'dish':
            #     self.reward_matrix[self.action_idx_dict['pickup_soup'], self.state_idx_dict[state_key]] += self.mdp.delivery_reward/5.0

    

class HumanAwareMediumMDPPlanner(MediumLevelMdpPlanner):
    """docstring for HumanAwareMediumMDPPlanner"""
    def __init__(self, mdp, mlp_params, hmlp, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8):

        super().__init__(mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8)

        self.hmlp = hmlp

    @staticmethod
    def from_mdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            mdp_planner = pickle.load(f)
            mdp = mdp_planner[0]
            params = mdp_planner[1]
            mlp_action_manager = mdp_planner[2]
            
            state_idx_dict = mdp_planner[3]
            state_dict = mdp_planner[4]

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            policy_matrix = mdp_planner[5]
            # value_matrix = mdp_planner.value_matrix
            
            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds
            
            return HumanAwareMediumMDPPlanner(mdp, params, mlp_action_manager, state_dict, state_idx_dict, policy_matrix=policy_matrix)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, hmlp, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'human_aware_medium_mdp' + '.pkl'

        if force_compute_all:
            mdp_planner = HumanAwareMediumMDPPlanner(mdp, mlp_params, hmlp)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner
        
        try:
            mdp_planner = HumanAwareMediumMDPPlanner.from_mdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp_policy(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = HumanAwareMediumMDPPlanner(mdp, mlp_params, hmlp)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner

        if info:
            print("Loaded HumanAwareMediumMDPPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner
    
    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        # print('In init_human_aware_states()')
        self.init_states(order_list=order_list)

        # add p1_obj to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'tomato', 'soup', 'dish', 'None']
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for ori_key, ori_value in original_state_dict.items():
                new_key = ori_key+'_'+obj
                # if i == 0:
                #     new_obj = original_state_dict[ori_key]+[obj] # update key
                #     self.state_dict[new_key] = new_obj # update value
                #     self.state_idx_dict[new_key] = original_state_dict[ori_key] # update key
                # else:
                new_obj = original_state_dict[ori_key]+[obj] 
                self.state_dict[new_key] = new_obj # update value
                self.state_idx_dict[new_key] = len(self.state_idx_dict)

    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        game_logic_transition = self.transition_matrix.copy()
        distance_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]
                
                # define state and action game transition logic
                p0_state, p1_obj = self.extract_p0(state_obj)
                
                # get next step p1 object
                p1_nxt_states = []
                if len(p0_state) > 2:
                    p1_nxt_states = self.hmlp.get_state_trans(p1_obj, p0_state[1], p0_state[2:])
                else:
                    p1_nxt_states = self.hmlp.get_state_trans(p1_obj, p0_state[1], [])

                for [p1_nxt_obj, aft_p1_num_item_in_pot, aft_p1_order_list, p1_trans_prob, p1_pref_prob] in p1_nxt_states:

                    # print([p1_nxt_obj, aft_p1_num_item_in_pot, aft_p1_order_list, p1_trans_prob, p1_pref_prob])

                    p0_obj, soup_finish, orders = self.ml_state_to_objs(p0_state)
                    p0_ori_key = p0_state[0]
                    for s in p0_state[1:]:
                        p0_ori_key = p0_ori_key + '_' + str(s)

                    # get next step p0 object based on p1 original
                    p1_ori_action, p0_nxt_p1_ori_p0_key = self.state_action_nxt_state(p0_obj, soup_finish, orders, p1_obj)

                    # print(p1_ori_action, p0_nxt_p1_ori_p0_key)

                    if action_key == p1_ori_action: # p0 nxt based on p1_ori
                        p0_nxt_p1_ori_nxt_idx = self.state_idx_dict[p0_nxt_p1_ori_p0_key + '_' + p1_obj]
                        game_logic_transition[action_idx][state_idx][p0_nxt_p1_ori_nxt_idx] += (1.0 - p1_trans_prob) * p1_pref_prob

                        # print(state_key,'--', action_key, '-->', p0_nxt_p1_ori_p0_key + '_' + p1_obj)

                    else: # p0_ori + p1_ori
                        p0_ori_p1_ori_nxt_idx = self.state_idx_dict[p0_ori_key + '_' + p1_obj]
                        game_logic_transition[action_idx][state_idx][p0_ori_p1_ori_nxt_idx] += (1.0 - p1_trans_prob) * p1_pref_prob

                        # print(state_key,'--', action_key, '-->', p0_ori_key + '_' + p1_obj)

                    # get next step p0 object based on p1 next state
                    p1_nxt_action, p0_nxt_p1_nxt_p0_key = self.state_action_nxt_state(p0_obj, aft_p1_num_item_in_pot, aft_p1_order_list, p1_nxt_obj)

                    # print(p1_nxt_action, p0_nxt_p1_nxt_p0_key, aft_p1_num_item_in_pot, aft_p1_order_list)

                    p1_nxt_key = str(aft_p1_num_item_in_pot)
                    for obj in aft_p1_order_list:
                        p1_nxt_key = p1_nxt_key + '_' + obj
                    p1_nxt_key = p1_nxt_key + '_' + p1_nxt_obj

                    if action_key == p1_nxt_action: # p0 nxt based on p1 next
                        p0_nxt_p1_nxt_nxt_idx= self.state_idx_dict[p0_nxt_p1_nxt_p0_key + '_' + p1_nxt_obj]
                        game_logic_transition[action_idx][state_idx][p0_nxt_p1_nxt_nxt_idx] += p1_trans_prob * p1_pref_prob

                        # print(state_key,'--', action_key, '-->', p0_nxt_p1_nxt_p0_key + '_' + p1_nxt_obj)

                    else: # action not matched; thus, p0 ori based on p1 next
                        # p0_ori_p1_nxt_nxt_idx = self.state_idx_dict[p0_obj + '_' + p1_nxt_key]
                        # game_logic_transition[action_idx][state_idx][p0_ori_p1_nxt_nxt_idx] += p1_trans_prob * p1_pref_prob
                        p0_ori_p1_ori_nxt_idx = self.state_idx_dict[p0_ori_key + '_' + p1_obj]
                        game_logic_transition[action_idx][state_idx][p0_ori_p1_ori_nxt_idx] += (p1_trans_prob) * p1_pref_prob



                        # print(state_key,'--', action_key, '-->', p0_obj + '_' + p1_nxt_key)

                    # print(game_logic_transition[action_idx][state_idx])
                    # tmp = input()

        # print(list(self.state_idx_dict.keys())[list(self.state_idx_dict.values()).index(58)],\
        #     list(self.state_idx_dict.keys())[list(self.state_idx_dict.values()).index(42)],\
        #     list(self.state_idx_dict.keys())[list(self.state_idx_dict.values()).index(267)],\
        #     list(self.state_idx_dict.keys())[list(self.state_idx_dict.values()).index(269)],\
        #     game_logic_transition[:, 42, 267],\
        #     game_logic_transition[:, 42, 269],\
        #     game_logic_transition[:, 42, 42],\
        #     np.sum(game_logic_transition[:, 58], axis=1),\
        #     np.sum(game_logic_transition[:, 42], axis=1),\
        #     game_logic_transition[0, 42],\
        #     game_logic_transition[4, 42])
        # tmp = input()

        self.transition_matrix = game_logic_transition

    def init_reward(self, reward_matrix=None):
        # state: obj + action + bool(soup nearly finish) + orders

        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict)), dtype=float)

        # when deliver order, pickup onion. probabily checking the change in states to give out rewards: if action is correct, curr_state acts and changes to rewardable next state. Then, we reward.

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj, soup_finish, orders = self.ml_state_to_objs(state_obj[:-1])

            if player_obj == 'soup':
                self.reward_matrix[self.action_idx_dict['deliver_soup'], self.state_idx_dict[state_key]] += self.mdp.delivery_reward/10.0

            # if player_obj == 'soup':
            #     self.reward_matrix[self.action_idx_dict['pickup_onion']][self.state_idx_dict[state_key]] -= self.mdp.delivery_reward/10.0
        
            if len(orders) == 0:
                self.reward_matrix[:,self.state_idx_dict[state_key]] += self.mdp.delivery_reward

            # if soup_finish == self.mdp.num_items_for_soup and player_obj == 'dish':
                # self.reward_matrix[self.action_idx_dict['pickup_soup'], self.state_idx_dict[state_key]] += self.mdp.delivery_reward/100.0

        # print(self.reward_matrix[:,42])
        # tmp = input()

    def extract_p0(self, state_obj):
        return state_obj[:-1], state_obj[-1]

    def init_mdp(self):
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_actions()
        self.init_transition_matrix()
        self.init_reward()

    def gen_state_dict_key(self, state, player, soup_finish, other_player):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)

        player_obj = None; other_player_obj = None
        if player.held_object is not None:
            player_obj = player.held_object.name
        if other_player.held_object is not None:
            other_player_obj = other_player.held_object.name

        order_str = None if state.order_list is None else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + '_' + str(order)

        state_str = str(player_obj)+'_'+str(soup_finish)+'_'+ order_str + '_' + str(other_player_obj)

        return state_str


class HumanSubtaskQMDPPlanner(MediumLevelMdpPlanner):
    def __init__(self, mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8):

        super().__init__(mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8)

        self.world_state_cost_dict = {}
        self.jmp = JointMotionPlanner(mdp, mlp_params)
        self.mp = self.jmp.motion_planner
        self.subtask_dict = {}
        self.subtask_idx_dict = {}

    @staticmethod
    def from_qmdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            return pickle.load(f)
            # mdp = mdp_planner.mdp
            # params = mdp_planner.params

            # state_idx_dict = mdp_planner.state_idx_dict
            # state_dict = mdp_planner.state_dict

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            # policy_matrix = mdp_planner.policy_matrix
            # value_matrix = mdp_planner.value_matrix
            
            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds
            
            # return HumanSubtaskQMDPPlanner(mdp, params, mlp_action_manager, state_dict, state_idx_dict, policy_matrix=policy_matrix)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'human_subtask_aware_qmdp' + '.pkl'

        if force_compute_all:
            mdp_planner = HumanSubtaskQMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp(filename)
            return mdp_planner
        
        try:
            mdp_planner = HumanSubtaskQMDPPlanner.from_qmdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = HumanSubtaskQMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        if info:
            print("Loaded HumanSubtaskQMDPPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pot, order list, agent 1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list) 

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'soup', 'dish', 'None'] # 'tomato'
        self.subtask_dict = copy.deepcopy(self.action_dict)
        del self.subtask_dict['drop_dish']
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for j, subtask in enumerate(self.subtask_dict.items()):
                self.subtask_idx_dict[subtask[0]] = j
                if self._init_is_valid_object_subtask_pair(obj, subtask[0]):
                    for ori_key, ori_value in original_state_dict.items():
                        new_key = ori_key+'_'+obj + '_' + subtask[0]
                        new_obj = original_state_dict[ori_key]+[obj] + [subtask[0]]
                        self.state_dict[new_key] = new_obj # update value
                        self.state_idx_dict[new_key] = len(self.state_idx_dict)

        # print('subtask dict =', self.subtask_dict)

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                normalize_count = 0
                
                # decode state information
                p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                # calculate next states for p1 (a.k.a. human)
                p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

                # calculate next states for p0 (conditioned on p1 (a.k.a. human))
                for p1_nxt_state in p1_nxt_states:
                    action, next_state_key = self.state_transition(p0_state, p1_nxt_world_info, human_state=p1_nxt_state)
                    # for action, next_state_key in zip(actions, next_state_keys):
                        # print(p0_state, p1_nxt_world_info, p1_nxt_state, action, next_state_keys)
                    if action_key == action:
                        next_state_idx= self.state_idx_dict[next_state_key]
                        self.transition_matrix[action_idx, state_idx, next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def get_successor_states(self, start_world_state, start_state_key, debug=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][2:]) <= 2: # [p0_obj, num_item_in_soup, orders, p1_obj, subtask] 
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        agent_action_idx_arr, next_state_idx_arr = np.where(self.transition_matrix[:, ori_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr, next_state_idx_arr):
            next_world_state, cost = self.mdp_action_state_to_world_state(next_action_idx, next_state_idx, start_world_state)
            successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))
            if debug: print('Action {} from {} to {} costs {} in {} seconds.'.format(self.get_key_from_value(self.action_idx_dict, next_action_idx), self.get_key_from_value(self.state_idx_dict, ori_state_idx), self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time()-start_time))

        return successor_states

    def decode_state_info(self, state_obj):
        return state_obj[0], state_obj[-2:], state_obj[1:-2]

    def _init_is_valid_object_subtask_pair(self, obj, subtask):
        if obj == 'None':
            if subtask == 'pickup_dish':
                return True
            elif subtask == 'pickup_onion':
                return True
            elif subtask == 'pickup_tomato':
                return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'tomato' and subtask == 'drop_tomato':
                return True
            elif (obj == 'dish') and subtask == 'pickup_soup':
                return True
            elif obj == 'soup' and subtask == 'deliver_soup':
                return True
            else:
                return False
        return True

    def _is_valid_object_subtask_pair(self, obj, subtask, soup_finish, greedy=False):
        if obj == 'None':
            if greedy != True and (subtask == 'pickup_dish' or subtask == 'pickup_onion') and soup_finish <= self.mdp.num_items_for_soup:
                return True
            elif greedy == True and subtask == 'pickup_onion' and soup_finish < self.mdp.num_items_for_soup:
                return True
            elif greedy == True and subtask == 'pickup_dish' and soup_finish == self.mdp.num_items_for_soup:
                return True
            elif subtask == 'pickup_tomato':
                return True
            # elif subtask == 'pickup_soup':
            #     return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'tomato' and subtask == 'drop_tomato':
                return True
            elif (obj == 'dish') and subtask == 'pickup_soup':
                return True
            # elif (obj == 'dish') and subtask == 'drop_dish':
            #     return True
            elif obj == 'soup' and subtask == 'deliver_soup':
                return True
            else:
                return False
        return True

    def human_state_subtask_transition(self, human_state, world_info):
        player_obj = human_state[0]; subtask = human_state[1]
        soup_finish = world_info[0]; orders = [] if len(world_info) < 2 else world_info[1:]
        next_obj = player_obj; next_subtasks = []; 
        next_soup_finish = soup_finish;

        if player_obj == 'None':
            if subtask == 'pickup_dish':
                next_obj = 'dish'
                next_subtasks = ['pickup_soup']#, 'drop_dish']

            elif subtask == 'pickup_onion':
                next_obj = 'onion'
                next_subtasks = ['drop_onion']
            
            elif subtask == 'pickup_tomato':
                next_obj = 'tomato'
                next_subtasks = ['drop_tomato']
            
            # elif subtask == 'pickup_soup':
            #     next_obj = 'soup'
            #     next_subtasks = ['deliver_soup']

        else:
            if player_obj == 'onion' and subtask == 'drop_onion' and soup_finish < self.mdp.num_items_for_soup:
                next_obj = 'None'
                next_soup_finish += 1
                next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'
            
            elif player_obj == 'onion' and subtask == 'drop_onion' and soup_finish == self.mdp.num_items_for_soup:
                next_obj = 'onion'
                next_subtasks = ['drop_onion']

            elif player_obj == 'tomato' and subtask == 'drop_tomato':
                next_obj = 'None'
                next_soup_finish += 1
                next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'

            elif (player_obj == 'dish') and subtask == 'pickup_soup':
                next_obj = 'soup'
                next_soup_finish = 0
                next_subtasks = ['deliver_soup']

            # elif (player_obj == 'dish') and subtask == 'drop_dish':
            #     next_obj = 'None'
            #     next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'

            elif player_obj == 'soup' and subtask == 'deliver_soup':
                next_obj = 'None'
                next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj, subtask)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        p1_nxt_states = []
        for next_subtask in next_subtasks:
            p1_nxt_states.append([next_obj, next_subtask])

        nxt_world_info = [next_soup_finish]
        for order in orders:
            nxt_world_info.append(order)

        return p1_nxt_states, nxt_world_info

    def state_transition(self, player_obj, world_info, human_state=[None, None]):
        # game logic
        soup_finish = world_info[0]; orders = [] if len(world_info) < 2 else world_info[1:]
        other_obj = human_state[0]; subtask = human_state[1]
        actions = ''; next_obj = player_obj; next_soup_finish = soup_finish

        if player_obj == 'None':
            if (soup_finish == self.mdp.num_items_for_soup) and (other_obj != 'dish' and subtask != 'pickup_dish'):
                actions = 'pickup_dish'
                next_obj = 'dish'
            elif (soup_finish == (self.mdp.num_items_for_soup-1)) and (other_obj == 'onion' and subtask == 'drop_onion'):
                actions = 'pickup_dish'
                next_obj = 'dish'
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == 'onion':
                    actions = 'pickup_onion'
                    next_obj = 'onion'

                elif next_order == 'tomato':
                    actions = 'pickup_tomato' 
                    next_obj = 'tomato'

                else:
                    actions = 'pickup_onion'
                    next_obj = 'onion'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                next_soup_finish += 1

            elif player_obj == 'tomato':
                actions = 'drop_tomato'
                next_obj = 'None'
                next_soup_finish += 1

            elif (player_obj == 'dish') and (soup_finish >= self.mdp.num_items_for_soup-1):
                actions = 'pickup_soup'
                next_obj = 'soup'
                next_soup_finish = 0

            elif (player_obj == 'dish') and (soup_finish < self.mdp.num_items_for_soup-1):
                actions = 'drop_dish'
                next_obj = 'None'

            elif player_obj == 'soup':
                actions = 'deliver_soup'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        next_state_keys = next_obj + '_' + str(next_soup_finish)

        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        for human_info in human_state:
            next_state_keys = next_state_keys + '_' + human_info

        return actions, next_state_keys

    def world_state_to_mdp_state_key(self, state, player, other_player, subtask):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)

        player_obj = None; other_player_obj = None
        if player.held_object is not None:
            player_obj = player.held_object.name
        if other_player.held_object is not None:
            other_player_obj = other_player.held_object.name

        order_str = None if len(state.order_list) == 0 else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + '_' + str(order)

        num_item_in_pot = 0
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]

        state_strs = str(player_obj)+'_'+str(num_item_in_pot)+'_'+ order_str + '_' + str(other_player_obj) + '_' + subtask

        return state_strs

    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]

    def gen_state_dict_key(self, p0_obj, p1_obj, soup_finish, orders, subtasks):
        # a0 hold, a1 hold, 

        player_obj = p0_obj if p0_obj is not None else 'None'
        other_player_obj = p1_obj if p1_obj is not None else 'None'

        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + '_' + str(order)

        state_strs = []
        for subtask in subtasks:
            state_strs.append(str(player_obj)+'_'+str(soup_finish)+'_'+ order_str + '_' + str(other_player_obj) + '_' + subtask)

        return state_strs

    def get_key_from_value(self, dictionary, state_value):
        try: 
            idx = list(dictionary.values()).index(state_value)
        except ValueError:
            return None
        else:
            return list(dictionary.keys())[idx]

    def map_action_to_location(self, world_state, state_str, action, obj, p0_obj=None, player_idx=None, counter_drop=True, state_dict=None):
        """
        Get the next location the agent will be in based on current world state, medium level actions, after-action state obj.
        """
        state_dict = self.state_dict if state_dict is None else state_dict
        p0_obj = p0_obj if p0_obj is not None else state_dict[state_str][0]
        other_obj = world_state.players[1-player_idx].held_object.name if world_state.players[1-player_idx].held_object is not None else 'None'
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []
        WAIT = False # If wait becomes true, one player has to wait for the other player to finish its current task and its next task

        if action == 'pickup' and obj != 'soup':
            if p0_obj != 'None' and counter_drop:
                location = self.drop_item(world_state)
            else:
                if obj == 'onion':
                    location = self.mdp.get_onion_dispenser_locations() 
                elif obj == 'tomato':
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == 'dish':
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    print(p0_obj, action, obj)
                    ValueError()
        elif action == 'pickup' and obj == 'soup':
            if p0_obj != 'dish' and p0_obj != 'None' and counter_drop:
                location = self.drop_item(world_state)
            elif p0_obj == 'None':
                location = self.mdp.get_dish_dispenser_locations()
            else:
                if state_str is not None:
                    num_item_in_pot = state_dict[state_str][1]
                    if num_item_in_pot == 0:
                        location = self.mdp.get_empty_pots(pots_states_dict)
                        if len(location) > 0: return location, True
                    elif num_item_in_pot > 0 and num_item_in_pot < self.mdp.num_items_for_soup:
                        location = self.mdp.get_partially_full_pots(pots_states_dict)
                        if len(location) > 0: return location, True
                    else:
                        location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                    if len(location) > 0: return location, WAIT

                location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                if len(location) == 0:
                    WAIT = True
                    # location = self.ml_action_manager.go_to_closest_feature_or_counter_to_goal(location)
                    location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                    # location = world_state.players[player_idx].pos_and_or
                    return location, WAIT

        elif action == 'drop':
            if obj == 'onion' or obj == 'tomato':

                if state_str is not None:
                    num_item_in_pot = state_dict[state_str][1]
                    if num_item_in_pot == 0:
                        location = self.mdp.get_empty_pots(pots_states_dict)
                    elif num_item_in_pot > 0 and num_item_in_pot < self.mdp.num_items_for_soup:
                        location = self.mdp.get_partially_full_pots(pots_states_dict)
                    else:
                        location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                        
                    if len(location) > 0: return location, WAIT

                location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                
                if len(location) == 0:
                    if other_obj != 'onion' and other_obj != 'tomato':
                        WAIT = True
                        location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                        # location = world_state.players[player_idx].pos_and_or
                        return location, WAIT
                    elif counter_drop:
                        location = self.drop_item(world_state)
                    else:
                        WAIT = True
                        location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                        # location = world_state.players[player_idx].pos_and_or
                        return location, WAIT

            elif obj == 'dish' and player_idx==0 and counter_drop: # agent_index = 0
                location = self.drop_item(world_state)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'deliver':
            if p0_obj != 'soup' and p0_obj != 'None' and counter_drop:
                location = self.mdp.get_empty_counter_locations(world_state)
            elif p0_obj != 'soup':
                if state_str is not None:
                    num_item_in_pot = state_dict[state_str][1]
                    if num_item_in_pot == 0:
                        location = self.mdp.get_empty_pots(pots_states_dict)
                        if len(location) > 0: return location, True
                    elif num_item_in_pot > 0 and num_item_in_pot < self.mdp.num_items_for_soup:
                        location = self.mdp.get_partially_full_pots(pots_states_dict)
                        if len(location) > 0: return location, True
                    else:
                        location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                    if len(location) > 0: return location, WAIT

                location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                if len(location) == 0:
                    WAIT = True
                    # location = self.ml_action_manager.go_to_closest_feature_or_counter_to_goal(location)
                    location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                    # location = world_state.players[player_idx].pos_and_or
                    return location, WAIT
            else:
                location = self.mdp.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location, WAIT

    def _shift_same_goal_pos(self, new_positions, change_idx):
        
        pos = new_positions[change_idx][0]
        ori = new_positions[change_idx][1]
        new_pos = pos; new_ori = ori
        if self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])
        else:
            print('pos = ', pos)
            ValueError()
        
        new_positions[change_idx] = (new_pos, new_ori)

        return new_positions[0], new_positions[1]

    def mdp_action_state_to_world_state(self, action_idx, ori_state_idx, ori_world_state, with_argmin=False):
        new_world_state = ori_world_state.deepcopy()
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)

        possible_agent_motion_goals, AI_WAIT = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[action][0], self.action_dict[action][1], player_idx=0) 
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[mdp_state_obj[-1]][0], self.action_dict[mdp_state_obj[-1]][1], p0_obj=mdp_state_obj[-2], player_idx=1) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
        # get next position for AI agent
        agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[0].pos_and_or, possible_agent_motion_goals, with_motion_goal=True) # select the feature position that is closest to current player's position in world state
        new_agent_pos = agent_feature_pos if agent_feature_pos is not None else new_world_state.players[0].get_pos_and_or()
        human_cost, human_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[1].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else new_world_state.players[1].get_pos_and_or()
        # print(new_agent_pos, new_human_pos)

        if new_agent_pos == new_human_pos:
            new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos], np.argmax(np.array([agent_cost, human_cost])))
            # print('after shift =', new_agent_pos, new_human_pos)

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].remove_object()
        if mdp_state_obj[0] != 'None' and mdp_state_obj[0] != 'soup':
            new_world_state.players[0].held_object = ObjectState(mdp_state_obj[0], new_agent_pos)
        new_world_state.players[0].update_pos_and_or(new_agent_pos[0], new_agent_pos[1])

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].remove_object()
        if mdp_state_obj[-2] != 'None' and mdp_state_obj[-2] != 'soup':
            new_world_state.players[1].held_object = ObjectState(mdp_state_obj[-2], new_human_pos)
        new_world_state.players[1].update_pos_and_or(new_human_pos[0], new_human_pos[1])

        total_cost = max([agent_cost, human_cost]) # in rss paper is max
        if AI_WAIT or HUMAN_WAIT: # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area length)
            total_cost = agent_cost + human_cost + ((self.mdp.width-1)+(self.mdp.height-1))/2

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost

    def world_to_state_keys(self, world_state, player, other_player, belief):
        mdp_state_keys = []
        for i, b in enumerate(belief):
            mdp_state_key = self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i))
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i)))
        return mdp_state_keys

    def joint_action_cost(self, world_state, goal_pos_and_or, COST_OF_STAY=1):
        joint_action_plan, end_motion_state, plan_costs = self.jmp.get_low_level_action_plan(world_state.players_pos_and_or, goal_pos_and_or, merge_one=True)
        # joint_action_plan, end_state, plan_costs = self.mlp.get_embedded_low_level_action_plan(world_state, goal_pos_and_or, other_agent, other_agent_idx)
        # print('joint_action_plan =', joint_action_plan, '; plan_costs =', plan_costs)

        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0

        num_of_non_stay_actions = len([a for a in joint_action_plan if a[0] != Action.STAY])
        num_of_stay_actions = len([a for a in joint_action_plan if a[0] == Action.STAY])

        return joint_action_plan[0], max(plan_costs)# num_of_non_stay_actions+num_of_stay_actions*COST_OF_STAY # in rss paper is max(plan_costs)

    def step(self, world_state, mdp_state_keys, belief, agent_idx, low_level_action=False):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dic will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        next_state_v = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        action_cost = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        qmdp_q = np.zeros((len(self.action_dict), len(belief)), dtype=float)
        # ml_action_to_low_action = np.zeros()

        # for each subtask, obtain next mdp state but with low level location based on finishing excuting current action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None:
                agent_action_idx_arr, next_mdp_state_idx_arr = np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
                nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr): # action_idx is encoded subtask action
                    # print('action_idx =', action_idx)
                    next_state_idx = next_mdp_state_idx_arr[j]
                    after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx, mdp_state_idx, world_state, with_argmin=True)
                    value_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_state_idx), search_depth=100)
                    joint_action, one_step_cost = self.joint_action_cost(world_state, after_action_world_state.players_pos_and_or)  
                    # print('joint_action =', joint_action, 'one_step_cost =', one_step_cost)
                    # print('Action.ACTION_TO_INDEX[joint_action[agent_idx]] =', Action.ACTION_TO_INDEX[joint_action[agent_idx]])
                    if not low_level_action:
                        # action_idx: are subtask action dictionary index
                        next_state_v[i, action_idx] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, action_idx] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    # print('action_idx =', self.get_key_from_value(self.action_idx_dict, action_idx), '; mdp_state_key =', mdp_state_key, '; next_state_key =', self.get_key_from_value(self.state_idx_dict, next_state_idx))
                    # print('next_state_v =', next_state_v[i])
        # print('action_cost =', action_cost)

        q = self.compute_Q(belief, next_state_v, action_cost)
        # print(q)
        action_idx = self.get_best_action(q)
        # print('get_best_action =', action_idx, '=', self.get_key_from_value(self.action_idx_dict, action_idx))
        # print("It took {} seconds for this step".format(time.time() - start_time))
        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
        return action_idx, self.action_dict[self.get_key_from_value(self.action_idx_dict, action_idx)], low_level_action
    
    def observed(self, world_state):
        # update the observation's pot status by looking at the pot status in the world
        if world_state.objects is not None and len(world_state.objects) > 0:
            for obj_pos, obj_state in world_state.objects.items():
                # print(obj_state)
                if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]

        return [num_item_in_pot]

    def belief_update(self, world_state, agent_player, observed_info, human_player, belief_vector, prev_dist_to_feature, greedy=False):
        """
        Update belief based on both human player's game logic and also it's current position and action.
        Belief shape is an array with size equal the length of subtask_dict.
        """
        start_time = time.time()

        [soup_finish] = observed_info

        distance_trans_belief = np.zeros((len(belief_vector), len(belief_vector)), dtype=float)
        human_pos_and_or = world_state.players[1].pos_and_or
        agent_pos_and_or = world_state.players[0].pos_and_or

        subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

        # get next position for human
        human_obj = human_player.held_object.name if human_player.held_object is not None else 'None'
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.zeros((len(belief_vector)), dtype=float)
        for i, belief in enumerate(belief_vector):
            ## estimating next subtask based on game logic
            game_logic_prob[i] = self._is_valid_object_subtask_pair(human_obj, subtask_key[i], soup_finish, greedy=greedy)*1.0
    
            ## tune subtask estimation based on current human's position and action (use minimum distance between features)
            possible_motion_goals, _ = self.map_action_to_location(world_state, None, self.subtask_dict[subtask_key[i]][0], self.subtask_dict[subtask_key[i]][1], p0_obj=human_obj, player_idx=1)
            # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
            human_dist_cost, feature_pos = self.mp.min_cost_to_feature(human_pos_and_or, possible_motion_goals, with_argmin=True) # select the feature position that is closest to current player's position in world state
            if str(feature_pos) not in prev_dist_to_feature:
                prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            dist_belief_prob[i] = (self.mdp.height+self.mdp.width) + (prev_dist_to_feature[str(feature_pos)] - human_dist_cost)
            # dist_belief_prob[i] = (self.mdp.height+self.mdp.width) - human_dist_cost if human_dist_cost < np.inf else (self.mdp.height + self.mdp.width)

            # update distance to feature
            prev_dist_to_feature[str(feature_pos)] = human_dist_cost

        # print('dist_belief_prob =', dist_belief_prob)
        # print('prev_dist_to_feature =', prev_dist_to_feature)
        # print('human_dist_cost =', human_dist_cost)

        game_logic_prob /= game_logic_prob.sum()
        dist_belief_prob /= dist_belief_prob.sum()

        game_logic_prob[game_logic_prob == 0.0] = 0.000001
        dist_belief_prob[dist_belief_prob== 0.0] = 0.000001

        new_belief = belief*game_logic_prob
        new_belief = new_belief*0.7 * dist_belief_prob*0.3

        new_belief /= new_belief.sum()
        # print("It took {} seconds for belief update".format(time.time() - start_time))

        return new_belief, prev_dist_to_feature

    def compute_V(self, next_world_state, mdp_state_key, search_depth=100):
        next_world_state_str = str(next_world_state)
        if next_world_state_str not in self.world_state_cost_dict:

            delivery_horizon=2
            debug=False
            h_fn=Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = start_world_state.order_list[:delivery_horizon]
            
            expand_fn = lambda state, ori_state_key: self.get_successor_states(state, ori_state_key)
            goal_fn = lambda ori_state_key: len(self.state_dict[ori_state_key][2:]) <= 2
            heuristic_fn = lambda state: h_fn(state)

            search_problem = SearchTree(start_world_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
            path_end_state, cost, over_limit = search_problem.bounded_A_star_graph_search(qmdp_root=mdp_state_key, info=False, cost_limit=search_depth)

            if over_limit:
                cost = self.optimal_plan_cost(path_end_state, cost)

            self.world_state_cost_dict[next_world_state_str] = cost

        # print('self.world_state_cost_dict length =', len(self.world_state_cost_dict))            
        return (self.mdp.height*self.mdp.width)*2 - self.world_state_cost_dict[next_world_state_str]

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(start_world_state, start_world_state.players[0], start_world_state.players[1], subtask)

    def compute_Q(self, b, v, c):
        # print('b =', b)
        # print('v =', v)
        # print('c =', c)

        # tmp=input()
        return b@(v+c)

    def get_best_action(self, q):
        return np.argmax(q)

    def init_mdp(self):
        self.init_actions()
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_transition()

    def compute_mdp(self, filename):
        start_time = time.time()

        final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # print('Total states =', self.num_states, '; Total actions =', self.num_actions)

        # print("It took {} seconds to create HumanSubtaskQMDPPlanner".format(time.time() - start_time))
        self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return 


# class  SteakHumanSubtaskQMDPPlanner(SteakMediumLevelMDPPlanner):
    def __init__(self, mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8):

        super().__init__(mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8)

        self.world_state_cost_dict = {}
        self.jmp = JointMotionPlanner(mdp, mlp_params)
        self.mp = self.jmp.motion_planner
        self.subtask_dict = {}
        self.subtask_idx_dict = {}

    @staticmethod
    def from_qmdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            return pickle.load(f)
            # mdp = mdp_planner.mdp
            # params = mdp_planner.params

            # state_idx_dict = mdp_planner.state_idx_dict
            # state_dict = mdp_planner.state_dict

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            # policy_matrix = mdp_planner.policy_matrix
            # value_matrix = mdp_planner.value_matrix
            
            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds
            
            # return SteakHumanSubtaskQMDPPlanner(mdp, params, mlp_action_manager, state_dict, state_idx_dict, policy_matrix=policy_matrix)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'steak_human_subtask_aware_qmdp' + '.pkl'

        if force_compute_all:
            mdp_planner = SteakHumanSubtaskQMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp(filename)
            return mdp_planner
        
        try:
            mdp_planner = SteakHumanSubtaskQMDPPlanner.from_qmdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = SteakHumanSubtaskQMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        if info:
            print("Loaded SteakHumanSubtaskQMDPPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pan, chopping time, washing time, order list, agent 1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list) 

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'hot_plate', 'steak', 'meat', 'plate', 'dish', 'None']
        self.subtask_dict = copy.deepcopy(self.action_dict)
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for j, subtask in enumerate(self.subtask_dict.items()):
                self.subtask_idx_dict[subtask[0]] = j
                if self._init_is_valid_object_subtask_pair(obj, subtask[0]):
                    for ori_key, ori_value in original_state_dict.items():
                        new_key = ori_key+'_'+obj + '_' + subtask[0]
                        new_obj = original_state_dict[ori_key]+[obj] + [subtask[0]]
                        self.state_dict[new_key] = new_obj # update value
                        self.state_idx_dict[new_key] = len(self.state_idx_dict)

        # print('subtask dict =', self.subtask_dict)

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                normalize_count = 0
                
                # decode state information
                p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                # calculate next states for p1 (a.k.a. human)
                p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

                # calculate next states for p0 (conditioned on p1 (a.k.a. human))
                for p1_nxt_state in p1_nxt_states:
                    action, next_state_key = self.state_transition(p0_state, p1_nxt_world_info, human_state=p1_nxt_state)
                    # for action, next_state_key in zip(actions, next_state_keys):
                        # print(p0_state, p1_nxt_world_info, p1_nxt_state, action, next_state_keys)
                    if action_key == action:
                        next_state_idx= self.state_idx_dict[next_state_key]
                        self.transition_matrix[action_idx, state_idx, next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def get_successor_states(self, start_world_state, start_state_key, debug=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][2:]) <= 2: # [p0_obj, num_item_in_soup, orders, p1_obj, subtask] 
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        agent_action_idx_arr, next_state_idx_arr = np.where(self.transition_matrix[:, ori_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr, next_state_idx_arr):
            next_world_state, cost = self.mdp_action_state_to_world_state(next_action_idx, next_state_idx, start_world_state)
            successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))
            if debug: print('Action {} from {} to {} costs {} in {} seconds.'.format(self.get_key_from_value(self.action_idx_dict, next_action_idx), self.get_key_from_value(self.state_idx_dict, ori_state_idx), self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time()-start_time))

        return successor_states

    def decode_state_info(self, state_obj):
        return state_obj[0], state_obj[-2:], state_obj[1:-2]

    def _init_is_valid_object_subtask_pair(self, obj, subtask):
        if obj == 'None':
            if subtask == 'pickup_steak':
                return True
            elif subtask == 'pickup_onion':
                return True
            elif subtask == 'pickup_meat':
                return True
            elif subtask == 'pickup_plate':
                return True
            elif subtask == 'pickup_garnish':
                return True
            elif subtask == 'pickup_hot_plate':
                return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'meat' and subtask == 'drop_meat':
                return True
            elif obj == 'plate' and subtask == 'drop_plate':
                return True
            elif obj == 'garnish' and subtask == 'pickup_steak':
                return True
            elif obj == 'garnish' and subtask == 'drop_garnish':
                return True
            elif obj == 'hot_plate' and subtask == 'pickup_garnish':
                return True
            elif obj == 'hot_plate' and subtask == 'drop_hot_plate':
                return True
            elif obj == 'dish' and subtask == 'deliver_dish':
                return True
            else:
                return False
        return True

    def _is_valid_object_subtask_pair(self, obj, subtask, num_item_in_pot, chop_time, wash_time, greedy=False):
        if obj == 'None':
            if subtask == 'pickup_onion' and chop_time < 0:
                return True
            elif subtask == 'pickup_meat' and num_item_in_pot < self.mdp.num_items_for_steak:
                return True
            elif subtask == 'pickup_plate' and wash_time < 0:
                return True
            elif subtask == 'pickup_hot_plate' and wash_time >= self.mdp.wash_time:
                return True
            elif subtask == 'heat_hot_plate' and wash_time >= 0 and wash_time < self.mdp.wash_time:
                return True
            elif subtask == 'chop_onion' and chop_time >= 0 and chop_time < self.mdp.chopping_time:
                return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion' and chop_time < 0:
                return True
            elif obj == 'meat' and subtask == 'drop_meat' and num_item_in_pot < self.mdp.num_items_for_steak:
                return True
            elif obj == 'plate' and subtask == 'drop_plate' and wash_time < 0:
                return True
            elif obj == 'garnish' and subtask == 'pickup_steak' and num_item_in_pot >= self.mdp.num_items_for_steak:
                return True
            elif obj == 'garnish' and subtask == 'drop_garnish':
                return True
            elif obj == 'hot_plate' and subtask == 'pickup_garnish' and chop_time >= self.mdp.chopping_time:
                return True
            elif obj == 'hot_plate' and subtask == 'drop_hot_plate' and chop_time < self.mdp.chopping_time:
                return True
            elif obj == 'dish' and subtask == 'deliver_dish':
                return True
            else:
                return False
        return True

    def human_state_subtask_transition(self, human_state, world_info):
        player_obj = human_state[0]; subtask = human_state[1]
        num_item_in_pot = world_info[0]; chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[4:]
        next_obj = player_obj; next_subtasks = []; 
        next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time;

        if player_obj == 'None':
            if subtask == 'pickup_meat':
                next_obj = 'meat'
                next_subtasks = ['drop_meat']

            elif subtask == 'pickup_onion':
                next_obj = 'onion'
                next_subtasks = ['drop_onion']

            elif subtask == 'chop_onion' and chop_time < self.mdp.chopping_time-1:
                next_obj = 'None'
                next_chop_time += 1
                next_subtasks = ['chop_onion']

            elif subtask == 'chop_onion' and chop_time >= self.mdp.chopping_time-1:
                next_obj = 'None'
                next_chop_time += 1
                next_subtasks = ['pickup_plate']

            elif subtask == 'pickup_plate':
                next_obj = 'plate'
                next_subtasks = ['drop_plate']
            
            elif subtask == 'heat_hot_plate' and wash_time < self.mdp.wash_time-1:
                next_obj = 'None'
                next_wash_time += 1
                next_subtasks = ['heat_hot_plate']
            
            elif subtask == 'heat_hot_plate' and wash_time >= self.mdp.wash_time-1:
                next_obj = 'None'
                next_wash_time += 1
                next_subtasks = ['pickup_hot_plate']

            elif subtask == 'pickup_hot_plate':
                next_obj = 'hot_plate'
                next_wash_time = 0
                next_subtasks = ['pickup_steak']

        else:
            next_subtasks = []
            if player_obj == 'meat' and subtask == 'drop_meat' and num_item_in_pot < self.mdp.num_items_for_steak and chop_time < 0:
                next_obj = 'None'
                next_num_item_in_pot += 1

                if chop_time < 0:
                    next_subtasks.append('pickup_onion')
                elif chop_time >= 0 and chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')

                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
            
            elif player_obj == 'onion' and subtask == 'drop_onion' and chop_time < 0:
                next_obj = 'None'
                next_subtasks.append('chop_onion')
                chop_time += 1
                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')

            elif player_obj == 'plate' and subtask == 'drop_plate' and wash_time < 0:
                next_obj = 'None'
                wash_time += 1
                next_subtasks.append('heat_hot_plate')

                if chop_time < 0:
                    next_subtasks.append('pickup_onion')
                elif chop_time >= 0 and chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')

            elif player_obj == 'steak' and subtask == 'pickup_garnish' and chop_time >= self.mdp.chopping_time:
                chop_time = -1
                next_obj = 'dish'
                next_subtasks.append('deliver_dish')
                
            elif player_obj == 'hot_plate' and subtask == 'pickup_steak' and num_item_in_pot >= self.mdp.num_items_for_steak:
                num_item_in_pot = 0
                next_obj = 'steak'
                next_subtasks.append('pickup_garnish')
    
            elif player_obj == 'dish' and subtask == 'deliver_dish':
                next_obj = 'None'
                if num_item_in_pot < self.mdp.num_items_for_steak:
                    next_subtasks.append('pickup_meat')
                elif chop_time < 0:
                    next_subtasks.append('pickup_onion')
                elif chop_time > 0 and chop_time < self.mdp.chopping_time:
                    next_subtasks.appned('chop_onion')
                
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                next_obj = 'None'
                next_subtasks.append('drop_'+player_obj)
                print(player_obj, subtask)
                # raise ValueError()

        if next_num_item_in_pot > self.mdp.num_items_for_steak:
            next_num_item_in_pot = self.mdp.num_items_for_steak

        p1_nxt_states = []
        for next_subtask in next_subtasks:
            p1_nxt_states.append([next_obj, next_subtask])

        nxt_world_info = [next_num_item_in_pot, chop_time, wash_time]
        for order in orders:
            nxt_world_info.append(order)

        return p1_nxt_states, nxt_world_info
    
    def state_transition(self, player_obj, world_info, human_state=[None, None]):
        # game logic
        num_item_in_pot = world_info[0]; chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[4:]
        other_obj = human_state[0]; subtask = human_state[1]
        actions = ''; next_obj = player_obj; next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time

        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (other_obj != 'meat' and subtask != 'pickup_meat'):
                actions = 'pickup_meat'
                next_obj = 'meat'
            elif (chop_time < 0) and (other_obj != 'onion' and subtask != 'pickup_onion'):
                actions = 'pickup_onion'
                next_obj = 'onion'
            elif (chop_time > 0) and (chop_time < self.mdp.chopping_time) and (wash_time < self.mdp.wash_time):
                actions = 'chop_onion'
                next_obj = 'None'
            elif (wash_time < 0) and (other_obj != 'plate' and subtask != 'pickup_plate'):
                actions = 'pickup_plate'
                next_obj = 'plate'
            elif (wash_time > 0) and (wash_time < self.mdp.wash_time):
                actions = 'heat_hot_plate'
                next_obj = 'None'
            elif (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
                actions = 'pickup_hot_plate'
                next_obj = 'hot_plate'
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == 'steak':
                    actions = 'pickup_meat'
                    next_obj = 'meat'

                else:
                    actions = 'pickup_meat'
                    next_obj = 'meat'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                next_chop_time = 0

            elif player_obj == 'meat':
                actions = 'drop_meat'
                next_obj = 'None'
                next_num_item_in_pot = 1

            elif player_obj == 'plate':
                actions = 'drop_plate'
                next_obj = 'None'
                next_wash_time = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions = 'pickup_steak'
                next_obj = 'steak'
                next_num_item_in_pot = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions = 'drop_hot_plate'
                next_obj = 'None'
                next_wash_time = 'None'

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = 'None'

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                actions = 'drop_steak'
                next_obj = 'None'

            elif player_obj == 'dish':
                actions = 'deliver_dish'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj)
                raise ValueError()

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order
            
        for human_info in human_state:
            next_state_keys = next_state_keys + '_' + human_info

        return actions, next_state_keys
    
    def world_state_to_mdp_state_key(self, state, player, other_player, subtask):
        state_str = super().gen_state_dict_key(state, player, other_player=other_player)
        
        if other_player.held_object is not None:
            other_player_obj = other_player.held_object.name

        state_str = state_str + '_' + str(other_player_obj) + '_' + subtask

        return state_str
    
    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]
    
    def gen_state_dict_key(self, p0_obj, p1_obj, num_item_in_pot, chop_time, wash_time, orders, subtasks):

        player_obj = p0_obj if p0_obj is not None else 'None'
        other_player_obj = p1_obj if p1_obj is not None else 'None'

        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + '_' + str(order)

        state_strs = []
        for subtask in subtasks:
            state_strs.append(str(player_obj)+'_'+str(num_item_in_pot)+'_' + str(chop_time) + '_' + str(wash_time) + '_'+ order_str + '_' + str(other_player_obj) + '_' + subtask)

        return state_strs
    
    def map_action_to_location(self, world_state, state_str, action, obj, p0_obj=None, player_idx=None, counter_drop=True, state_dict=None):

        p0_obj = p0_obj if p0_obj is not None else self.state_dict[state_str][0]
        other_obj = world_state.players[1-player_idx].held_object.name if world_state.players[1-player_idx].held_object is not None else 'None'
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []

        WAIT = False # If wait becomes true, one player has to wait for the other player to finish its current task and its next task

        if action == 'pickup':
            if p0_obj != 'None' and counter_drop:
                location = self.drop_item(world_state)
            else:
                if obj == 'onion':
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == 'plate':
                    location = self.mdp.get_dish_dispens()
                elif obj == 'meat':
                    location = self.mdp.get_meat_dispenser_locations()
                elif obj == 'hot_plate':
                    location = self.mdp.get_sink_status(world_state)['full'] + self.mdp.get_sink_status(world_state)['ready']

                    if len(location) == 0:
                        WAIT = True
                        location = self.mdp.get_sink_status['empty']
                        return location, WAIT
                    
                elif obj == 'garish':
                    location = self.mdp.get_chopping_board_status(world_state)['full'] + self.mdp.get_chopping_board_status(world_state)['ready']

                    if len(location) == 0:
                        WAIT = True
                        location = self.mdp.get_chopping_board_status['empty']
                        return location, WAIT
                    
                elif obj == 'steak':
                    location = self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

                    if len(location) == 0:
                        WAIT = True
                        location = self.mdp.get_chopping_board_status['empty']
                        return location, WAIT
                
                else:
                    print(p0_obj, action, obj)
                    ValueError()

        elif action == 'drop':
            if obj == 'meat':
                location = self.mdp.get_empty_pots(pots_states_dict)
                if len(location) == 0 and other_obj != 'meat':
                    WAIT = True
                    location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                    return location, WAIT
            elif obj == 'onion':
                location = self.mdp.get_chopping_board_status(world_state)['empty']
                if len(location) == 0 and other_obj != 'onion':
                    WAIT = True
                    location = self.mdp.get_chopping_board_status(world_state)['ready']
                    return location, WAIT
            elif obj == 'plate':
                location = self.mdp.get_sink_status(world_state)['empty']
                if len(location) == 0 and other_obj != 'plate':
                    WAIT = True
                    location = self.mdp.get_sink_status(world_state)['ready']
                    return location, WAIT
            elif obj == 'hot_plate' or obj == 'steak':
                location = self.drop_item(world_state)
            
            if len(location) == 0 and counter_drop:
                location = self.drop_item(world_state)

        elif action == 'deliver':
            if p0_obj != 'dish':
                location = self.drop_item(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location, WAIT
    
    def _shift_same_goal_pos(self, new_positions, change_idx):
        
        pos = new_positions[change_idx][0]
        ori = new_positions[change_idx][1]
        new_pos = pos; new_ori = ori
        if self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])
        else:
            print('pos = ', pos)
            ValueError()
        
        new_positions[change_idx] = (new_pos, new_ori)

        return new_positions[0], new_positions[1]

    def mdp_action_state_to_world_state(self, action_idx, ori_state_idx, ori_world_state, with_argmin=False):
        new_world_state = ori_world_state.deepcopy()
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)

        possible_agent_motion_goals, AI_WAIT = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[action][0], self.action_dict[action][1], player_idx=0) 
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[mdp_state_obj[-1]][0], self.action_dict[mdp_state_obj[-1]][1], p0_obj=mdp_state_obj[-2], player_idx=1) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
        # get next position for AI agent
        agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[0].pos_and_or, possible_agent_motion_goals, with_motion_goal=True) # select the feature position that is closest to current player's position in world state
        new_agent_pos = agent_feature_pos if agent_feature_pos is not None else new_world_state.players[0].get_pos_and_or()
        human_cost, human_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[1].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else new_world_state.players[1].get_pos_and_or()
        # print(new_agent_pos, new_human_pos)

        if new_agent_pos == new_human_pos:
            new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos], np.argmax(np.array([agent_cost, human_cost])))
            # print('after shift =', new_agent_pos, new_human_pos)

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].remove_object()
        if mdp_state_obj[0] != 'None' and mdp_state_obj[0] != 'soup':
            new_world_state.players[0].held_object = ObjectState(mdp_state_obj[0], new_agent_pos)
        new_world_state.players[0].update_pos_and_or(new_agent_pos[0], new_agent_pos[1])

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].remove_object()
        if mdp_state_obj[-2] != 'None' and mdp_state_obj[-2] != 'soup':
            new_world_state.players[1].held_object = ObjectState(mdp_state_obj[-2], new_human_pos)
        new_world_state.players[1].update_pos_and_or(new_human_pos[0], new_human_pos[1])

        total_cost = max([agent_cost, human_cost]) # in rss paper is max
        if AI_WAIT or HUMAN_WAIT: # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area length)
            total_cost = agent_cost + human_cost + ((self.mdp.width-1)+(self.mdp.height-1))/2

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost

    def world_to_state_keys(self, world_state, player, other_player, belief):
        mdp_state_keys = []
        for i, b in enumerate(belief):
            mdp_state_key = self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i))
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i)))
        return mdp_state_keys

    def joint_action_cost(self, world_state, goal_pos_and_or, COST_OF_STAY=1):
        joint_action_plan, end_motion_state, plan_costs = self.jmp.get_low_level_action_plan(world_state.players_pos_and_or, goal_pos_and_or, merge_one=True)
        # joint_action_plan, end_state, plan_costs = self.mlp.get_embedded_low_level_action_plan(world_state, goal_pos_and_or, other_agent, other_agent_idx)
        # print('joint_action_plan =', joint_action_plan, '; plan_costs =', plan_costs)

        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0

        num_of_non_stay_actions = len([a for a in joint_action_plan if a[0] != Action.STAY])
        num_of_stay_actions = len([a for a in joint_action_plan if a[0] == Action.STAY])

        return joint_action_plan[0], max(plan_costs)# num_of_non_stay_actions+num_of_stay_actions*COST_OF_STAY # in rss paper is max(plan_costs)

    def step(self, world_state, mdp_state_keys, belief, agent_idx, low_level_action=False):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dic will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        next_state_v = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        action_cost = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        qmdp_q = np.zeros((len(self.action_dict), len(belief)), dtype=float)
        # ml_action_to_low_action = np.zeros()

        # for each subtask, obtain next mdp state but with low level location based on finishing excuting current action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None:
                agent_action_idx_arr, next_mdp_state_idx_arr = np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
                nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr): # action_idx is encoded subtask action
                    # print('action_idx =', action_idx)
                    next_state_idx = next_mdp_state_idx_arr[j]
                    after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx, mdp_state_idx, world_state, with_argmin=True)
                    value_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_state_idx), search_depth=100)
                    joint_action, one_step_cost = self.joint_action_cost(world_state, after_action_world_state.players_pos_and_or)  
                    # print('joint_action =', joint_action, 'one_step_cost =', one_step_cost)
                    # print('Action.ACTION_TO_INDEX[joint_action[agent_idx]] =', Action.ACTION_TO_INDEX[joint_action[agent_idx]])
                    if not low_level_action:
                        # action_idx: are subtask action dictionary index
                        next_state_v[i, action_idx] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, action_idx] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    # print('action_idx =', self.get_key_from_value(self.action_idx_dict, action_idx), '; mdp_state_key =', mdp_state_key, '; next_state_key =', self.get_key_from_value(self.state_idx_dict, next_state_idx))
                    # print('next_state_v =', next_state_v[i])
        # print('action_cost =', action_cost)

        q = self.compute_Q(belief, next_state_v, action_cost)
        # print(q)
        action_idx = self.get_best_action(q)
        # print('get_best_action =', action_idx, '=', self.get_key_from_value(self.action_idx_dict, action_idx))
        # print("It took {} seconds for this step".format(time.time() - start_time))
        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
        return action_idx, self.action_dict[self.get_key_from_value(self.action_idx_dict, action_idx)], low_level_action
    
    def observe(self, world_state):
        num_item_in_pot, chop_state, sink_state = 0, -1, -1

        for obj in world_state.objects.values():
            if obj.name == 'hot_plate' and obj.position in self.mdp.get_sink_locations():
                wash_time = obj.state
                if wash_time > sink_state:
                    sink_state = wash_time
            elif obj.name == 'steak' and obj.position in self.mdp.get_pot_locations():
                _, _, cook_time = obj.state
                if cook_time > 0:
                    num_item_in_pot = 1
            elif obj.name == 'garnish' and obj.position in self.mdp.get_chopping_board_locations():
                chop_time = obj.state
                if chop_time > chop_state:
                    chop_state = chop_time

        if chop_state < 0:
            chop_state = None
        if sink_state < 0:
            sink_state = None


        return [num_item_in_pot, chop_state, sink_state]

    def belief_update(self, world_state, agent_player, observed_info, human_player, belief_vector, prev_dist_to_feature, greedy=False):
        """
        Update belief based on both human player's game logic and also it's current position and action.
        Belief shape is an array with size equal the length of subtask_dict.
        """
        [num_item_in_pot, chop_time, wash_time] = observed_info
        start_time = time.time()

        distance_trans_belief = np.zeros((len(belief_vector), len(belief_vector)), dtype=float)
        human_pos_and_or = world_state.players[1].pos_and_or
        agent_pos_and_or = world_state.players[0].pos_and_or

        subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

        # get next position for human
        human_obj = human_player.held_object.name if human_player.held_object is not None else 'None'
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.zeros((len(belief_vector)), dtype=float)
        for i, belief in enumerate(belief_vector):
            ## estimating next subtask based on game logic
            game_logic_prob[i] = self._is_valid_object_subtask_pair(human_obj, subtask_key[i], num_item_in_pot, chop_time, wash_time, greedy=greedy)*1.0
    
            ## tune subtask estimation based on current human's position and action (use minimum distance between features)
            possible_motion_goals, _ = self.map_action_to_location(world_state, None, self.subtask_dict[subtask_key[i]][0], self.subtask_dict[subtask_key[i]][1], p0_obj=human_obj, player_idx=1)
            # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
            human_dist_cost, feature_pos = self.mp.min_cost_to_feature(human_pos_and_or, possible_motion_goals, with_argmin=True) # select the feature position that is closest to current player's position in world state
            if str(feature_pos) not in prev_dist_to_feature:
                prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            dist_belief_prob[i] = (self.mdp.height+self.mdp.width) + (prev_dist_to_feature[str(feature_pos)] - human_dist_cost)
            # dist_belief_prob[i] = (self.mdp.height+self.mdp.width) - human_dist_cost if human_dist_cost < np.inf else (self.mdp.height + self.mdp.width)

            # update distance to feature
            prev_dist_to_feature[str(feature_pos)] = human_dist_cost

        # print('dist_belief_prob =', dist_belief_prob)
        # print('prev_dist_to_feature =', prev_dist_to_feature)
        # print('human_dist_cost =', human_dist_cost)

        game_logic_prob /= game_logic_prob.sum()
        dist_belief_prob /= dist_belief_prob.sum()

        game_logic_prob[game_logic_prob == 0.0] = 0.000001
        dist_belief_prob[dist_belief_prob== 0.0] = 0.000001

        new_belief = belief*game_logic_prob
        new_belief = new_belief*0.7 * dist_belief_prob*0.3

        new_belief /= new_belief.sum()
        # print("It took {} seconds for belief update".format(time.time() - start_time))

        return new_belief, prev_dist_to_feature

    def compute_V(self, next_world_state, mdp_state_key, search_depth=100):
        next_world_state_str = str(next_world_state)
        if next_world_state_str not in self.world_state_cost_dict:

            delivery_horizon=2
            debug=False
            h_fn=Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = start_world_state.order_list[:delivery_horizon]
            
            expand_fn = lambda state, ori_state_key: self.get_successor_states(state, ori_state_key)
            goal_fn = lambda ori_state_key: len(self.state_dict[ori_state_key][2:]) <= 2
            heuristic_fn = lambda state: h_fn(state)

            search_problem = SearchTree(start_world_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
            path_end_state, cost, over_limit = search_problem.bounded_A_star_graph_search(qmdp_root=mdp_state_key, info=False, cost_limit=search_depth)

            if over_limit:
                cost = self.optimal_plan_cost(path_end_state, cost)

            self.world_state_cost_dict[next_world_state_str] = cost

        # print('self.world_state_cost_dict length =', len(self.world_state_cost_dict))            
        return (self.mdp.height*self.mdp.width)*2 - self.world_state_cost_dict[next_world_state_str]

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(start_world_state, start_world_state.players[0], start_world_state.players[1], subtask)

    def compute_Q(self, b, v, c):
        # print('b =', b)
        # print('v =', v)
        # print('c =', c)

        # tmp=input()
        return b@(v+c)

    def get_best_action(self, q):
        return np.argmax(q)

    def init_mdp(self):
        self.init_actions()
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_transition()

    def compute_mdp(self, filename):
        start_time = time.time()

        final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # print('Total states =', self.num_states, '; Total actions =', self.num_actions)

        # print("It took {} seconds to create HumanSubtaskQMDPPlanner".format(time.time() - start_time))
        self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return 

class  SteakHumanSubtaskQMDPPlanner(SteakMediumLevelMDPPlanner):
    def __init__(self, mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8, jmp = None, vision_limited_human = None):

        super().__init__(mdp, mlp_params, \
        state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8, jmp=jmp)

        self.world_state_cost_dict = {}
        self.jmp = JointMotionPlanner(mdp, mlp_params) if jmp is None else jmp
        self.mp = self.jmp.motion_planner
        self.subtask_dict = {}
        self.subtask_idx_dict = {}
        self.sim_human_model = vision_limited_human
        if self.sim_human_model is not None:
            self.sim_human_model.set_agent_index(vision_limited_human.agent_index)
        # if vision_limited_human is not None: 
        #     self.human_knowledge = vision_limited_human.knowledge_base

    @staticmethod
    def from_qmdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            return pickle.load(f)
            # mdp = mdp_planner.mdp
            # params = mdp_planner.params

            # state_idx_dict = mdp_planner.state_idx_dict
            # state_dict = mdp_planner.state_dict

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            # policy_matrix = mdp_planner.policy_matrix
            # value_matrix = mdp_planner.value_matrix
            
            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds
            
            # return SteakHumanSubtaskQMDPPlanner(mdp, params, mlp_action_manager, state_dict, state_idx_dict, policy_matrix=policy_matrix)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False, jmp=None, vision_limited_human=None):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'steak_human_subtask_aware_qmdp' + '.pkl'

        if force_compute_all:
            mdp_planner = SteakHumanSubtaskQMDPPlanner(mdp, mlp_params, vision_limited_human=vision_limited_human)
            mdp_planner.compute_mdp(filename)
            return mdp_planner
        
        try:
            mdp_planner = SteakHumanSubtaskQMDPPlanner.from_qmdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = SteakHumanSubtaskQMDPPlanner(mdp, mlp_params, jmp=jmp, vision_limited_human=vision_limited_human)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        if info:
            print("Loaded SteakHumanSubtaskQMDPPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pan, chopping time, washing time, order list, agent 1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list) 

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        # objects = ['onion', 'hot_plate', 'steak', 'meat', 'plate', 'dish', 'None']
        self.subtask_dict = copy.deepcopy(self.action_dict)
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        # for i, obj in enumerate(objects):
        for j, subtask in enumerate(self.subtask_dict.items()):
            self.subtask_idx_dict[subtask[0]] = j
            for ori_key, ori_value in original_state_dict.items():
                ori_state_info = [ori_key.split('_')[0]]
                for i, k in enumerate(ori_key.split('_')[1:]):
                    if k == 'plate' and i == 0 and ori_state_info[0] == 'hot':
                        ori_state_info[0] = 'hot_plate'
                    else:
                        if k == 'None':
                            ori_state_info.append(-1)
                        else:
                            ori_state_info.append(k)
                # if self._is_valid_object_subtask_pair(subtask[0], int(ori_state_info[1]), int(ori_state_info[2]), int(ori_state_info[3])):
                new_key = ori_key + '_' + subtask[0]
                new_obj = original_state_dict[ori_key] + [subtask[0]]
                self.state_dict[new_key] = new_obj # update value
                self.state_idx_dict[new_key] = len(self.state_idx_dict)

        # print('subtask dict =', self.subtask_dict)

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                normalize_count = 0
                
                # decode state information
                p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                # calculate next states for p1 (a.k.a. human)
                p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

                # append original state of p1 (human) to represent unfinished subtask state transition
                p1_nxt_states.append(p1_state)
                p1_nxt_world_info += [world_info]

                # calculate next states for p0 (conditioned on p1 (a.k.a. human))
                for i, p1_nxt_state in enumerate(p1_nxt_states):
                    actions, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)
                    for action, next_state_key in zip(actions, next_state_keys):
                        # print(p0_state, p1_nxt_world_info[i], p1_nxt_state, action, next_state_keys)
                        if action_key == action:
                            next_state_idx= self.state_idx_dict[next_state_key]
                            self.transition_matrix[action_idx, state_idx, next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def get_successor_states(self, start_world_state, start_state_key, debug=False, add_rewards=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][4:-1]) == 0: # [p0_obj, num_item_in_soup, orders, p1_obj, subtask] 
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        agent_action_idx_arr, next_state_idx_arr = np.where(self.transition_matrix[:, ori_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr, next_state_idx_arr):
            next_world_state, cost = self.mdp_action_state_to_world_state(next_action_idx, next_state_idx, start_world_state)
            successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))
            if debug: print('Action {} from {} to {} costs {} in {} seconds.'.format(self.get_key_from_value(self.action_idx_dict, next_action_idx), self.get_key_from_value(self.state_idx_dict, ori_state_idx), self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time()-start_time))

        return successor_states

    def get_key_from_value(self, dictionary, state_value):
        try: 
            idx = list(dictionary.values()).index(state_value)
        except ValueError:
            return None
        else:
            return list(dictionary.keys())[idx]
        
    def decode_state_info(self, state_obj): # state_obj = id 0; other_subtask = last element; world info = everything in between
        return state_obj[0], state_obj[-1], state_obj[1:-1]

    # def _init_is_valid_object_subtask_pair(self, obj, subtask):
        if obj == 'None':
            if subtask == 'pickup_steak':
                return True
            elif subtask == 'pickup_onion':
                return True
            elif subtask == 'pickup_meat':
                return True
            elif subtask == 'pickup_plate':
                return True
            elif subtask == 'pickup_garnish':
                return True
            elif subtask == 'pickup_hot_plate':
                return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'meat' and subtask == 'drop_meat':
                return True
            elif obj == 'plate' and subtask == 'drop_plate':
                return True
            elif obj == 'garnish' and subtask == 'pickup_steak':
                return True
            elif obj == 'garnish' and subtask == 'drop_garnish':
                return True
            elif obj == 'hot_plate' and subtask == 'pickup_garnish':
                return True
            elif obj == 'hot_plate' and subtask == 'drop_hot_plate':
                return True
            elif obj == 'dish' and subtask == 'deliver_dish':
                return True
            else:
                return False
        return True

    def _is_valid_object_subtask_pair(self, subtask, num_item_in_pot, chop_time, wash_time, vision_limit=False, human_obj=None, other_agent_holding=None):
        '''
        Since we do not consider the other agent's action for the human's subtask initialization, we mainly just care about whether the object the human is holding pairs with the subtask. Also, since the other agent may change the world, we forgo that the world as constraints on the subtasks.
        When considering belief update, we need to consider how the robot's holding object effects the human when the human is a robot-aware human model.
        '''
        if chop_time == 'None' or chop_time == None:
            chop_time = -1
        if wash_time == 'None' or wash_time == None:
            wash_time = -1

        # map subtask to possible object holding
        if human_obj is None:
            subtask_action = subtask.split('_')[0]
            subtask_obj = '_'.join(subtask.split('_')[1:])
            if (subtask_action in ['pickup', 'chop', 'heat']) and subtask_obj not in ['steak', 'garnish']:
                obj = 'None'
            elif subtask == 'pickup_steak':
                obj = 'hot_plate'
            elif subtask == 'pickup_garnish':
                obj = 'steak'
            else:
                obj = subtask_obj
        else:
            obj = human_obj

        if obj == 'None':
            if subtask == 'pickup_onion' and chop_time < 0 and other_agent_holding != 'onion':
                return True
            elif subtask == 'pickup_meat' and num_item_in_pot < self.mdp.num_items_for_steak and other_agent_holding != 'meat':
                return True
            elif subtask == 'pickup_plate' and wash_time < 0 and other_agent_holding != 'plate':
                return True
            elif subtask == 'pickup_hot_plate' and wash_time >= self.mdp.wash_time:# and other_agent_holding != 'hot_plate':# and other_agent_holding != 'steak':
                return True
            elif subtask == 'heat_hot_plate' and wash_time >= 0 and wash_time < self.mdp.wash_time:
                return True
            elif subtask == 'chop_onion' and chop_time >= 0 and chop_time < self.mdp.chopping_time:
                return True
            else: # this is an instance that will be triggered when there are no other things to pick up.
                # if (subtask == 'pickup_meat' and other_agent_holding != 'meat') or (subtask == 'pickup_onion' and other_agent_holding != 'onion') or (subtask == 'pickup_plate' and other_agent_holding != 'plate'):
                #     return True
                # else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':# and chop_time < 0:
                return True
            elif obj == 'meat' and subtask == 'drop_meat':# and num_item_in_pot < self.mdp.num_items_for_steak:
                return True
            elif obj == 'plate' and subtask == 'drop_plate':# and wash_time < 0:
                return True
            elif obj == 'hot_plate' and subtask == 'pickup_steak' and num_item_in_pot >= self.mdp.num_items_for_steak:
                return True
            elif obj == 'steak' and (subtask == 'pickup_garnish' or subtask == 'drop_steak'):# and chop_time >= self.mdp.chopping_time:
                return True
            elif obj == 'hot_plate' and subtask == 'drop_hot_plate':# and chop_time < self.mdp.chopping_time:
                return True
            elif obj == 'dish' and subtask == 'deliver_dish':
                return True
            else:
                return False

    def human_state_subtask_transition(self, subtask, world_info):
        # player_obj = human_state[0] 
        num_item_in_pot = int(world_info[0]); chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[3:]
        # next_obj = player_obj; 

        if chop_time == 'None':
            chop_time = -1
        else:
            chop_time = int(chop_time)
        if wash_time == 'None':
            wash_time = -1
        else:
            wash_time = int(wash_time)
            
        next_subtasks = []
        nxt_world_info = []
        next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time; next_orders = orders.copy()

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and subtask_obj not in ['steak', 'garnish']:
            player_obj = 'None'
        elif subtask == 'pickup_steak':
            player_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            player_obj = 'steak'
        else:
            player_obj = subtask_obj

        if player_obj == 'None':
            if subtask == 'pickup_meat':# and num_item_in_pot < self.mdp.num_items_for_steak:
                next_obj = 'meat'
                next_subtasks = ['drop_meat']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_onion':# and chop_time < 0:
                next_obj = 'onion'
                next_subtasks = ['drop_onion']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'chop_onion' and chop_time < self.mdp.chopping_time-1 and chop_time >= 0:
                next_obj = 'None'
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time
                next_subtasks = ['chop_onion']
                nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'chop_onion' and chop_time >= self.mdp.chopping_time-1:
                next_obj = 'None'
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time

                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                elif wash_time >= self.mdp.wash_time:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
            
            elif subtask == 'chop_onion' and chop_time < 0:
                next_subtasks = ['pickup_onion']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_plate':# and wash_time < 0:
                next_obj = 'plate'
                next_subtasks = ['drop_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
            
            elif subtask == 'heat_hot_plate' and wash_time < self.mdp.wash_time-1 and wash_time >= 0:
                next_obj = 'None'
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_subtasks = ['heat_hot_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
            
            elif subtask == 'heat_hot_plate' and wash_time >= self.mdp.wash_time-1:
                next_obj = 'None'
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_subtasks = ['pickup_hot_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)

                # if chop_time < 0:
                #     next_subtasks.append('pickup_onion')
                # elif chop_time >= 0 and chop_time < self.mdp.chopping_time:
                #     next_subtasks.append('chop_onion')

            elif subtask == 'heat_hot_plate' and wash_time < 0:
                next_subtasks = ['pickup_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_hot_plate' and num_item_in_pot >= self.mdp.num_items_for_steak:
                next_obj = 'hot_plate'
                next_subtasks = ['pickup_steak']
                nxt_world_info += self.gen_world_info_list(chop_time, -1, num_item_in_pot, orders)
            
            elif subtask == 'pickup_hot_plate' and num_item_in_pot < self.mdp.num_items_for_steak:
                next_obj = 'hot_plate'
                next_subtasks = ['pickup_hot_plate'] #next_subtasks = ['drop_hot_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, -1, num_item_in_pot, orders)

            elif subtask == 'pickup_steak':
                next_num_item_in_pot = 0
                if chop_time >= self.mdp.chopping_time: 
                    next_subtasks = ['pickup_garnish']
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                else:
                    next_subtasks = ['pickup_steak']#next_subtasks = ['drop_steak']
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
            else:
                print(subtask, world_info)
                raise ValueError()
        else:
            if subtask == 'drop_meat':# and num_item_in_pot < self.mdp.num_items_for_steak:
                next_obj = 'None'
                if num_item_in_pot < self.mdp.num_items_for_steak: next_num_item_in_pot = 1

                if chop_time < 0:
                    next_subtasks.append('pickup_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                elif chop_time >= 0 and chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                elif wash_time >= self.mdp.wash_time:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
            
            elif subtask == 'drop_onion':
                next_obj = 'None'
                if chop_time < 0: next_chop_time = 0

                if chop_time < self.mdp.chopping_time: 
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                else:
                    if wash_time < 0:
                        next_subtasks.append('pickup_plate')
                        nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                    elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                        next_subtasks.append('heat_hot_plate')
                        nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                    elif wash_time >= self.mdp.wash_time:
                        next_subtasks.append('pickup_hot_plate')
                        nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'drop_plate':
                next_obj = 'None'
                if wash_time < 0: next_wash_time = 0
                
                if wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
                else:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
            
            elif subtask == 'drop_hot_plate':
                next_obj = 'None'
                if num_item_in_pot < self.mdp.num_items_for_steak:
                    next_subtasks.append('pickup_meat')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                else:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'drop_steak':
                next_obj = 'None'
                if chop_time < 0:
                    next_subtasks.append('pickup_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                elif chop_time >= 0 and chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                else:
                    next_subtasks.append('pickup_steak')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_garnish' and chop_time >= self.mdp.chopping_time:
                next_obj = 'dish'
                next_subtasks.append('deliver_dish')
                nxt_world_info += self.gen_world_info_list(-1, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_garnish' and chop_time < self.mdp.chopping_time:
                next_obj = 'None'
                next_subtasks.append('pickup_garnish') # next_subtasks.append('drop_steak')
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                
            elif subtask == 'pickup_steak' and num_item_in_pot >= self.mdp.num_items_for_steak:
                next_num_item_in_pot = 0
                next_obj = 'steak'
                next_subtasks.append('pickup_garnish')
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

            elif subtask == 'pickup_steak' and num_item_in_pot < self.mdp.num_items_for_steak:
                next_obj = 'None'
                next_subtasks.append('drop_hot_plate')
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

            elif subtask == 'deliver_dish':
                if len(orders) >= 1:
                    next_orders.pop(0)

                next_obj = 'None'
                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif chop_time < 0:
                    next_subtasks.append('pickup_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif num_item_in_pot < self.mdp.num_items_for_steak:
                    next_subtasks.append('pickup_meat')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif chop_time >= 0 and chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif wash_time >=self.mdp.wash_time:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                
            else:
                # next_obj = 'None'
                # next_subtasks.append('drop_'+player_obj)
                print(player_obj, subtask)
                raise ValueError()

        # if next_num_item_in_pot > self.mdp.num_items_for_steak:
        #     next_num_item_in_pot = self.mdp.num_items_for_steak

        # p1_nxt_states = []
        # for next_subtask in next_subtasks:
        #     p1_nxt_states.append([next_subtask])
        #     nxt_world_info += self.gen_world_info_list(next_chop_time, next_wash_time, next_num_item_in_pot, orders)

        return next_subtasks, nxt_world_info
    
    def world_based_human_state_subtask_transition(self, subtask, world_info, other_agent_obj='None'):
        # player_obj = human_state[0] 
        num_item_in_pot = int(world_info[0]); chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[3:]
        # next_obj = player_obj; 

        if chop_time == 'None':
            chop_time = -1
        else:
            chop_time = int(chop_time)
        if wash_time == 'None':
            wash_time = -1
        else:
            wash_time = int(wash_time)
            
        next_subtasks = []
        nxt_world_info = []
        next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time; next_orders = orders.copy()

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup']) and subtask_obj not in ['garnish']:
            player_obj = subtask_obj
        elif subtask == 'pickup_garnish':
            player_obj = 'dish'
        else:
            player_obj = 'None'

        # decide next subtask based on the environment
        if player_obj == 'None':
            if chop_time >= 0 and chop_time < self.mdp.chopping_time:
                next_subtasks += ['chop_onion']
            elif wash_time >= 0 and wash_time < self.mdp.wash_time:
                next_subtasks += ['heat_hot_plate']
            elif num_item_in_pot == 0 and len(orders) > 0 and other_agent_obj != 'meat':
                next_subtasks += ['pickup_meat']
            elif chop_time < 0 and other_agent_obj != 'onion':
                next_subtasks += ['pickup_onion']
            elif wash_time < 0 and other_agent_obj != 'plate':
                next_subtasks += ['pickup_plate']
            elif (chop_time >= self.mdp.chopping_time or other_agent_obj == 'onion') and wash_time >= self.mdp.wash_time and not (other_agent_obj == 'hot_plate' or other_agent_obj == 'steak'):
                next_subtasks += ['pickup_hot_plate']
            elif (chop_time >= self.mdp.chopping_time or other_agent_obj == 'onion') and wash_time < self.mdp.wash_time and other_agent_obj == 'plate' and not (other_agent_obj == 'hot_plate' or other_agent_obj == 'steak'):
                next_subtasks += ['pickup_hot_plate']
            else:
                next_subtasks += ['pickup_plate']
        else:
            if player_obj == 'onion':
                next_subtasks = ['drop_onion']
            elif player_obj == 'meat':
                next_subtasks = ['drop_meat']
            elif player_obj == 'plate':
                next_subtasks = ['drop_plate']
            elif player_obj == 'hot_plate':
                next_subtasks = ['pickup_steak']
            elif player_obj == 'steak':
                next_subtasks = ['pickup_garnish']
            elif player_obj == 'dish':
                next_subtasks = ['deliver_dish']

        if len(next_subtasks) == 0:
            next_subtasks = [subtask]

        # update world info
        if (subtask_action in ['pickup']) and subtask_obj not in ['hot_plate', 'steak', 'garnish']:
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
        elif subtask == 'pickup_hot_plate':
            nxt_world_info += self.gen_world_info_list(chop_time, -1, num_item_in_pot, orders)
        elif subtask == 'pickup_steak':
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, 0, orders)
        elif subtask == 'pickup_garnish':
            nxt_world_info += self.gen_world_info_list(-1, wash_time, num_item_in_pot, orders)
        elif subtask == 'drop_meat':
            if num_item_in_pot == 0: next_num_item_in_pot = 1 # meaning you drop at the right location instead of just on the counter
            nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, next_num_item_in_pot, orders)
        elif subtask == 'drop_onion':
            if chop_time < 0: next_chop_time = 0 # meaning you drop at the right location instead of just on the counter
            nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
        elif subtask == 'drop_plate':
            if wash_time < 0: next_wash_time = 0 # meaning you drop at the right location instead of just on the counter
            nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
        elif subtask == 'chop_onion':
            next_chop_time = min(chop_time + 1, self.mdp.chopping_time)
            nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
        elif subtask == 'heat_hot_plate':
            next_wash_time = min(wash_time + 1, self.mdp.wash_time)
            nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
        elif subtask == 'deliver_dish':
            if len(orders) >= 1:
                next_orders.pop(0)
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
        elif subtask in ['drop_hot_plate', 'drop_steak', 'drop_dish']:
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
        else:
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
            print(subtask, world_info, other_agent_obj)
            raise ValueError()

        return next_subtasks, nxt_world_info
    
    def state_transition(self, player_obj, world_info, human_state=None, human_obj=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]; chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[3:]
        # other_obj = human_state[0]; 
        if human_state is not None: 
            subtask = human_state[0]
            
            subtask_action = subtask.split('_')[0]
            subtask_obj = '_'.join(subtask.split('_')[1:])
            
        if human_obj is None:
            if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
                human_obj = 'None'
            elif subtask == 'pickup_steak':
                human_obj = 'hot_plate'
            elif subtask == 'pickup_garnish':
                human_obj = 'steak'
            else:
                human_obj = subtask_obj

        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = ''; next_obj = player_obj; next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat'):
                actions = 'pickup_meat'
                next_obj = 'meat'
            elif (chop_time < 0) and (human_obj != 'onion'):
                actions = 'pickup_onion'
                next_obj = 'onion'
            elif (wash_time < 0) and (human_obj != 'plate'):
                actions = 'pickup_plate'
                next_obj = 'plate'
            elif ((chop_time >= 0) and (chop_time < self.mdp.chopping_time) and (subtask != 'chop_onion')) or ((chop_time < 0) and (human_obj == 'onion')):
                actions = 'chop_onion'
                next_obj = 'None'
                next_chop_time += 1
            elif ((wash_time >= 0) and (wash_time < self.mdp.wash_time) and (subtask != 'heat_hot_plate')) or ((wash_time < 0) and (human_obj == 'plate')):
                actions = 'heat_hot_plate'
                next_obj = 'None'
                next_wash_time += 1
            elif ((chop_time == self.mdp.chopping_time) or (subtask == 'chop_onion')) and ((wash_time == self.mdp.wash_time) or (subtask == 'heat_hot_plate')) and (subtask != 'pickup_hot_plate'):
                actions = 'pickup_hot_plate'
                next_obj = 'hot_plate'
                next_wash_time = -1
            else:
                actions = 'pickup_meat'
                next_obj = 'meat'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                if chop_time < 0: next_chop_time = 0 # doesn't change since no avaliable board to drop

            elif player_obj == 'meat':
                actions = 'drop_meat'
                next_obj = 'None'
                next_num_item_in_pot = 1

            elif player_obj == 'plate':
                actions = 'drop_plate'
                next_obj = 'None'
                if wash_time < 0: next_wash_time = 0 # doesn't change since no avaliable sink to drop

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions = 'pickup_steak'
                next_obj = 'steak'
                next_num_item_in_pot = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions = 'drop_hot_plate'
                next_obj = 'None'

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = -1

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = -1

            elif (player_obj == 'dish'):
                actions = 'deliver_dish'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj, world_info, subtask)
                raise ValueError()
            
        if next_chop_time < 0:
            next_chop_time = 'None'
        elif next_chop_time > self.mdp.chopping_time:
            next_chop_time = self.mdp.chopping_time

        if next_wash_time < 0:
            next_wash_time = 'None'
        elif next_wash_time > self.mdp.wash_time:
            next_wash_time = self.mdp.wash_time

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        next_state_keys = next_state_keys + '_' + subtask

        return actions, next_state_keys
    
    def stochastic_state_transition(self, player_obj, world_info, human_state=None):#, next_human_state=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]; chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[3:]
        # other_obj = human_state[0]; 
        subtask = human_state
        next_subtask = human_state
        
        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = []
        next_state_keys = []
        next_obj = player_obj; next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
            human_obj = 'None'
        elif subtask == 'pickup_steak':
            human_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            human_obj = 'steak'
        else:
            human_obj = subtask_obj

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat'):
                actions += ['pickup_meat']
                next_state_keys += self.gen_state_key('meat', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (chop_time < 0) and (human_obj != 'onion'):
                actions += ['pickup_onion']
                next_state_keys += self.gen_state_key('onion', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (wash_time < 0) and (human_obj != 'plate' or human_obj != 'hot_plate'): # consider the human_object not hot plate since not priority
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if ((chop_time >= 0) and (chop_time < self.mdp.chopping_time) and (subtask != 'chop_onion')):# or ((chop_time < 0) and (human_obj == 'onion')):
                actions += ['chop_onion']
                next_state_keys += self.gen_state_key('None', next_chop_time + 1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            # this is added with the assumption that you take the chop_onion action while the human has not finished their subtask, therefore, the state should not change to the state of after completing chop_onion action.
            if ((chop_time < 0) and (human_obj == 'onion')):
                # actions += ['chop_onion']
                # next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                pass # since usually after dropping the onion on the board, the person who dropped it will continue to chop it
            
            if ((wash_time >= 0) and (wash_time < self.mdp.wash_time) and (subtask != 'heat_hot_plate')): # or ((wash_time < 0) and (human_obj == 'plate')):
                actions += ['heat_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time + 1, next_num_item_in_pot, orders, next_subtask)
                
            # this is added with the assumption that you take the heat_hot_plate action while the human has not finished their subtask, therefore, the state should not change to the state of after completing head_hot_plate action.
            if ((wash_time < 0) and (human_obj == 'plate')):
                # actions += ['heat_hot_plate']
                # next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                pass # since usually after dropping the plate in the sink, the person who dropped it will continue to heat it
            
            # Note: removed the condition that the robot can still pick up the hot_plate when the human has not finished heating the last step
            # if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and ((wash_time >= self.mdp.wash_time) or (subtask == 'heat_hot_plate')) and (subtask != 'pickup_hot_plate'):
            if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and wash_time >= self.mdp.wash_time and (subtask != 'pickup_hot_plate'):
                actions += ['pickup_hot_plate']
                next_state_keys += self.gen_state_key('hot_plate', next_chop_time, -1, next_num_item_in_pot, orders, next_subtask)
            
            if len(actions) == 0:
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

        else:
            if player_obj == 'onion':
                actions += ['drop_onion']
                if chop_time < 0: # doesn't change since no avaliable board to drop
                    next_state_keys += self.gen_state_key('None', 0, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif player_obj == 'meat':
                actions += ['drop_meat']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, 1, orders, next_subtask)

            elif player_obj == 'plate':
                actions += ['drop_plate']
                if wash_time < 0: # doesn't change since no avaliable sink to drop
                    next_state_keys += self.gen_state_key('None', next_chop_time, 0, next_num_item_in_pot, orders, next_subtask)
                elif wash_time > 0 and chop_time >= self.mdp.chopping_time and num_item_in_pot >= self.mdp.num_items_for_steak: # do not drop plate since we are in the plating stage and no other actions are availiable
                    next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions += ['pickup_steak']
                next_state_keys += self.gen_state_key('steak', next_chop_time, next_wash_time, 0, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions += ['drop_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'dish'):
                actions += ['deliver_dish']
                if len(orders) >= 1:
                    # next_orders = orders[:-1]
                    orders.pop(0)
                #     next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, next_orders, next_subtask)
                # else:
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            else:
                print(player_obj, world_info, next_subtask)
                raise ValueError()

        return actions, next_state_keys
    
    def consider_subtask_stochastic_state_transition(self, player_obj, world_info, human_state=None):#, next_human_state=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]; chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[3:]
        # other_obj = human_state[0]; 
        subtask = human_state
        next_subtask = human_state
        
        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = []
        next_state_keys = []
        next_obj = player_obj; next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
            human_obj = 'None'
        elif subtask == 'pickup_steak':
            human_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            human_obj = 'steak'
        else:
            human_obj = subtask_obj

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat') and (subtask != 'pickup_meat'):
                actions += ['pickup_meat']
                next_state_keys += self.gen_state_key('meat', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (chop_time < 0) and (human_obj != 'onion') and (subtask != 'pickup_onion'):
                actions += ['pickup_onion']
                next_state_keys += self.gen_state_key('onion', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (wash_time < 0) and (human_obj != 'plate' or human_obj != 'hot_plate') and (subtask != 'pickup_plate'): # consider the human_object not hot plate since not priority
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if ((chop_time >= 0) and (chop_time < self.mdp.chopping_time) and (subtask != 'chop_onion')):# or ((chop_time < 0) and (human_obj == 'onion')):
                actions += ['chop_onion']
                next_state_keys += self.gen_state_key('None', next_chop_time + 1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            # this is added with the assumption that you take the chop_onion action while the human has not finished their subtask, therefore, the state should not change to the state of after completing chop_onion action.
            if ((chop_time < 0) and (human_obj == 'onion')):
                # actions += ['chop_onion']
                # next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                pass # since usually after dropping the onion on the board, the person who dropped it will continue to chop it
            
            if ((wash_time >= 0) and (wash_time < self.mdp.wash_time) and (subtask != 'heat_hot_plate')): # or ((wash_time < 0) and (human_obj == 'plate')):
                actions += ['heat_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time + 1, next_num_item_in_pot, orders, next_subtask)
                
            # this is added with the assumption that you take the heat_hot_plate action while the human has not finished their subtask, therefore, the state should not change to the state of after completing head_hot_plate action.
            if ((wash_time < 0) and (human_obj == 'plate')):
                # actions += ['heat_hot_plate']
                # next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                pass # since usually after dropping the plate in the sink, the person who dropped it will continue to heat it
            
            # Note: removed the condition that the robot can still pick up the hot_plate when the human has not finished heating the last step
            # if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and ((wash_time >= self.mdp.wash_time) or (subtask == 'heat_hot_plate')) and (subtask != 'pickup_hot_plate'):
            if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and wash_time >= self.mdp.wash_time and (subtask != 'pickup_hot_plate'):
                actions += ['pickup_hot_plate']
                next_state_keys += self.gen_state_key('hot_plate', next_chop_time, -1, next_num_item_in_pot, orders, next_subtask)
            
            if len(actions) == 0:
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

        else:
            if player_obj == 'onion':
                actions += ['drop_onion']
                if chop_time < 0: # doesn't change since no avaliable board to drop
                    next_state_keys += self.gen_state_key('None', 0, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif player_obj == 'meat':
                actions += ['drop_meat']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, 1, orders, next_subtask)

            elif player_obj == 'plate':
                actions += ['drop_plate']
                if wash_time < 0: # doesn't change since no avaliable sink to drop
                    next_state_keys += self.gen_state_key('None', next_chop_time, 0, next_num_item_in_pot, orders, next_subtask)
                elif wash_time > 0 and chop_time >= self.mdp.chopping_time and num_item_in_pot >= self.mdp.num_items_for_steak: # do not drop plate since we are in the plating stage and no other actions are availiable
                    next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions += ['pickup_steak']
                next_state_keys += self.gen_state_key('steak', next_chop_time, next_wash_time, 0, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions += ['drop_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'dish'):
                actions += ['deliver_dish']
                if len(orders) >= 1:
                    # next_orders = orders[:-1]
                    orders.pop(0)
                #     next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, next_orders, next_subtask)
                # else:
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            else:
                print(player_obj, world_info, next_subtask)
                raise ValueError()

        return actions, next_state_keys
    
    def non_subtask_stochastic_state_transition(self, player_obj, world_info, human_state=None):#, next_human_state=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]; chop_time = world_info[1]; wash_time = world_info[2]; orders = [] if len(world_info) < 4 else world_info[3:]
        # other_obj = human_state[0]; 
        subtask = human_state
        next_subtask = human_state
        
        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = []
        next_state_keys = []
        next_obj = player_obj; next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
            human_obj = 'None'
        elif subtask == 'pickup_steak':
            human_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            human_obj = 'steak'
        else:
            human_obj = subtask_obj

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat'):
                actions += ['pickup_meat']
                next_state_keys += self.gen_state_key('meat', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (chop_time < 0) and (human_obj != 'onion'):
                actions += ['pickup_onion']
                next_state_keys += self.gen_state_key('onion', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (wash_time < 0) and (human_obj != 'plate' or human_obj != 'hot_plate'): # consider the human_object not hot plate since not priority
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
            if (chop_time >= 0) and (chop_time < self.mdp.chopping_time):# and (subtask != 'chop_onion')):# or ((chop_time < 0) and (human_obj == 'onion')):
                actions += ['chop_onion']
                next_state_keys += self.gen_state_key('None', next_chop_time + 1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            # this is added with the assumption that you take the chop_onion action while the human has not finished their subtask, therefore, the state should not change to the state of after completing chop_onion action.
            if ((chop_time < 0) and (human_obj == 'onion')):
                # actions += ['chop_onion']
                # next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                pass # since usually after dropping the onion on the board, the person who dropped it will continue to chop it
            
            if (wash_time >= 0) and (wash_time < self.mdp.wash_time):# and (subtask != 'heat_hot_plate')): # or ((wash_time < 0) and (human_obj == 'plate')):
                actions += ['heat_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time + 1, next_num_item_in_pot, orders, next_subtask)
                
            # this is added with the assumption that you take the heat_hot_plate action while the human has not finished their subtask, therefore, the state should not change to the state of after completing head_hot_plate action.
            if ((wash_time < 0) and (human_obj == 'plate')):
                # actions += ['heat_hot_plate']
                # next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                pass # since usually after dropping the plate in the sink, the person who dropped it will continue to heat it
            
            # Note: removed the condition that the robot can still pick up the hot_plate when the human has not finished heating the last step
            # if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and ((wash_time >= self.mdp.wash_time) or (subtask == 'heat_hot_plate')) and (subtask != 'pickup_hot_plate'):
            if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and wash_time >= self.mdp.wash_time:# and (subtask != 'pickup_hot_plate'):
                actions += ['pickup_hot_plate']
                next_state_keys += self.gen_state_key('hot_plate', next_chop_time, -1, next_num_item_in_pot, orders, next_subtask)
            
            if len(actions) == 0:
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

        else:
            if player_obj == 'onion':
                actions += ['drop_onion']
                if chop_time < 0: # doesn't change since no avaliable board to drop
                    next_state_keys += self.gen_state_key('None', 0, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif player_obj == 'meat':
                actions += ['drop_meat']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, 1, orders, next_subtask)

            elif player_obj == 'plate':
                actions += ['drop_plate']
                if wash_time < 0: # doesn't change since no avaliable sink to drop
                    next_state_keys += self.gen_state_key('None', next_chop_time, 0, next_num_item_in_pot, orders, next_subtask)
                elif wash_time > 0 and chop_time >= self.mdp.chopping_time and num_item_in_pot >= self.mdp.num_items_for_steak: # do not drop plate since we are in the plating stage and no other actions are availiable
                    next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions += ['pickup_steak']
                next_state_keys += self.gen_state_key('steak', next_chop_time, next_wash_time, 0, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions += ['drop_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'dish'):
                actions += ['deliver_dish']
                if len(orders) >= 1:
                    # next_orders = orders[:-1]
                    orders.pop(0)
                #     next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, next_orders, next_subtask)
                # else:
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot, orders, next_subtask)

            else:
                print(player_obj, world_info, next_subtask)
                raise ValueError()

        return actions, next_state_keys
    
    def gen_world_info_list(self, next_chop_time, next_wash_time, next_num_item_in_pot, orders):
        if next_chop_time < 0:
            next_chop_time = 'None'
        elif next_chop_time > self.mdp.chopping_time:
            next_chop_time = self.mdp.chopping_time

        if next_wash_time < 0:
            next_wash_time = 'None'
        elif next_wash_time > self.mdp.wash_time:
            next_wash_time = self.mdp.wash_time

        nxt_world = [next_num_item_in_pot, next_chop_time, next_wash_time]
        for order in orders:
            nxt_world.append(order)

        return [nxt_world]
    
    def gen_state_key(self, next_obj, next_chop_time, next_wash_time, next_num_item_in_pot, orders, subtask):
        if next_chop_time != 'None':
            next_chop_time = int(next_chop_time)
        else:
            next_chop_time = -1

        if next_wash_time != 'None':
            next_wash_time = int(next_wash_time)
        else:
            next_wash_time = -1
        
        next_num_item_in_pot = int(next_num_item_in_pot)


        if next_chop_time < 0:
            next_chop_time = 'None'
        elif next_chop_time > self.mdp.chopping_time:
            next_chop_time = self.mdp.chopping_time

        if next_wash_time < 0:
            next_wash_time = 'None'
        elif next_wash_time > self.mdp.wash_time:
            next_wash_time = self.mdp.wash_time

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        next_state_keys = next_state_keys + '_' + subtask

        return [next_state_keys]
    
    def world_state_to_mdp_state_key(self, state, player, other_player, subtask=None, RETURN_NON_SUBTASK=False, RETURN_OBJ=False):
        state_str = super().gen_state_dict_key(state, player, other_player=other_player, RETURN_OBJ=RETURN_OBJ)
        
        # if other_player.held_object is not None:
        #     other_player_obj = other_player.held_object.name

        if RETURN_NON_SUBTASK:
            return state_str
         
        state_str = state_str + '_' + subtask

        return state_str
    
    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]
    
    def gen_state_dict_key(self, p0_obj, p1_obj, num_item_in_pot, chop_time, wash_time, orders, subtasks):

        player_obj = p0_obj if p0_obj is not None else 'None'
        # other_player_obj = p1_obj if p1_obj is not None else 'None'

        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + '_' + str(order)

        state_strs = []
        for subtask in subtasks:
            state_strs.append(str(player_obj)+'_'+str(num_item_in_pot)+'_' + str(chop_time) + '_' + str(wash_time) + '_'+ order_str + '_' + subtask)

        return state_strs
    
    def map_action_to_location(self, world_state, action, obj, p0_obj=None, player_idx=None, counter_drop=True, state_dict=None, occupied_goal=True):

        other_obj = world_state.players[1-player_idx].held_object.name if world_state.players[1-player_idx].held_object is not None else 'None'
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []

        WAIT = False # If wait becomes true, one player has to wait for the other player to finish its current task and its next task
        counter_obj = self.mdp.get_counter_objects_dict(world_state, list(self.mdp.terrain_pos_dict['X']))
        if action == 'pickup' and obj in ['onion', 'plate', 'meat', 'hot_plate']:
            if p0_obj != 'None' and p0_obj != obj and counter_drop:
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += counter_obj[obj]
                if obj == 'onion':
                    location += self.mdp.get_onion_dispenser_locations()
                elif obj == 'plate':
                    location += self.mdp.get_dish_dispenser_locations()
                elif obj == 'meat':
                    location += self.mdp.get_meat_dispenser_locations()
                elif obj == 'hot_plate':
                    location += (self.mdp.get_sink_status(world_state)['full'] + self.mdp.get_sink_status(world_state)['ready'])

                    if len(location) == 0:
                        WAIT = True
                        location += self.mdp.get_sink_status(world_state)['empty']
                    
        elif action == 'pickup' and obj == 'garnish':
            if p0_obj != 'steak' and p0_obj != 'None' and counter_drop:
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += counter_obj[obj]
                location += (self.mdp.get_chopping_board_status(world_state)['full'] + self.mdp.get_chopping_board_status(world_state)['ready'])

                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_chopping_board_status(world_state)['empty']
                    return location, WAIT
            
        elif action == 'pickup' and obj == 'steak':
            if p0_obj != 'hot_plate' and p0_obj != 'None' and counter_drop:
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += counter_obj[obj]
                location += (self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict))

                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_empty_pots(pots_states_dict)
                    return location, WAIT

        elif action == 'drop':
            if p0_obj == obj or p0_obj == 'None':
                if obj == 'meat':
                    location += self.mdp.get_empty_pots(pots_states_dict)
                    if len(location) == 0 and other_obj != 'meat' and occupied_goal:
                        WAIT = True
                        location += self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

                elif obj == 'onion':
                    location += self.mdp.get_chopping_board_status(world_state)['empty']
                    if len(location) == 0 and other_obj != 'onion' and occupied_goal:
                        WAIT = True
                        location += self.mdp.get_chopping_board_status(world_state)['ready']+ self.mdp.get_chopping_board_status(world_state)['full']

                elif obj == 'plate':
                    location += self.mdp.get_sink_status(world_state)['empty']
                    if len(location) == 0 and other_obj != 'plate' and occupied_goal:
                        WAIT = True
                        location += self.mdp.get_sink_status(world_state)['ready'] + self.mdp.get_sink_status(world_state)['full']

                elif (obj == 'hot_plate' or obj == 'steak') and counter_drop:
                    WAIT = True
                    location += self.drop_item(world_state)

            else:
                WAIT = True
            
            if (len(location) == 0 or WAIT) and counter_drop:
                location += self.drop_item(world_state)
            
        elif action == 'deliver':
            if p0_obj != 'dish' and p0_obj != 'None':
                WAIT = True
                location += self.drop_item(world_state)
            elif  p0_obj == 'None':
                WAIT = True
                location += counter_obj[obj]
                if len(location) == 0:
                    location += self.mdp.get_key_objects_locations()
            else:
                location += self.mdp.get_serving_locations()

        elif action == 'chop' and obj == 'onion':
            if p0_obj != 'None':
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += self.mdp.get_chopping_board_status(world_state)['full']
                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_chopping_board_status(world_state)['empty'] + self.mdp.get_chopping_board_status(world_state)['ready'] 
            
        elif action == 'heat' and obj == 'hot_plate':
            if p0_obj != 'None':
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += self.mdp.get_sink_status(world_state)['full']
                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_sink_status(world_state)['empty'] + self.mdp.get_sink_status(world_state)['ready'] 
        else:
            print(p0_obj, action, obj)
            ValueError()

        
        return location, WAIT
    
    def _shift_same_goal_pos(self, new_positions, change_idx):
        
        pos = new_positions[change_idx][0]
        ori = new_positions[change_idx][1]
        new_pos = pos; new_ori = ori
        if self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])
        else:
            print('pos = ', pos)
            ValueError()
        
        new_positions[change_idx] = (new_pos, new_ori)

        return new_positions[0], new_positions[1]

    def mdp_action_state_to_world_state(self, action_idx, ori_state_idx, ori_world_state, with_argmin=False):
        new_world_state = ori_world_state.deepcopy()
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)
        robot_obj = ori_world_state.players[self.agent_index].held_object.name if ori_world_state.players[self.agent_index].held_object is not None else 'None'

        possible_agent_motion_goals, AI_WAIT = self.map_action_to_location(ori_world_state, self.action_dict[action][0], self.action_dict[action][1], p0_obj= robot_obj, player_idx=self.agent_index) 
        if new_world_state.players[(1-self.agent_index)].held_object != None:
            human_obj = new_world_state.players[(1-self.agent_index)].held_object.name
        else:
            human_obj = 'None'
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(ori_world_state, self.action_dict[mdp_state_obj[-1]][0], self.action_dict[mdp_state_obj[-1]][1], p0_obj=human_obj, player_idx=(1-self.agent_index)) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
        # get next position for AI agent
        agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[0].pos_and_or, possible_agent_motion_goals, with_motion_goal=True) # select the feature position that is closest to current player's position in world state
        new_agent_pos = agent_feature_pos if agent_feature_pos is not None else new_world_state.players[0].get_pos_and_or()
        human_cost, human_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[1].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else new_world_state.players[1].get_pos_and_or()
        # print(new_agent_pos, new_human_pos)

        if new_agent_pos == new_human_pos:
            new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos], np.argmax(np.array([agent_cost, human_cost])))
            # print('after shift =', new_agent_pos, new_human_pos)

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].remove_object()
        if mdp_state_obj[0] != 'None' and mdp_state_obj[0] != 'dish':
            new_world_state.players[0].held_object = ObjectState(mdp_state_obj[0], new_agent_pos)
        new_world_state.players[0].update_pos_and_or(new_agent_pos[0], new_agent_pos[1])

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].remove_object()
        mdp_state_obj_action = mdp_state_obj[-1].split('_')[0]
        mdp_state_obj_obj = '_'.join(mdp_state_obj[-1].split('_')[1:])
        if (mdp_state_obj_action != 'pickup' and mdp_state_obj_action != 'chop' and mdp_state_obj_action != 'heat') and mdp_state_obj_obj != 'dish':
            new_world_state.players[1].held_object = ObjectState(mdp_state_obj_obj, new_human_pos)
        new_world_state.players[1].update_pos_and_or(new_human_pos[0], new_human_pos[1])

        total_cost = max([agent_cost, human_cost]) # in rss paper is max
        if AI_WAIT or HUMAN_WAIT: # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area length)
            total_cost = agent_cost + human_cost + ((self.mdp.width-1)+(self.mdp.height-1))/2

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost
    
    def map_state_to_subtask(self, mdp_state_obj, next_mdp_state_obj):
        '''
        The next state's subtask can be completed or not. The key is to know what the robot's subtasks can be.
        '''

        ## edge case: if robot keeps holding plate, then return action subtask to pickup plate
        if mdp_state_obj[0] == 'plate' and next_mdp_state_obj[0] == 'plate':
            return ['pickup'], ['plate']

        human_subtask = mdp_state_obj[-1]
        human_action = human_subtask.split('_')[0]
        human_obj = '_'.join(human_subtask.split('_')[1:])
        actions, objs = [], []

        for i in range(4):
            state_obj = mdp_state_obj[i]
            if state_obj != next_mdp_state_obj[i]:
                if i == 0: # ai agent holding 
                    if next_mdp_state_obj[i] == 'None': # dropped object
                        if state_obj == 'dish':
                            actions.append('deliver')
                        else:
                            actions.append('drop')
                        objs.append(state_obj)

                    elif state_obj == 'None':
                        actions.append('pickup')
                        objs.append(next_mdp_state_obj[i])

                    elif state_obj == 'hot_plate' and next_mdp_state_obj[i] == 'steak':
                        actions.append('pickup')
                        objs.append('steak')
                    
                    elif state_obj == 'steak' and next_mdp_state_obj[i] == 'dish':
                        actions.append('pickup')
                        objs.append('garnish')

                elif i == 2:
                    tmp_state_obj = state_obj if state_obj != 'None' else -1
                    tmp_next_state_obj = next_mdp_state_obj[i] if next_mdp_state_obj[i] != 'None' else -1
                    if (tmp_state_obj < tmp_next_state_obj) and (tmp_state_obj >= 0):# and (human_subtask != 'chop_onion'): # status of chop board
                        actions.append('chop')
                        objs.append('onion')

                elif i == 3:
                    tmp_state_obj = state_obj if state_obj != 'None' else -1
                    tmp_next_state_obj = next_mdp_state_obj[i] if next_mdp_state_obj[i] != 'None' else -1
                    if (tmp_state_obj < tmp_next_state_obj) and (tmp_state_obj >= 0):# and (human_subtask != 'heat_hot_plate'): # status of sink
                        actions.append('heat')
                        objs.append('hot_plate')
        
        if len(actions) > 1:
            if human_obj in objs:
                rmv_idx = objs.index(human_obj)
                objs.pop(rmv_idx)
                actions.pop(rmv_idx)

        if len(actions) == 0:
            agent_actions, _ = self.stochastic_state_transition(mdp_state_obj[0], mdp_state_obj[1:-1], human_state=next_mdp_state_obj[-1])

            for agent_action in agent_actions:
                action, obj = agent_action.split('_')[0], '_'.join(agent_action.split('_')[1:])
                actions.append(action)
                objs.append(obj)

        return actions, objs

    def mdp_state_to_world_state(self, ori_state_idx, next_state_idx, ori_world_state, with_argmin=False, cost_mode='max', consider_wait=True, occupied_goal=True):
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]

        next_mdp_state_key = self.get_key_from_value(self.state_idx_dict, next_state_idx)
        next_mdp_state_obj = self.state_dict[next_mdp_state_key]

        new_world_states = []
        if mdp_state_obj == next_mdp_state_obj:
            if with_argmin:
                new_world_states.append([ori_world_state, 0, [ori_world_state.players[0].get_pos_and_or(), ori_world_state.players[1].get_pos_and_or()]])
            else:
                new_world_states.append([ori_world_state, 0])

            return np.array(new_world_states, dtype=object)
    
        # compute the robot's action such that we know what the world state should be like as the robot moves to the next world position based on its action. Can have more than one outcome.
        agent_actions, agent_action_objs = self.map_state_to_subtask(mdp_state_obj, next_mdp_state_obj)

        for agent_action, agent_action_obj in zip(agent_actions, agent_action_objs):
            new_world_state = ori_world_state.deepcopy()
            
            # get the human's action to location
            if new_world_state.players[abs(1-self.agent_index)].held_object != None:
                human_obj = new_world_state.players[abs(1-self.agent_index)].held_object.name
            else:
                human_obj = 'None'
            
            possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(ori_world_state, self.action_dict[mdp_state_obj[-1]][0], self.action_dict[mdp_state_obj[-1]][1], p0_obj=human_obj, player_idx=(abs(1-self.agent_index)), occupied_goal=occupied_goal) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
            human_cost, human_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[1-self.agent_index].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
            new_human_pos = human_feature_pos if human_feature_pos is not None else new_world_state.players[1-self.agent_index].get_pos_and_or()
            
            # carry out the human's subtask and update the world
            if self.agent_index == 0:
                new_world_state = self.jmp.derive_state(ori_world_state, (ori_world_state.players[self.agent_index].pos_and_or, new_human_pos), [('stay', 'interact')])
            else:
                new_world_state = self.jmp.derive_state(ori_world_state, (new_human_pos, ori_world_state.players[self.agent_index].pos_and_or), [('interact', 'stay')])
            
            # compute the robot's action to location given that the human has taken their action
            possible_agent_motion_goals, AI_WAIT = self.map_action_to_location(new_world_state, agent_action, agent_action_obj, p0_obj=(new_world_state.players[self.agent_index].held_object.name if new_world_state.players[self.agent_index].held_object is not None else 'None'), player_idx=self.agent_index, occupied_goal=occupied_goal)
            # get next position for AI agent
            agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(new_world_state.players[self.agent_index].pos_and_or, possible_agent_motion_goals, with_motion_goal=True) # select the feature position that is closest to current player's position in world state
            new_agent_pos = agent_feature_pos if agent_feature_pos is not None else new_world_state.players[self.agent_index].get_pos_and_or()
            
            # print(new_agent_pos, new_human_pos)

            # check if the goal overlaps, if so, move the human out of the way as the human action is already performed
            if new_agent_pos == new_human_pos:
                new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos], 1-self.agent_index)
                # new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos], np.argmax(np.array([agent_cost, human_cost])))
                # print('after shift =', new_agent_pos, new_human_pos)

            # # update next position for human (uses mdp_state_obj since the subtask is the human's action)
            # if new_world_state.players[1-self.agent_index].has_object():
            #     new_world_state.players[1-self.agent_index].remove_object()
            # mdp_state_obj_action = mdp_state_obj[-1].split('_')[0]
            # mdp_state_obj_obj = '_'.join(mdp_state_obj[-1].split('_')[1:])
            # # the held object should be the object after completeing the mdp_state_obj_action
            # if (mdp_state_obj_action in ['chop', 'heat', 'deliver', 'drop']):
            #     human_obj = 'None'
            # elif mdp_state_obj[-1] == 'pickup_garnish':
            #     new_world_state.players[1-self.agent_index].held_object = ObjectState('dish', new_human_pos)
            # else:
            #     new_world_state.players[1-self.agent_index].held_object = ObjectState(mdp_state_obj_obj, new_human_pos)
            # new_world_state.players[1-self.agent_index].update_pos_and_or(new_human_pos[0], new_human_pos[1])

            # # update next position for AI agent (uses the next_mdp_state_obj)
            # if new_world_state.players[self.agent_index].has_object():
            #     new_world_state.players[self.agent_index].remove_object()
            # if next_mdp_state_obj[0] != 'None':
            #     new_world_state.players[self.agent_index].held_object = ObjectState(next_mdp_state_obj[0], new_agent_pos)
            # new_world_state.players[self.agent_index].update_pos_and_or(new_agent_pos[0], new_agent_pos[1])
            # if (AI_WAIT or HUMAN_WAIT) and consider_wait:
            #     if AI_WAIT: 
            #         total_cost = human_cost + 1
            #         new_world_state = self.jmp.derive_state(ori_world_state, (ori_world_state.players[self.agent_index].pos_and_or, new_human_pos), [('stay', 'interact')])
            #     if HUMAN_WAIT: 
            #         total_cost = agent_cost + 1
            #         new_world_state = self.jmp.derive_state(ori_world_state, (new_agent_pos, ori_world_state.players[1-self.agent_index].pos_and_or), [('interact', 'stay')])
            # else:

            # get new world with agent's action performed
            if self.agent_index == 0:
                new_world_state = self.jmp.derive_state(new_world_state, (new_agent_pos, new_human_pos), [('interact', 'stay')])
            else:
                new_world_state = self.jmp.derive_state(new_world_state, (new_human_pos, new_agent_pos), [('stay', 'interact')])
                
            # add interaction cost or stay cost when goal overlaps
            agent_cost += 1
            human_cost += 1
        
            if cost_mode == 'average':
                total_cost = sum([agent_cost, human_cost])/2
            elif cost_mode == 'sum':
                total_cost = sum([agent_cost, human_cost])
            elif cost_mode == 'max':
                total_cost = max([agent_cost, human_cost]) # in rss paper is max
            elif cost_mode == 'robot':
                total_cost = agent_cost
            elif cost_mode == 'human':
                total_cost = human_cost

            if (AI_WAIT or HUMAN_WAIT) and consider_wait:
                if AI_WAIT: total_cost = human_cost
                if HUMAN_WAIT: total_cost = agent_cost
            # if AI_WAIT or HUMAN_WAIT: # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area length)
            #     total_cost = agent_cost + human_cost + ((self.mdp.width-1)+(self.mdp.height-1))/2

            if with_argmin:
                new_world_states.append([new_world_state, total_cost, [new_agent_pos, new_human_pos], [AI_WAIT, HUMAN_WAIT]])
                # return new_world_state, total_cost, [new_agent_pos, new_human_pos]
            else:
                new_world_states.append([new_world_state, total_cost])

        return np.array(new_world_states, dtype=object)

    def world_to_state_keys(self, world_state, player, other_player, belief):
        mdp_state_keys = []
        used_belief = []
        for i, b in enumerate(belief):
            mdp_state_key = self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i))
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i)))
                used_belief.append(i)
        return [mdp_state_keys, used_belief]

    def joint_action_cost(self, world_state, goal_pos_and_or, COST_OF_STAY=1, RETURN_PLAN=False, PLAN_COST='short'):
        joint_action_plan, end_motion_state, plan_costs = self.jmp.get_low_level_action_plan(world_state.players_pos_and_or, goal_pos_and_or, merge_one=True)
        # joint_action_plan, end_state, plan_costs = self.mlp.get_embedded_low_level_action_plan(world_state, goal_pos_and_or, other_agent, other_agent_idx)
        # print('joint_action_plan =', joint_action_plan, '; plan_costs =', plan_costs)

        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0

        num_of_non_stay_actions = len([a for a in joint_action_plan if a[0] != Action.STAY])
        num_of_stay_actions = len([a for a in joint_action_plan if a[0] == Action.STAY])

        if PLAN_COST == 'short':
            total_cost = min(plan_costs)
        elif PLAN_COST == 'average':
            total_cost = sum(plan_costs)/2 if sum(plan_costs) > 0 else 0
        elif PLAN_COST == 'max':
            total_cost = max(plan_costs)
        elif PLAN_COST == 'robot':
            total_cost = plan_costs[0]
        elif PLAN_COST == 'human':
            total_cost = plan_costs[1]
        else:
            total_cost = max(plan_costs)
        
        if RETURN_PLAN:
            return np.array(joint_action_plan), total_cost
        
        return joint_action_plan[0], total_cost # num_of_non_stay_actions+num_of_stay_actions*COST_OF_STAY # in rss paper is max(plan_costs)

    def step(self, world_state, mdp_state_keys_and_belief, belief, agent_idx, low_level_action=False, observation=None):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dic will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        if low_level_action:
            next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
            action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
            qmdp_q = np.zeros((Action.NUM_ACTIONS, len(belief)), dtype=float)
            
        else:
            next_state_v = np.zeros((len(belief), len(self.action_dict)), dtype=float)
            action_cost = np.zeros((len(belief), len(self.action_dict)), dtype=float)
            qmdp_q = np.zeros((len(self.action_dict), len(belief)), dtype=float)
        # ml_action_to_low_action = np.zeros()

        # for each subtask, obtain next mdp state but with low level location based on finishing excuting current action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []

        mdp_state_keys = mdp_state_keys_and_belief[0]
        used_belief = mdp_state_keys_and_belief[1]
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None and belief[used_belief[i]] > 0.01:
                agent_action_idx_arr, next_mdp_state_idx_arr = np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
                nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr): # action_idx is encoded subtask action
                    # print('action_idx =', action_idx)
                    next_state_idx = next_mdp_state_idx_arr[j]
                    after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx, mdp_state_idx, world_state, with_argmin=True)
                    value_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=25, search_time_limit=0.01)
                    joint_action, one_step_cost = self.joint_action_cost(world_state, after_action_world_state.players_pos_and_or)  
                    if one_step_cost > 1000000:
                        one_step_cost = 1000000
                    # print('joint_action =', joint_action, 'one_step_cost =', one_step_cost)
                    # print('Action.ACTION_TO_INDEX[joint_action[agent_idx]] =', Action.ACTION_TO_INDEX[joint_action[agent_idx]])
                    if not low_level_action:
                        # action_idx: are subtask action dictionary index
                        next_state_v[i, action_idx] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, action_idx] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    # print('action_idx =', self.get_key_from_value(self.action_idx_dict, action_idx), '; mdp_state_key =', mdp_state_key, '; next_state_key =', self.get_key_from_value(self.state_idx_dict, next_state_idx))
                    # print('next_state_v =', next_state_v[i])
        # print('action_cost =', action_cost)

        q = self.compute_Q(belief, next_state_v, action_cost)
        print('q value =', q)
        print('next_state_value:', next_state_v)
        print('action_cost:', action_cost)
        action_idx = self.get_best_action(q)
        print('get_best_action =', action_idx, '=', self.get_key_from_value(self.action_idx_dict, action_idx))
        print("It took {} seconds for this step".format(time.time() - start_time))
        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
        return action_idx, self.action_dict[self.get_key_from_value(self.action_idx_dict, action_idx)], low_level_action
    
    def observe(self, world_state, robot_agent, human_agent):
        num_item_in_pot, chop_state, sink_state = 0, -1, -1
        robot_agent_obj, human_agent_obj = None, None
        for obj in world_state.objects.values():
            if obj.name == 'hot_plate' and obj.position in self.mdp.get_sink_locations():
                wash_time = obj.state
                if wash_time > sink_state:
                    sink_state = wash_time
            elif obj.name == 'steak' and obj.position in self.mdp.get_pot_locations():
                _, _, cook_time = obj.state
                if cook_time > 0:
                    num_item_in_pot = 1
            elif obj.name == 'garnish' and obj.position in self.mdp.get_chopping_board_locations():
                chop_time = obj.state
                if chop_time > chop_state:
                    chop_state = chop_time

        if chop_state < 0:
            chop_state = None
        if sink_state < 0:
            sink_state = None

        if robot_agent.held_object is not None:
            robot_agent_obj = robot_agent.held_object.name
        if human_agent.held_object is not None:
            human_agent_obj = human_agent.held_object.name

        return [num_item_in_pot, chop_state, sink_state], robot_agent_obj, human_agent_obj

    def kb_to_state_info(self, kb):
        num_item_in_pot = 0
        pots = kb['pot_states']['steak']
        non_emtpy_pots = pots['cooking'] + pots['ready']
        if len(non_emtpy_pots) > 0:
            num_item_in_pot = 1
        
        chop_time = -1
        non_empty_boards = kb['chop_states']['ready'] + kb['chop_states']['full']
        if len(non_empty_boards) > 0:
            if kb[non_empty_boards[0]] is not None:
                chop_time = kb[non_empty_boards[0]].state
            else:
                raise ValueError()
        
        wash_time = -1
        non_empty_sink = kb['sink_states']['ready'] + kb['sink_states']['full']
        if len(non_empty_sink) > 0:
            if kb[non_empty_sink[0]] is not None:
                wash_time = kb[non_empty_sink[0]].state
            else:
                raise ValueError()

        robot_obj = kb['other_player'].held_object.name if kb['other_player'].held_object is not None else 'None'

        return num_item_in_pot, chop_time, wash_time, robot_obj
    
    def belief_update(self, world_state, agent_player, observed_info, human_player, belief_vector, prev_dist_to_feature, greedy=False, vision_limit=False):
        """
        Update belief based on both human player's game logic and also it's current position and action.
        Belief shape is an array with size equal the length of subtask_dict.
        human_player is the human agent class that is in the simulator.
        NOTE/TODO: the human_player needs to be simulated when we later use an actual human to run experiments.
        """
        new_prev_dist_to_feature = {}
        [world_num_item_in_pot, world_chop_time, world_wash_time] = observed_info

        # human knowledge base: the observed information should be updated according to the human's vision limitation.
        # if vision_limit:
        self.sim_human_model.update(world_state)

        # self.human_knowledge['pot_states'] = self.sim_human_model.knowledge_base['pot_states']
        # self.human_knowledge['sink_states'] = self.sim_human_model.knowledge_base['sink_states']
        # self.human_knowledge['chop_states'] = self.sim_human_model.knowledge_base['chop_states']
        # self.human_knowledge['other_player'] = self.sim_human_model.knowledge_base['other_player']
        
        num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(self.sim_human_model.knowledge_base)
        
        print('Robot understanding of human obs = ', num_item_in_pot, chop_time, wash_time)
        # else:
        #     num_item_in_pot = world_num_item_in_pot
        #     chop_time = world_chop_time
        #     wash_time = world_wash_time
        #     robot_obj = agent_player.held_object.name if agent_player.held_object is not None else 'None'

        start_time = time.time()

        distance_trans_belief = np.zeros((len(belief_vector), len(belief_vector)), dtype=float)
        human_pos_and_or = world_state.players[1].pos_and_or
        # agent_pos_and_or = self.sim_human_model.knowledge_base['other_player'].pos_and_or

        subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

        # get next position for human
        human_obj = human_player.held_object.name if human_player.held_object is not None else 'None'
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.zeros((len(belief_vector)), dtype=float)
        print('subtasks:', self.subtask_dict.keys())
        for i, belief in enumerate(belief_vector):
            ## estimating next subtask based on game logic
            game_logic_prob[i] = self._is_valid_object_subtask_pair(subtask_key[i], num_item_in_pot, chop_time, wash_time, vision_limit=vision_limit, human_obj=human_obj, other_agent_holding=robot_obj)*1.0
    
            ## tune subtask estimation based on current human's position and action (use minimum distance between features)
            possible_motion_goals, WAIT = self.map_action_to_location(world_state, self.subtask_dict[subtask_key[i]][0], self.subtask_dict[subtask_key[i]][1], p0_obj=human_obj, player_idx=(1-self.agent_index), counter_drop=True) 
            
            # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
            if WAIT:
                dist_belief_prob[i] = 0
            else:
                human_dist_cost, feature_pos = self.mp.min_cost_to_feature(human_pos_and_or, possible_motion_goals, with_argmin=True) # select the feature position that is closest to current player's position in world state

                if str(feature_pos) not in prev_dist_to_feature:
                    prev_dist_to_feature[str(feature_pos)] = human_dist_cost

                # TODO: the offset of dist_belief_prob to avoid being smaller than 0, is the average distance between features. 
                # dist_belief_prob[i] = ((self.mdp.height+self.mdp.width)/2) + (prev_dist_to_feature[str(feature_pos)] - human_dist_cost)
                dist_belief_prob[i] = (prev_dist_to_feature[str(feature_pos)] - human_dist_cost)

                # if human_dist_cost == 0: then you are at the target, hence we set the probability to be 1
                if human_dist_cost == 0: 
                    dist_belief_prob[i] = 1

                # dist_belief_prob[i] = (self.mdp.height+self.mdp.width) - human_dist_cost if human_dist_cost < np.inf else (self.mdp.height + self.mdp.width)

                # update distance to feature
                new_prev_dist_to_feature[str(feature_pos)] = human_dist_cost

        prev_dist_to_feature = new_prev_dist_to_feature
        print('prev_dist_to_feature =', prev_dist_to_feature)

        # Note: prev_dist_to_feature is not updating all positions, but only the possible goals. Hence the game logic should multipy the distance first then the distance is normalized
        # print('human_dist_cost =', human_dist_cost)
        if game_logic_prob.sum() > 0.0:
            game_logic_prob /= game_logic_prob.sum()
        else:
            game_logic_prob[2] = 1.0 # since no next subtask for human, set the pickup plate to 1 (this matches the human logic)
        game_logic_prob[game_logic_prob <= 0.000001] = 0.000001
        print('game_logic_prob =', game_logic_prob)
        print('dist_belief_prob =', dist_belief_prob)
        # let all dist_belief_prob > 0
        offset = min(dist_belief_prob)
        for i in range(len(dist_belief_prob)):
            if dist_belief_prob[i] != 0.0:
                dist_belief_prob[i] -= offset
        # only update the belief if the distance differences provides information (aka not all zeros)
        dist_belief_prob *= game_logic_prob
        if dist_belief_prob.sum() > 0.0:
            dist_belief_prob[dist_belief_prob <= 0.000001] = 0.000001
            dist_belief_prob /= dist_belief_prob.sum()
            print('dist_belief_prob =', dist_belief_prob)

        print('original belief:', belief_vector)
        new_belief = belief_vector*game_logic_prob
        new_belief /= new_belief.sum()
        new_belief = new_belief*0.7 + dist_belief_prob*0.3

        new_belief /= new_belief.sum()
        count_small = len(new_belief[new_belief <= 0.01])
        new_belief[new_belief > 0.01] -= (0.01*count_small)
        new_belief[new_belief <= 0.01] = 0.01
        print('new_belief =', new_belief)
        print('max belif =', list(self.subtask_dict.keys())[np.argmax(new_belief)])
        # print("It took {} seconds for belief update".format(time.time() - start_time))

        return new_belief, prev_dist_to_feature

    def compute_V(self, next_world_state, mdp_state_key, belief_prob=None, belief_idx=None, search_depth=200, search_time_limit=10, add_rewards=False, gamma=False, debug=False):
        start_time = time.time()
        next_world_state_str = str(next_world_state)
        flag = True #False
        
        if belief_prob is not None:
            if belief_prob[belief_idx] <= 0.03:
                return 0
                
        # if ((next_world_state_str, mdp_state_key) not in self.world_state_cost_dict):
        #     flag = True
        #     # save computation: if belief is low, no need to compute next state value since later in comput Q value, the row will have a small value
        #     if belief_prob is not None:
        #         if belief_prob[belief_idx] <= 0.01:
        #             self.world_state_cost_dict[(next_world_state_str,mdp_state_key)] = 0
        #             return self.world_state_cost_dict[(next_world_state_str,mdp_state_key)]
                    
        # elif (self.world_state_cost_dict[(next_world_state_str,mdp_state_key)] == 0):
        #     flag = True
        # elif math.isinf((self.world_state_cost_dict[(next_world_state_str,mdp_state_key)])):
        #     flag = True

        if flag:                    
            delivery_horizon=2
            h_fn=Steak_Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = start_world_state.order_list[:delivery_horizon]
            
            expand_fn = lambda state, ori_state_key: self.get_successor_states(state, ori_state_key, add_rewards=add_rewards)
            goal_fn = lambda ori_state_key: len(self.state_dict[ori_state_key][4:-1]) == 0
            heuristic_fn = lambda state: h_fn(state)

            search_problem = SearchTree(start_world_state, goal_fn, expand_fn, heuristic_fn, debug=self.debug)
            path_end_state, cost, over_limit, end_state_key = search_problem.bounded_A_star_graph_search(qmdp_root=mdp_state_key, info=False, cost_limit=search_depth, time_limit=search_time_limit, gamma=gamma, debug=self.debug)

            if over_limit or cost > 10000:
                cost = self.optimal_plan_cost(path_end_state, cost)

            if add_rewards:
                state_obj = self.state_dict[end_state_key].copy()
                player_obj, pot_state, chop_state, sink_state = state_obj[:4]
                remaining_orders = state_obj[4:-1]
                human_subtask = state_obj[-1]
                # [player_obj, pot_state, chop_state, sink_state, remaining_orders] = self.world_state_to_mdp_state_key(path_end_state, path_end_state.players[0], path_end_state.players[1], subtask=None, RETURN_NON_SUBTASK=True, RETURN_OBJ=True)
                human_action = human_subtask.split('_')[0]
                human_obj = '_'.join(human_subtask.split('_')[1:])
                if human_action in ['pickup', 'chop', 'heat'] and human_obj not in ['garnish', 'steak']:
                    human_obj = 'None'
                elif human_action == 'pickup' and human_obj == 'garnish':
                    human_obj = 'steak'
                elif human_action == 'pickup' and human_obj == 'steak':
                    human_obj = 'hot_plate'

                delta_cost = (-40)*len(remaining_orders)
                if chop_state == 'None' or chop_state == None:
                    chop_state = 0
                else:
                    chop_state += 3 #1
                if sink_state == 'None' or sink_state == None:
                    sink_state = 0
                else:
                    sink_state += 3 #1
                    
                # the rewards are given in two phases. One where you prep and the other where you collect and plate.
                # if len(remaining_orders) > 0:
                # if player_obj not in ['hot_plate', 'dish', 'steak'] and human_obj not in ['hot_plate', 'dish', 'steak']:
                # delta_cost += ((4)*pot_state + (2)*chop_state + (1)*sink_state)
                delta_cost += ((5)*pot_state + (1)*chop_state + (1)*sink_state)
                # else:
                if 'hot_plate' in [player_obj, human_obj]:
                    delta_cost += 12 # 9
                if 'steak' in [player_obj, human_obj]:
                    delta_cost += 20 # 18
                if 'dish' in [player_obj, human_obj]:
                    delta_cost += 30
                
                if player_obj not in ['None', None, 'hot_plate', 'steak', 'dish']:
                    delta_cost += 1
                if human_obj not in ['None', None, 'hot_plate', 'steak', 'dish']:
                    delta_cost += 1
                
                if len(remaining_orders) == 0:
                    delta_cost = 100 # set to 100 such that would not optimize after termination and focus only on decreasing cost

                if self.debug: 
                    print('world info:', player_obj, pot_state, chop_state, sink_state, remaining_orders, human_obj)
                    print('delta_cost:cost', (delta_cost), cost)
                
                cost -= (delta_cost)*2*(0.9**(2-len(remaining_orders)))#*((3-len(remaining_orders))/3)
            self.world_state_cost_dict[(next_world_state_str,mdp_state_key)] = cost

        # print('self.world_state_cost_dict length =', len(self.world_state_cost_dict))            
        # print("It took {} seconds for computing value, {}, and {}".format((time.time() - start_time), next_world_state, mdp_state_key))
        return max((self.mdp.height*self.mdp.width)*5 - self.world_state_cost_dict[(next_world_state_str,mdp_state_key)], 0)

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(start_world_state, start_world_state.players[0], start_world_state.players[1], subtask)

    def compute_Q(self, b, v, c, gamma=0.9):
        print('b =', b)
        print('v =', v)
        print('c =', c)

        return b@((v*gamma)+c)

    def get_best_action(self, q):
        return np.argmax(q)

    def init_mdp(self):
        self.init_actions()
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_transition()

    def compute_mdp(self, filename):
        start_time = time.time()

        final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # print('Total states =', self.num_states, '; Total actions =', self.num_actions)

        # print("It took {} seconds to create HumanSubtaskQMDPPlanner".format(time.time() - start_time))
        self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return 


class SteakKnowledgeBasePlanner(SteakHumanSubtaskQMDPPlanner):
    def __init__(self, mdp, mlp_params, state_dict={}, state_idx_dict={}, action_dict={}, action_idx_dict={}, transition_matrix=None, reward_matrix=None, policy_matrix=None, value_matrix=None, num_states=0, num_rounds=0, epsilon=0.01, discount=0.8, jmp=None, vision_limited_human=None, debug=False):
        super().__init__(mdp, mlp_params, state_dict, state_idx_dict, action_dict, action_idx_dict, transition_matrix, reward_matrix, policy_matrix, value_matrix, num_states, num_rounds, epsilon, discount, jmp, vision_limited_human)

        self.list_objs = ['None', 'meat', 'onion', 'plate', 'hot_plate', 'steak', 'dish']
        self.kb_space = (self.mdp.num_items_for_steak+1) * (self.mdp.chopping_time+1) * (self.mdp.wash_time+1) * len(self.list_objs) # num_in_pot_item * chop_time * wash_time * holding
        self.init_kb_idx_dict()
        self.debug = debug
    
    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False, jmp=None, vision_limited_human=None, debug=False):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'steak_knowledge_aware_qmdp' + '.pkl'

        if force_compute_all:
            mdp_planner = SteakKnowledgeBasePlanner(mdp, mlp_params, vision_limited_human=vision_limited_human, debug=debug)
            mdp_planner.compute_mdp(filename)
            return mdp_planner
        
        try:
            mdp_planner = SteakKnowledgeBasePlanner.from_qmdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = SteakKnowledgeBasePlanner(mdp, mlp_params, jmp=jmp, vision_limited_human=vision_limited_human, debug=debug)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        if info:
            print("Loaded SteakKnowledgeBasePlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner
    
    def init_kb_idx_dict(self):
        self.kb_idx_dict = {}
        count = 0
        for num_item_for_steak in range(self.mdp.num_items_for_steak+1):
            for chop_time in range(-1, self.mdp.chopping_time+1):
                for wash_time in range(-1, self.mdp.wash_time+1):
                    for obj in self.list_objs:
                        kb_key = '.'.join([str(num_item_for_steak), str(chop_time), str(wash_time), obj])
                        self.kb_idx_dict[kb_key] = count
                        count += 1

    def init_s_kb_trans_matrix(self, s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.s_kb_trans_matrix = s_kb_trans_matrix if s_kb_trans_matrix is not None else np.identity((len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

            # append original state of p1 (human) to represent unfinished subtask state transition
            p1_nxt_states.append(p1_state)
            p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)
                # consider the next state where agent 0 does not complete action execution
                p0_not_complete_key = self.get_key_from_value(self.state_dict, [state_obj[0]]+p1_nxt_world_info[i]+[p1_nxt_state])
                if (p0_not_complete_key not in next_state_keys) and p0_not_complete_key != state_key:
                    next_state_keys.append(p0_not_complete_key)
                for next_state_key in next_state_keys:
                    next_state_idx = self.state_idx_dict[next_state_key]
                    self.s_kb_trans_matrix[state_idx, next_state_idx] += 1.0

            if np.sum(self.s_kb_trans_matrix[state_idx]) > 0.0:
                self.s_kb_trans_matrix[state_idx] /= np.sum(self.s_kb_trans_matrix[state_idx])

        self.s_kb_trans_matrix[self.s_kb_trans_matrix == 0.0] = 0.000001

    def init_sprim_s_kb_trans_matrix(self, sprim_s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.sprim_s_kb_trans_matrix = sprim_s_kb_trans_matrix if sprim_s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict)), dtype=object)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            self.sprim_s_kb_trans_matrix[state_idx] = {}

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

            # append original state of p1 (human) to represent unfinished subtask state transition
            p1_nxt_states.append(p1_state)
            p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)
                
                for next_state_key in next_state_keys:
                    needed_kb_keys_for_next = self.get_needed_kb_key(next_state_key, p1_nxt_state, state_key)
                    next_state_idx = self.state_idx_dict[next_state_key]
                    for needed_kb_key_for_next in needed_kb_keys_for_next:
                        if (self.kb_idx_dict[needed_kb_key_for_next], next_state_idx) in self.sprim_s_kb_trans_matrix[state_idx].keys():
                            self.sprim_s_kb_trans_matrix[state_idx][(self.kb_idx_dict[needed_kb_key_for_next], next_state_idx)] += 1.0
                        else:
                            self.sprim_s_kb_trans_matrix[state_idx][(self.kb_idx_dict[needed_kb_key_for_next], next_state_idx)] = 1.0
                
                # add for self trainsition probability
                needed_kb_keys_for_next = self.get_needed_kb_key(state_key, p1_state, state_key)
                for needed_kb_key_for_next in needed_kb_keys_for_next:
                    self.sprim_s_kb_trans_matrix[state_idx][(self.kb_idx_dict[needed_kb_key_for_next], state_idx)] = 1.0
  
        if len(self.sprim_s_kb_trans_matrix[state_idx]) > 0:
            sum_count = np.sum(list(self.sprim_s_kb_trans_matrix[state_idx].values()))
            for k,_ in self.sprim_s_kb_trans_matrix[state_idx].items():
                self.sprim_s_kb_trans_matrix[state_idx][k] /= sum_count

    def init_optimal_s_kb_trans_matrix(self, optimal_s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.optimal_s_kb_trans_matrix = optimal_s_kb_trans_matrix if optimal_s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)#np.identity((len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            # p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)
            p1_nxt_states, p1_nxt_world_info = self.world_based_human_state_subtask_transition(p1_state, world_info, other_agent_obj=p0_state)

            # append original state of p1 (human) to represent unfinished subtask state transition
            # p1_nxt_states.append(p1_state)
            # p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.consider_subtask_stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)
                # _, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)#, human_state=p1_nxt_state)
                # consider the next state where agent 0 does not complete action execution
                # p0_not_complete_key = self.get_key_from_value(self.state_dict, [state_obj[0]]+p1_nxt_world_info[i]+[p1_nxt_state])
                # if (p0_not_complete_key not in next_state_keys) and p0_not_complete_key != state_key:
                    # next_state_keys.append(p0_not_complete_key)
                for next_state_key in next_state_keys:
                    next_state_idx = self.state_idx_dict[next_state_key]
                    self.optimal_s_kb_trans_matrix[state_idx, next_state_idx] += 1.0

            if np.sum(self.optimal_s_kb_trans_matrix[state_idx]) > 0.0:
                self.optimal_s_kb_trans_matrix[state_idx] /= np.sum(self.optimal_s_kb_trans_matrix[state_idx])

        self.optimal_s_kb_trans_matrix[self.optimal_s_kb_trans_matrix == 0.0] = 0.000001

    def init_optimal_non_subtask_s_kb_trans_matrix(self, optimal_non_subtask_s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.optimal_non_subtask_s_kb_trans_matrix = optimal_non_subtask_s_kb_trans_matrix if optimal_non_subtask_s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)#np.identity((len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            # p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)
            p1_nxt_states, p1_nxt_world_info = self.world_based_human_state_subtask_transition(p1_state, world_info, other_agent_obj=p0_state)

            # append original state of p1 (human) to represent unfinished subtask state transition
            # p1_nxt_states.append(p1_state)
            # p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.non_subtask_stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)#, human_state=p1_nxt_state)
                # consider the next state where agent 0 does not complete action execution
                # p0_not_complete_key = self.get_key_from_value(self.state_dict, [state_obj[0]]+p1_nxt_world_info[i]+[p1_nxt_state])
                # if (p0_not_complete_key not in next_state_keys) and p0_not_complete_key != state_key:
                    # next_state_keys.append(p0_not_complete_key)
                for next_state_key in next_state_keys:
                    next_state_idx = self.state_idx_dict[next_state_key]
                    self.optimal_non_subtask_s_kb_trans_matrix[state_idx, next_state_idx] += 1.0

            if np.sum(self.optimal_non_subtask_s_kb_trans_matrix[state_idx]) > 0.0:
                self.optimal_non_subtask_s_kb_trans_matrix[state_idx] /= np.sum(self.optimal_non_subtask_s_kb_trans_matrix[state_idx])

        self.optimal_non_subtask_s_kb_trans_matrix[self.optimal_non_subtask_s_kb_trans_matrix == 0.0] = 0.000001

    
    def get_successor_states(self, start_world_state, start_state_key, debug=False, add_rewards=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][4:-1]) == 0: # [p0_obj, num_item_in_soup, orders, p1_obj, subtask] 
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        # next_state_idx_arr = np.where(self.s_kb_trans_matrix[ori_state_idx] > 0.000001)[0]
        next_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[ori_state_idx] > 0.000001)[0]
        
        start_time = time.time()
        for next_state_idx in next_state_idx_arr:
            # Note: don't go to occupied goals since we asuume these action have to success
            next_world_states_info = self.mdp_state_to_world_state(ori_state_idx, next_state_idx, start_world_state, consider_wait=True, occupied_goal=False)
            for next_world_state, cost in next_world_states_info:
                if add_rewards:
                    next_state_obj = self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)].copy()
                    player_obj, pot_state, chop_state, sink_state = next_state_obj[:4]
                    remaining_orders = next_state_obj[4:-1]
                    human_subtask = next_state_obj[-1]
                    human_action = human_subtask.split('_')[0]
                    human_obj = '_'.join(human_subtask.split('_')[1:])

                    if human_action in ['pickup', 'chop', 'heat'] and human_obj not in ['garnish', 'steak']:
                        human_obj = 'None'
                    elif human_action == 'pickup' and human_obj == 'garnish':
                        human_obj = 'steak'
                    elif human_action == 'pickup' and human_obj == 'steak':
                        human_obj = 'hot_plate'
                    
                    delta_cost = (-7)*len(remaining_orders)
                    if chop_state == 'None' or chop_state == None:
                        chop_state = 0
                    else:
                        chop_state += 1
                    if sink_state == 'None' or sink_state == None:
                        sink_state = 0
                    else:
                        sink_state += 1
                    # the rewards are given in two phases. One where you prep and the other where you collect and plate.
                    # print('world info:', player_obj, pot_state, chop_state, sink_state, remaining_orders)
                    # if len(remaining_orders) > 0:
                    # if player_obj not in ['hot_plate', 'dish', 'steak'] and human_obj not in ['hot_plate', 'dish', 'steak']:
                    delta_cost += ((1.5)*pot_state + (0.4)*chop_state + (0.4)*sink_state)
                    # else:
                    if 'hot_plate' in [player_obj, human_obj]:
                        delta_cost += 2.5
                    if 'steak' in [player_obj, human_obj]:
                        delta_cost += 4.5
                    if 'dish' in [player_obj, human_obj]:
                        delta_cost += 6.5
                        # print('delta_cost:cost', delta_cost, cost)
                    # cost -= ((delta_cost*(3-len(remaining_orders)))/10)
                    # cost -= (delta_cost)*(1.1**(2-len(remaining_orders)))/5
                successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))
                if debug: print('From {} to {} costs {} in {} seconds.'.format(self.get_key_from_value(self.state_idx_dict, ori_state_idx), self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time()-start_time))

        return successor_states

    # def next_state_prob(self, world_state, agent_player, observed_info, human_player, vision_limit=True):
    #     """
    #     Update belief based on both human player's game logic and also it's current position and action.
    #     Belief shape is an array with size equal the length of subtask_dict.
    #     human_player is the human agent class that is in the simulator.
    #     NOTE/TODO: the human_player needs to be simulated when we later use an actual human to run experiments.
    #     """
    #     [world_num_item_in_pot, world_chop_time, world_wash_time] = observed_info
    #     belief_vector = np.zeros(len(self.subtask_dict))

    #     # human knowledge base: the observed information should be updated according to the human's vision limitation.
    #     # if vision_limit:
    #     new_knowledge_base = self.sim_human_model.get_knowledge_base(world_state, rollout_kb=True)

    #     num_item_in_pot = 0
    #     pots = new_knowledge_base['pot_states']['steak']
    #     non_emtpy_pots = pots['cooking'] + pots['ready']
    #     if len(non_emtpy_pots) > 0:
    #         num_item_in_pot = 1
        
    #     chop_time = -1
    #     non_empty_boards = new_knowledge_base['chop_states']['ready'] + new_knowledge_base['chop_states']['full']
    #     if len(non_empty_boards) > 0:
    #         chop_time = new_knowledge_base[non_empty_boards[0]].state
        
    #     wash_time = -1
    #     non_empty_sink = new_knowledge_base['sink_states']['ready'] + new_knowledge_base['sink_states']['full']
    #     if len(non_empty_sink) > 0:
    #         if new_knowledge_base[non_empty_sink[0]] is not None:
    #             wash_time = new_knowledge_base[non_empty_sink[0]].state
    #         else:
    #             wash_time = self.mdp.wash_time

    #     robot_obj = new_knowledge_base['other_player'].held_object.name if new_knowledge_base['other_player'].held_object is not None else 'None'
        
    #     print('Robot understanding of human obs = ', num_item_in_pot, chop_time, wash_time)
    #     # else:
    #     #     num_item_in_pot = world_num_item_in_pot
    #     #     chop_time = world_chop_time
    #     #     wash_time = world_wash_time
    #     #     robot_obj = agent_player.held_object.name if agent_player.held_object is not None else 'None'

    #     subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

    #     # get next position for human
    #     human_obj = human_player.held_object.name if human_player.held_object is not None else 'None'
    #     game_logic_prob = np.zeros((len(belief_vector)), dtype=float)

    #     print('subtasks:', self.subtask_dict.keys())
    #     for i in range(len(belief_vector)):
    #         ## estimating next subtask based on game logic
    #         game_logic_prob[i] = self._is_valid_object_subtask_pair(subtask_key[i], num_item_in_pot, chop_time, wash_time, vision_limit=vision_limit, human_obj=human_obj, other_agent_holding=robot_obj)*1.0
    
    #     game_logic_prob /= game_logic_prob.sum()
    #     game_logic_prob[game_logic_prob == 0.0] = 0.000001
    #     print('game_logic_prob =', game_logic_prob)

    #     return game_logic_prob
    
    # def cond_on_high_step(self, world_state, mdp_state_keys_and_belief, belief, agent_idx, low_level_action=True, observation=None, explicit_communcation=False):
    #     """
    #     Compute plan cost that starts from the next qmdp state defined as next_state_v().
    #     Compute the action cost of excuting a step towards the next qmdp state based on the
    #     current low level state information.

    #     next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
    #         the action_dic will be representing the 6 low level action index (north, south...).
    #         If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
    #     """
    #     start_time = time.time()
    #     next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
    #     action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)

    #     # for each subtask, obtain next mdp state but with low level location based on finishing excuting current action and subtask
    #     nxt_possible_mdp_state = []

    #     mdp_state_keys = mdp_state_keys_and_belief[0]
    #     used_belief = mdp_state_keys_and_belief[1]
    #     for i, mdp_state_key in enumerate(mdp_state_keys):
    #         mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
    #         if mdp_state_idx is not None and belief[used_belief[i]] > 0.01:
    #             agent_action_idx_arr, next_mdp_state_idx_arr = np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
    #             nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
    #             track_laction_freq = {}
    #             track_trans = []

    #             for j, action_idx in enumerate(agent_action_idx_arr): # action_idx is encoded subtask action
    #                 next_state_idx = next_mdp_state_idx_arr[j] # high-level transition probability
    #                 after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx, mdp_state_idx, world_state, with_argmin=True)
    #                 value_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=25, search_time_limit=0.01)
    #                 joint_action, one_step_cost = self.joint_action_cost(world_state, after_action_world_state.players_pos_and_or)
    #                 nxt_obs_info = self.observe(after_action_world_state, after_action_world_state.players[agent_idx], after_action_world_state.players[abs(agent_idx-1)])
    #                 nxt_state_kb_prob = self.next_state_prob(after_action_world_state, after_action_world_state.players[agent_idx], nxt_obs_info, after_action_world_state.players[abs(agent_idx-1)], vision_limit=True)

    #                 if one_step_cost > 1000000:
    #                     one_step_cost = 1000000

    #                 if joint_action[0] in track_laction_freq.keys():
    #                     track_laction_freq[Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += 1
    #                 else:
    #                     track_laction_freq[Action.ACTION_TO_INDEX[joint_action[agent_idx]]] = 1

    #                 # compute the probability of low-level action to high-level action
    #                 track_trans.append([joint_action, action_idx, next_state_idx, nxt_state_kb_prob, value_cost, one_step_cost])

    #                 # print('joint_action =', joint_action, 'one_step_cost =', one_step_cost)
    #                 # print('Action.ACTION_TO_INDEX[joint_action[agent_idx]] =', Action.ACTION_TO_INDEX[joint_action[agent_idx]])
                
    #             for trans in track_trans:
    #                 [joint_action, action_idx, next_state_idx, nxt_state_kb_prob, value_cost, one_step_cost] = trans
    #                 laction = Action.ACTION_TO_INDEX[joint_action[agent_idx]]
    #                 prob_high_cond_low_action = track_laction_freq[laction]/sum(list(track_laction_freq.values()))
    #                 next_state_v[i, laction] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx] * nxt_state_kb_prob[self.subtask_idx_dict[self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1]]] * prob_high_cond_low_action)
                    
    #                 # if explicit_communcation:
    #                 #     nxt_state_kb_prob
    #                 # print(next_state_v[i, action_idx])

    #                 ## compute one step cost with joint motion considered
    #                 action_cost[i, laction] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

    #     q = self.compute_Q(belief, next_state_v, action_cost)
    #     print('q value =', q)
    #     print('next_state_value:', next_state_v)
    #     print('action_cost:', action_cost)
    #     action_idx = self.get_best_action(q)
    #     print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
    #     print("It took {} seconds for this step".format(time.time() - start_time))
        
    #     return Action.INDEX_TO_ACTION[action_idx], None, low_level_action

    def kb_based_state(self, state_obj_key, kb, kb_key=False):
        new_state_obj = self.state_dict[state_obj_key].copy()
        
        if kb_key:
            num_item_in_pot, chop_time, wash_time, robot_obj = kb.split('.')
            num_item_in_pot = int(num_item_in_pot)
            if chop_time != 'None':
                chop_time = int(chop_time)
            if wash_time != 'None':
                wash_time = int(wash_time)
        else:
            # update state info with kb
            num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        
        new_state_obj[0] = robot_obj
        new_state_obj[1] = num_item_in_pot
        new_state_obj[2] = chop_time
        new_state_obj[3] = wash_time

        return new_state_obj 

    def kb_based_human_subtask_state(self, curr_human_subtask, kb, kb_key=False, human_obj=None):
        
        if human_obj is None:
            curr_subtask_action = curr_human_subtask.split('_')[0]
            curr_subtask_obj = '_'.join(curr_human_subtask.split('_')[1:])
            if (curr_subtask_action in ['pickup', 'chop', 'heat']) and curr_subtask_obj not in ['steak', 'garnish']:
                human_obj = 'None'
            elif curr_human_subtask == 'pickup_steak':
                human_obj = 'hot_plate'
            elif curr_human_subtask == 'pickup_garnish':
                human_obj = 'steak'
            else:
                human_obj = curr_subtask_obj

        if kb_key:
            num_item_in_pot, chop_time, wash_time, robot_obj = kb.split('.')
            num_item_in_pot = int(num_item_in_pot)
            if chop_time != 'None':
                chop_time = int(chop_time)
            else:
                chop_time = -1
            if wash_time != 'None':
                wash_time = int(wash_time)
            else:
                wash_time = -1
        else:
            # update state info with kb
            num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        
        actions = []
        next_world_infos = []
        if human_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat') and (robot_obj != 'meat'):
                actions.append('pickup_meat')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            if (chop_time < 0) and (human_obj != 'onion') and (robot_obj != 'onion'):
                actions.append('pickup_onion')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            if (wash_time < 0) and (human_obj != 'plate') and (robot_obj != 'plate'):
                actions.append('pickup_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            if ((chop_time >= 0) and (chop_time < self.mdp.chopping_time)) or ((chop_time < 0) and (human_obj == 'onion')):
                actions.append('chop_onion')
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time
                next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])
            
            if ((wash_time >= 0) and (wash_time < self.mdp.wash_time)) or ((wash_time < 0) and (human_obj == 'plate')):
                actions.append('heat_hot_plate')
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

            if (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
                actions.append('pickup_hot_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, 'None'])
            
            if len(actions) == 0:
                actions.append('pickup_meat')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        else:
            if human_obj == 'onion':
                actions.append('drop_onion')
                next_chop_time = chop_time
                if chop_time < 0: next_chop_time = 0 # doesn't change since no avaliable board to drop
                next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])

            elif human_obj == 'meat':
                actions.append('drop_meat')
                next_num_item_in_pot = 1
                next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

            elif human_obj == 'plate':
                actions.append('drop_plate')
                next_wash_time = wash_time
                if wash_time < 0: next_wash_time = 0 # doesn't change since no avaliable sink to drop
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

            elif (human_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions.append('pickup_steak')
                next_num_item_in_pot = 0
                next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

            elif (human_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions.append('drop_hot_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            elif (human_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions.append('pickup_garnish')
                next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

            elif (human_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions.append('pickup_garnish')
                next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

            elif (human_obj == 'dish'):
                actions.append('deliver_dish')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            else:
                print(human_obj, [robot_obj, num_item_in_pot, chop_time, wash_time])
                raise ValueError()
            
        return actions, next_world_infos
    
    def det_kb_based_human_subtask_state(self, curr_human_subtask, kb, kb_key=False, human_obj=None):
        
        if human_obj is None:
            curr_subtask_action = curr_human_subtask.split('_')[0]
            curr_subtask_obj = '_'.join(curr_human_subtask.split('_')[1:])
            if (curr_subtask_action in ['pickup', 'chop', 'heat']) and curr_subtask_obj not in ['steak', 'garnish']:
                human_obj = 'None'
            elif curr_human_subtask == 'pickup_steak':
                human_obj = 'hot_plate'
            elif curr_human_subtask == 'pickup_garnish':
                human_obj = 'steak'
            else:
                human_obj = curr_subtask_obj

        if kb_key:
            num_item_in_pot, chop_time, wash_time, robot_obj = kb.split('.')
            num_item_in_pot = int(num_item_in_pot)
            if chop_time != 'None':
                chop_time = int(chop_time)
            else:
                chop_time = -1
            if wash_time != 'None':
                wash_time = int(wash_time)
            else:
                wash_time = -1
        else:
            # update state info with kb
            num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        
        actions = []
        next_world_infos = []
        if human_obj == 'None':
            if ((chop_time >= 0) and (chop_time < self.mdp.chopping_time)) or ((chop_time < 0) and (human_obj == 'onion')):
                actions.append('chop_onion')
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time
                next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])
            elif ((wash_time >= 0) and (wash_time < self.mdp.wash_time)) or ((wash_time < 0) and (human_obj == 'plate')):
                actions.append('heat_hot_plate')
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

            elif (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat') and (robot_obj != 'meat'):
                actions.append('pickup_meat')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            elif (chop_time < 0) and (human_obj != 'onion') and (robot_obj != 'onion'):
                actions.append('pickup_onion')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            elif (wash_time < 0) and (human_obj != 'plate') and (robot_obj != 'plate'):
                actions.append('pickup_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            elif (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
                actions.append('pickup_hot_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, 'None'])
            
            if len(actions) == 0:
                actions.append('pickup_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        else:
            if human_obj == 'onion':
                actions.append('drop_onion')
                next_chop_time = chop_time
                if chop_time < 0: next_chop_time = 0 # doesn't change since no avaliable board to drop
                next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])

            elif human_obj == 'meat':
                actions.append('drop_meat')
                next_num_item_in_pot = 1
                next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

            elif human_obj == 'plate':
                actions.append('drop_plate')
                next_wash_time = wash_time
                if wash_time < 0: next_wash_time = 0 # doesn't change since no avaliable sink to drop
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

            elif (human_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions.append('pickup_steak')
                next_num_item_in_pot = 0
                next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

            elif (human_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions.append('drop_hot_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            elif (human_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions.append('pickup_garnish')
                next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

            elif (human_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions.append('pickup_garnish')
                next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

            elif (human_obj == 'dish'):
                actions.append('deliver_dish')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            else:
                print(human_obj, [robot_obj, num_item_in_pot, chop_time, wash_time])
                raise ValueError()
            
        return actions, next_world_infos
    
    def kb_compare(self, kb1, kb2):
        # kb1['pot_states']
        # kb1['sink_states']
        # kb1['chop_states']
        # kb1['other_player']

        # diff = DeepDiff(kb1, kb2)
        # return len(diff['values_changed']) == 0

        return all((kb2.get(k) == v for k, v in kb1.items()))

    # def next_kb_prob(self, start_world_state, goal_kb, h_fn=None, delivery_horizon=4, debug=False, search_time=0.01, other_agent_plan=None):
    #     """
    #     Solves A* Search problem to find sequence of low-level actions and observe the knowledge base of the new world state.

    #     Returns:
    #         ml_plan (list): plan not including starting state in form
    #             [(joint_action, successor_state), ..., (joint_action, goal_state)]
    #         cost (int): A* Search cost
    #     """
    #     start_state = start_world_state.deepcopy()
    #     start_kb = self.sim_human_model.get_knowledge_base(start_state, rollout_kb=True)
    #     if self.kb_compare(start_kb, goal_kb):
    #         return None, 0, 1

    #     if start_state.order_list is None:
    #         start_state.order_list = ["any"] * delivery_horizon
    #     else:
    #         start_state.order_list = start_state.order_list[:delivery_horizon]
        
    #     expand_fn = lambda state, depth: self.get_kb_successor_states(state, other_agent_plan[depth])
    #     goal_fn = lambda state: self.kb_compare(self.sim_human_model.get_knowledge_base(state, rollout_kb=True), goal_kb)
    #     heuristic_fn = Steak_Heuristic(self.mp).simple_heuristic

    #     search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
    #     # path_end_state, cost, over_limit = search_problem.bounded_A_star_graph_search(qmdp_root=mdp_state_key, info=False, cost_limit=search_depth, time_limit=search_time_limit)
    #     ml_plan, cost = search_problem.coupled_A_star_graph_search(info=False, time_limit=search_time, path_limit=len(other_agent_plan)-1)
    #     prob = 1
    #     if cost > 0:
    #         if other_agent_plan is not None:
    #             prob = 1/pow(Action.NUM_ACTIONS, cost)
    #         else:
    #             prob = 1/pow(Action.NUM_ACTIONS*Action.NUM_ACTIONS, cost)
        
    #     if len(ml_plan) > 1:
    #         action_plan, _ = zip(*ml_plan)
    #         return (action_plan[1][0][-1], action_plan[1][1][-1]), cost, prob
    #     else:
    #         return None, 0, 1

    def get_kb_key(self, kb):
        num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        kb_key = '.'.join([str(num_item_in_pot), str(chop_time), str(wash_time), str(robot_obj)])
        return kb_key

    def get_needed_kb_key(self, next_state_key, next_human_subtask, ori_state_key):
        ori_state_obj = ori_state_key.split('_')
        next_state_obj = next_state_key.split('_')

        kb_keys = []
        # Possibility 1: the next state key directly reflects the needed kb, so all values come from the next key values
        if next_state_obj[0] == 'hot' and next_state_obj[1] == 'plate':
            next_robot_obj = 'hot_plate'
            next_num_item_in_pot, next_chop_time, next_wash_time = next_state_obj[2:5]
        else:
            next_robot_obj = next_state_obj[0]
            next_num_item_in_pot, next_chop_time, next_wash_time = next_state_obj[1:4]

        if next_chop_time == 'None':
            next_chop_time = -1
        if next_wash_time == 'None':
            next_wash_time = -1

        kb_keys.append('.'.join([str(next_num_item_in_pot), str(next_chop_time), str(next_wash_time), str(next_robot_obj)]))

        # Possibility 2: robot changes world based
        kb_robot_obj = next_robot_obj
        if ori_state_obj[0] == 'hot' and ori_state_obj[1] == 'plate':
            ori_robot_obj = 'hot_plate'
            ori_num_item_in_pot, ori_chop_time, ori_wash_time = ori_state_obj[2:5]
        else:
            ori_robot_obj = ori_state_obj[0]
            ori_num_item_in_pot, ori_chop_time, ori_wash_time = ori_state_obj[1:4]

        if ori_chop_time == 'None':
            ori_chop_time = -1
        if ori_wash_time == 'None':
            ori_wash_time = -1

        if next_robot_obj == 'None': # note that we only consider changes in one step, so we assume the previous holding object is reasonable
            if int(ori_num_item_in_pot) < int(next_num_item_in_pot):
                kb_robot_obj = 'meat'
                kb_keys.append('.'.join([str(ori_num_item_in_pot), str(next_chop_time), str(next_wash_time), str(kb_robot_obj)]))

            if int(ori_chop_time) < int(next_chop_time) and int(ori_chop_time) < 0:
                kb_robot_obj = 'onion'
                kb_keys.append('.'.join([str(next_num_item_in_pot), str(ori_chop_time), str(next_wash_time), str(kb_robot_obj)]))

            if int(ori_wash_time) < int(next_wash_time) and int(ori_wash_time) < 0:
                kb_robot_obj = 'plate'
                kb_keys.append('.'.join([str(next_num_item_in_pot), str(next_chop_time), str(ori_wash_time), str(kb_robot_obj)]))

        # we do not need to consider ELSE condition: if the other state changes but the robot has something in hand, then it is assumed to be a change made previously, so we do not consider in the kb as this kb should be the kb of the previous step.

        return kb_keys

    def get_kb_successor_states(self, start_state, kb, other_agent_action=None, explore_interact=False):
        successor_kb = []
        joint_motion_actions = []
        if explore_interact: 
            explore_actions = Action.ALL_ACTIONS
        else:
            explore_actions = Action.MOTION_ACTIONS
        if other_agent_action is not None:
            for a in explore_actions:
                joint_motion_action = (a, other_agent_action) if self.agent_index == 0 else (other_agent_action, a)
                joint_motion_actions.append(joint_motion_action)
        # else:
            # joint_motion_actions = itertools.product(explore_actions, explore_actions)
        # dummy_state = start_state.deepcopy()

        for joint_action in joint_motion_actions:
            # dummy_sim_human = self.sim_human_model.deepcopy(start_state)
            # dummy_sim_human.agent_index = abs(1-self.agent_index)
            # dummy_sim_human.update(dummy_state)
            
            new_positions, new_orientations = self.mdp.compute_new_positions_and_orientations(start_state.players, joint_action)
            successor_state = self.jmp.derive_state(start_state, tuple(zip(*[new_positions, new_orientations])), [joint_action])

            successor_kb.append((self.sim_human_model.get_knowledge_base(successor_state, rollout_kb=kb), successor_state, 1))
            
            # del dummy_sim_human

        # del dummy_state
        return successor_kb

    def roll_out_for_kb(self, start_world_state, one_step_human_kb, delivery_horizon=4, debug=False, search_time=0.01, search_depth=5, other_agent_plan=None, explore_interact=False):
        start_state = start_world_state.deepcopy()
        start_kb = self.sim_human_model.get_knowledge_base(start_state, rollout_kb=one_step_human_kb)

        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon
        else:
            start_state.order_list = start_state.order_list[:delivery_horizon]
        
        expand_fn = lambda state, kb, depth: self.get_kb_successor_states(state, kb, None if depth > (len(other_agent_plan)-1) else other_agent_plan[depth], explore_interact=explore_interact)
        heuristic_fn = Steak_Heuristic(self.mp).simple_heuristic

        search_problem = SearchTree(start_state, None, expand_fn, heuristic_fn, debug=debug)
        _, _, kb_prob_dict = search_problem.bfs_track_path(lambda kb: self.get_kb_key(kb), kb_root=start_kb, info=False, time_limit=search_time, path_limit=len(other_agent_plan)-1, search_depth=search_depth)
        # _, _, kb_prob_dict = search_problem.bfs(kb_key_root=self.get_kb_key(start_kb), info=False, time_limit=search_time, path_limit=len(other_agent_plan)-1, search_depth=search_depth)
        
        return kb_prob_dict
    
    def get_human_traj_robot_stays(self, world_state, human_subtask_obj):
        # get human holding object name
        human_obj = 'None' if world_state.players[1-self.agent_index].held_object == None else world_state.players[1-self.agent_index].held_object.name

        # limit the human to take the optimal action to complete its subtask (robot's belief)
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(world_state, human_subtask_obj[0], human_subtask_obj[1], p0_obj=human_obj, player_idx=abs(1-self.agent_index)) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)

        human_cost, human_feature_pos = self.mp.min_cost_to_feature(world_state.players[abs(1-self.agent_index)].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else world_state.players[(1-self.agent_index)].get_pos_and_or()
        agent_pos = world_state.players[self.agent_index].get_pos_and_or()

        # shift by one grid if goal position overlappes with the robot agent
        if agent_pos == new_human_pos:
            _, new_human_pos = self._shift_same_goal_pos([agent_pos, new_human_pos], np.argmax(np.array([0, human_cost])))

        # get grid path from human's original position to goal
        ori_human_pos = world_state.players[(1-self.agent_index)].get_pos_and_or()
        next_las, _, _ = self.mp.get_plan(ori_human_pos, new_human_pos)

        return next_las

    def subtask_based_next_state(self, subtask, ori_world_info, next_subtask=None):
        ori_state_obj = self.state_dict[ori_world_info+'_'+subtask]
        action = subtask.split('_')[0]
        obj = '_'.join(subtask.split('_')[1:])
        tmp_state_obj = ori_state_obj.copy()
        orders = tmp_state_obj[4:-1]

        # do not change the robot agent's holding object (aka. tmp_state_obj[0])
        if action == 'drop':
            if obj == 'meat':
                if ori_state_obj[1] == 0: 
                    tmp_state_obj[1] = 1
            elif obj == 'onion':
                if ori_state_obj[2] == 'None': 
                    tmp_state_obj[2] = 0
                elif ori_state_obj[2] < self.mdp.chopping_time:
                    tmp_state_obj[2] += 1
            elif obj == 'plate':
                if ori_state_obj[3] == 'None': 
                    tmp_state_obj[3] = 0
                elif ori_state_obj[3] < self.mdp.wash_time:
                    tmp_state_obj[3] += 1
            # tmp_state_obj[0] = 'None' 
        elif action == 'pickup':
            # if obj == 'garnish':
            #     tmp_state_obj[0] = 'dish'
            # else:
            #     tmp_state_obj[0] = obj
            pass
        elif action == 'chop':
            if ori_state_obj[2] != 'None':
                if ori_state_obj[2] < self.mdp.chopping_time:
                    tmp_state_obj[2] += 1
        elif action == 'heat':
            if ori_state_obj[3] != 'None':
                if ori_state_obj[3] < self.mdp.wash_time:
                    tmp_state_obj[3] += 1
        elif action == 'deliver':
            if len(orders) > 0:
                orders.pop()
        
        new_state_obj = tmp_state_obj[:4]
        for o in orders:
            new_state_obj.append(o)
        
        if next_subtask is not None:
            new_state_obj.append(next_subtask)

        return new_state_obj
    
    def old_step(self, world_state, mdp_state_keys_and_belief, belief, agent_idx, low_level_action=True, observation=None, explicit_communcation=False, SEARCH_DEPTH=5, SEARCH_TIME=1, KB_SEARCH_DEPTH=3):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dic will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        est_next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        
        # Get current high-level state based on all possible beliefs of the human subtask
        mdp_state_keys = mdp_state_keys_and_belief[0]
        used_belief = mdp_state_keys_and_belief[1] # idx of the human subtask
        computed_v_cost = {} # a temp storage for computed value cost to save some computation time

        for i, mdp_state_key in enumerate(mdp_state_keys):
            if belief[used_belief[i]] > 0.2 or all(belief < 0.072):#belief[used_belief[i]] > 0.01:
                mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
                curr_belief = self.get_key_from_value(self.subtask_idx_dict, i)
                if self.sim_human_model is not None:
                    curr_kb = self.get_kb_key(self.sim_human_model.get_knowledge_base(world_state))
                else:
                    curr_kb = self.get_kb_key(world_state)

                # assume the robot doesn't move and compute the human's trajectory to complete its subtask. This is for later speeding up our roll out step.
                human_subtask = self.subtask_dict[self.state_dict[mdp_state_key][-1]]
                next_las = self.get_human_traj_robot_stays(world_state, human_subtask)
                human_next_la = next_las[0]
                next_kb_prob = np.zeros(Action.NUM_ACTIONS)

                for la in Action.ALL_ACTIONS:
                    joint_motion_action = (la, human_next_la) if self.agent_index == 0 else (human_next_la, la)
                    new_positions, new_orientations = self.mdp.compute_new_positions_and_orientations(world_state.players, joint_motion_action)
                    one_la_successor_state = self.jmp.derive_state(world_state, tuple(zip(*[new_positions, new_orientations])), [joint_motion_action])
                    la_step_cost = sum([abs(new_positions[0][0] - world_state.players[0].position[0]), abs(new_positions[0][1] - world_state.players[0].position[1])])

                    # the KB' we want to seek is the one S' needs {NOPE!!!, it is actually that the S, KB' leads to S'}
                    next_kbs_and_prob = self.roll_out_for_kb(one_la_successor_state, search_depth=KB_SEARCH_DEPTH, other_agent_plan=next_las, explore_interact=True)
                    robot_world_state_info = self.world_state_to_mdp_state_key(one_la_successor_state, one_la_successor_state.players[0], one_la_successor_state.players[1], RETURN_NON_SUBTASK=True, RETURN_OBJ=True)
                    
                    # check if human subtask is still the same
                    one_la_human_subtasks = [self.state_dict[mdp_state_key][-1]]
                    human_holding0 = None if world_state.players[1].held_object == None else world_state.players[1].held_object.name
                    human_holding1 = None if one_la_successor_state.players[1].held_object == None else one_la_successor_state.players[1].held_object.name
                    agent_holding0 = None if world_state.players[0].held_object == None else world_state.players[0].held_object.name
                    agent_holding1 = None if one_la_successor_state.players[0].held_object == None else one_la_successor_state.players[0].held_object.name

                    # update the human subtask when the human's holding changes, since this is not shown in the kb, we have a seperate if else statement.
                    

                    # if (human_holding0 != human_holding1 or human_next_la == 'interact'):
                    if (human_next_la == 'interact'):
                        human_changed_world = False
                        i_pos = Action.move_in_direction(one_la_successor_state.players[1].position, one_la_successor_state.players[1].orientation)
                        if world_state.has_object(i_pos) and one_la_successor_state.has_object(i_pos):
                            obj0 = world_state.get_object(i_pos).state
                            obj1 = one_la_successor_state.get_object(i_pos).state
                            if obj0 != obj1:
                                human_changed_world = True
                        elif world_state.has_object(i_pos) or one_la_successor_state.has_object(i_pos):
                            human_changed_world = True

                        if human_changed_world: 
                            # one_la_human_subtasks, _ = self.human_state_subtask_transition(self.state_dict[mdp_state_key][-1], robot_world_state_info[1:])
                            kb_robot_world_state_info = self.get_kb_key(self.sim_human_model.get_knowledge_base(one_la_successor_state, rollout_kb=curr_kb)).split('.')[:-1] + [robot_world_state_info[4]]
                            one_la_human_subtasks, _ = self.human_state_subtask_transition(self.state_dict[mdp_state_key][-1], kb_robot_world_state_info)
                    ## TODO: why not consider the robot changing the world? Only commented out since it seems to work better for initial steps to find the reasonable actions
                    # the idea is update the human subtask when the enivornment changes
                    elif (agent_holding0 != agent_holding1) or (self.state_dict[mdp_state_key][1:4] != robot_world_state_info[1:4]): #or (human_next_la == 'interact'):
                        if self.sim_human_model is not None:
                            one_la_kb_key = self.get_kb_key(self.sim_human_model.get_knowledge_base(one_la_successor_state, rollout_kb=curr_kb))
                        else:
                            one_la_kb_key = self.get_kb_key(one_la_successor_state)

                        # get the next low-level step environment to determine the human's subtask
                        one_la_human_subtasks, _ = np.array(self.kb_based_human_subtask_state(self.state_dict[mdp_state_key][-1], one_la_kb_key, kb_key=True), dtype=object)

                    one_la_human_subtask_count = 0
                    for one_la_human_subtask in one_la_human_subtasks:
                        one_la_state_idx = self.get_mdp_state_idx(self.world_state_to_mdp_state_key(one_la_successor_state, one_la_successor_state.players[0], one_la_successor_state.players[1], one_la_human_subtask))
                        
                        if one_la_state_idx != mdp_state_idx:

                            # since human's holding doesn't change, that means it's subtask goal has not changed, therefore, the comparison should be with the mdp_state_key's human subtask. Keep in mind that the human's subtask goal in mdp_state_key is a goal that is currently being executed and not yet complete.
                            # if (human_holding0 == human_holding1 and human_next_la != 'interact') and self.state_dict[mdp_state_key][-1] != one_la_human_subtask:
                            #     est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += 0

                            # after_action_world_states = self.mdp_state_to_world_state(one_la_state_idx, one_la_state_idx, one_la_successor_state, with_argmin=True)
                            # else:
                            #     after_action_world_states = self.mdp_state_to_world_state(mdp_state_idx, one_la_state_idx, world_state, with_argmin=True)
                            
                            # if (agent_holding0 != agent_holding1):
                            #     one_step_cost = 1 
                            #     print(one_la_successor_state.players_pos_and_or)
                            #     # total_one_step_cost += one_step_cost
                                
                            #     # V(S')
                            #     # if (s_kb_prim_idx, one_la_state_idx) not in computed_v_cost.keys():
                            #     cost = self.compute_V(one_la_successor_state, self.get_key_from_value(self.state_idx_dict, one_la_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                            #     print('one la state key: cost', self.get_key_from_value(self.state_idx_dict, one_la_state_idx), cost)

                            #     # total_v_cost += cost
                                
                            #     est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += (cost * (1/len(one_la_human_subtasks))) #* (1/len(after_action_world_states))
                            #     # action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= (one_step_cost) * (1/len(one_la_human_subtasks))# * (1/one_step_cost) 
                            # else:

                            # else:
                            one_la_human_subtask_count += 1
                            after_action_world_states = self.mdp_state_to_world_state(mdp_state_idx, one_la_state_idx, world_state, with_argmin=True)
                            total_v_cost = 0
                            # total_one_step_cost = 0
                            for after_action_world_state in after_action_world_states[:,0]:
                                if self.jmp.is_valid_joint_motion_pair(one_la_successor_state.players_pos_and_or, after_action_world_state.players_pos_and_or):
                                    # already considers the interaction action
                                    _, one_step_cost = self.joint_action_cost(one_la_successor_state, after_action_world_state.players_pos_and_or, PLAN_COST='robot')#'short'
                                else:
                                    one_step_cost = (self.mdp.height*self.mdp.width)*2 
                                
                                # cost = self.compute_V(one_la_successor_state, self.get_key_from_value(self.state_idx_dict, one_la_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                                cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, one_la_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                                print('one la state key: cost: one_step_cost: add_cost', self.get_key_from_value(self.state_idx_dict, one_la_state_idx), cost, one_step_cost, (cost/(one_step_cost*200)))
                                
                                est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += ((cost+(cost/(one_step_cost*200))) * (1/len(after_action_world_states)) * (1/len(one_la_human_subtasks)))
                            
                            # if one_la_human_subtask_count > 0:
                            #     est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] *= (1/one_la_human_subtask_count)

                        else:
                            kb_state_values = {}
                            print(list(next_kbs_and_prob.items()))
                            for next_kb_key, [next_kb_prob, _] in next_kbs_and_prob.items():
                                kb_state_values[next_kb_key] = 0
                                nxt_human_subtasks, _ = np.array(self.kb_based_human_subtask_state(one_la_human_subtask, next_kb_key, kb_key=True), dtype=object)

                                # P(KB'|KB, L_A): for next human subtasks based on kb, we assume the human does complete the task, therefore, the world changes as well
                                # for (nxt_human_subtask, nxt_world_info) in zip(nxt_human_subtasks, nxt_world_infos):
                                    # kb_state_keys = '_'.join([str(i) for i in (self.subtask_based_next_state('_'.join(human_subtask), robot_world_state_info, next_subtask=nxt_human_subtask[0]))])
                                    # kb_state_keys = '_'.join([next_kb_key.split('.')[-1]] + [str(i) for i in nxt_world_info] + nxt_human_subtask)
                                    # s_kb_prim_idx = self.state_idx_dict[kb_state_keys]

                                    # T(S'|S, KB'): all possible next states based on KB' (human new subtask based on completing old subtask with KB)
                                # next_mdp_state_idx_arr = np.where(self.s_kb_trans_matrix[mdp_state_idx] > 0.000001)[0]
                                next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[mdp_state_idx] > 0.000001)[0]

                                # Assuming that the next state constains states where the success is both half for both agents, we then over-write the human -subtask based on the nxt_human_subtasks. This is to match the subtask with the knowledge base the human has but still have the other state information to represent the updated world based on the success probability.
                                kb_based_next_mdp_state_idx_arr = np.array([], dtype=int)
                                # if next_kb_key != curr_kb:
                                for next_mdp_state_idx in next_mdp_state_idx_arr:
                                    kb_state_obj = self.state_dict[self.get_key_from_value(self.state_idx_dict, next_mdp_state_idx)].copy()
                                    for nxt_human_subtask in nxt_human_subtasks:
                                        kb_state_obj[-1] = nxt_human_subtask                                    
                                        kb_based_state_key = self.get_key_from_value(self.state_dict, kb_state_obj)
                                        kb_based_state_idx = self.state_idx_dict[kb_based_state_key]
                                        kb_based_next_mdp_state_idx_arr = np.concatenate((kb_based_next_mdp_state_idx_arr, [kb_based_state_idx]))

                                next_mdp_state_idx_arr = np.unique(kb_based_next_mdp_state_idx_arr)

                                # # next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[mdp_state_idx] > 0.000001)[0]
                                # ## Next states should qualify for two criteria:
                                # # 1. is reachable from current state (If KB explore has found a KB state, then is considered reachable)
                                # if next_kb_key != curr_kb:
                                #     kb_state_obj = self.kb_based_state(mdp_state_key, next_kb_key, kb_key=True)
                                #     for j in range(len(kb_state_obj)):
                                #         if kb_state_obj[j] == -1:
                                #             kb_state_obj[j] = 'None'
                                # # 2. contains the correct next human subtask (keep this even though there will be a gap between the current state to the next state since logically you do not see the transition of the kb change directly in the computation, but it is considered by including the next_kb_prob in the computation)
                                #     next_mdp_state_idx_arr = np.array([], dtype=int)
                                #     for nxt_human_subtask in nxt_human_subtasks:
                                #         kb_state_obj[-1] = nxt_human_subtask                                    
                                #         kb_based_state_key = self.get_key_from_value(self.state_dict, kb_state_obj)
                                #         kb_based_state_idx = self.state_idx_dict[kb_based_state_key]

                                #         # # we do not use the next states of kb_based_state_idx since then we will be considering taking a high-level step in advance (wrong!)
                                #         # next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, [kb_based_state_idx]))
                                #         next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, np.where(self.s_kb_trans_matrix[kb_based_state_idx] > 0.000001)[0]))
                                #         # next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, np.where(self.optimal_s_kb_trans_matrix[kb_based_state_idx] > 0.000001)[0]))

                                # # # else:
                                # #     kb_state_obj[:-1] = self.state_dict[mdp_state_key][:-1]
                                # #     kb_based_state_idx = self.state_idx_dict[self.get_key_from_value(self.state_dict, kb_state_obj)]
                                # #     next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, np.where(self.s_kb_trans_matrix[kb_based_state_idx] > 0.000001)[0]))
                                
                                # next_mdp_state_idx_arr = np.unique(next_mdp_state_idx_arr)

                                # only compute human_subtasks that are the same as nxt_human_subtask induced by KB'
                                nxt_state_counter = 0
                                all_nxt_state_value = 0
                                all_nxt_one_step_cost = 0
                                for next_state_idx in next_mdp_state_idx_arr:
                                    # if (self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1] in nxt_human_subtasks) and (mdp_state_idx != next_state_idx):
                                    if (mdp_state_idx != next_state_idx):
                                        nxt_state_counter+=1
                                        print('nxt_human_subtask', self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)])
                                        # if one_la_state_idx == mdp_state_idx:
                                        after_action_world_states = self.mdp_state_to_world_state(one_la_state_idx, next_state_idx, one_la_successor_state, with_argmin=True)
                                        # else:
                                        #     after_action_world_states = self.mdp_state_to_world_state(mdp_state_idx, one_la_state_idx, world_state, with_argmin=True)
                                        total_v_cost = 0
                                        total_one_step_cost = 0
                                        for after_action_world_state_info in after_action_world_states:
                                            if self.jmp.is_valid_joint_motion_pair(one_la_successor_state.players_pos_and_or, after_action_world_state_info[0].players_pos_and_or):
                                                [ai_wait, human_wait] = after_action_world_state_info[3]
                                                if ai_wait or human_wait:
                                                    _, one_step_cost = self.joint_action_cost(one_la_successor_state, after_action_world_state_info[0].players_pos_and_or, PLAN_COST= 'human' if ai_wait else 'robot')
                                                else:
                                                    _, one_step_cost = self.joint_action_cost(one_la_successor_state, after_action_world_state_info[0].players_pos_and_or, PLAN_COST='robot')# change to max such that we make sure the state is reached to accuratly represent the probability of obtaining this state value #average
                                                one_step_cost += 1 # consider the current state to the one_la_successor_state action
                                            else:
                                                one_step_cost = (self.mdp.height*self.mdp.width)*2 
                                            
                                            print('(', one_la_successor_state.players_pos_and_or, after_action_world_state_info[0].players_pos_and_or, ') one_step_cost', one_step_cost)
                                            total_one_step_cost += one_step_cost
                                            
                                            # V(S')
                                            # if (s_kb_prim_idx, next_state_idx) not in computed_v_cost.keys():
                                            cost = self.compute_V(after_action_world_state_info[0], self.get_key_from_value(self.state_idx_dict, next_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                                            print('next state key: cost: add_cost', self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, (cost/(one_step_cost*200)))

                                            total_v_cost += cost
                                            kb_state_values[next_kb_key] += (cost + (cost/(one_step_cost*200))) * (1/len(after_action_world_states)) 
                                                                                
                                        kb_state_values[next_kb_key] *= (next_kb_prob * (1/len(one_la_human_subtasks)))
                                        print('(',one_la_state_idx, next_state_idx, ')', 'total_v_cost =', total_v_cost, '; total one step cost:', total_one_step_cost, 'next_kb_prob:', next_kb_prob, 'num_after_action_world:', len(after_action_world_states), 'num_one_la_subtask:', len(one_la_human_subtasks))

                                if nxt_state_counter > 0:
                                    kb_state_values[next_kb_key] *= (1/nxt_state_counter)
                                    print('nxt_state_counter:', nxt_state_counter)

                            est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += max(kb_state_values.values())
                                        
                                            # action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= (one_step_cost * next_kb_prob * (1/len(after_action_world_states)) * (1/len(one_la_human_subtasks)))# * (1/one_step_cost) 

                                        # if (s_kb_prim_idx, next_state_idx) not in computed_v_cost.keys():
                                        # computed_v_cost[(s_kb_prim_idx, next_state_idx)] = total_v_cost/len(after_action_world_states)
                                        # all_nxt_state_value += total_v_cost/len(after_action_world_states)
                                        # all_nxt_one_step_cost += total_one_step_cost/len(after_action_world_states)

                                        # for value_cost, one_step_cost in computed_v_cost[(s_kb_prim_idx, next_state_idx)]:
                                        # if (self.kb_idx_dict[next_kb_key], next_state_idx) in self.sprim_s_kb_trans_matrix[s_kb_prim_idx]:
                                            # est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += (computed_v_cost[(s_kb_prim_idx, next_state_idx)] * self.sprim_s_kb_trans_matrix[s_kb_prim_idx][(self.kb_idx_dict[next_kb_key], next_state_idx)] * next_kb_prob) * (1/len(after_action_world_states))
                                            
                                            # action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= avg_one_step_cost * self.sprim_s_kb_trans_matrix[s_kb_prim_idx][(self.kb_idx_dict[next_kb_key], next_state_idx)] * next_kb_prob
                                            
                                
                                
                                #     action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= (all_nxt_one_step_cost+1) * (1/nxt_state_counter) * next_kb_prob

        q = self.compute_Q(belief, est_next_state_v, action_cost)
        print('q value =', q)
        print('next_state_value:', est_next_state_v)
        print('action_cost:', action_cost)       
        action_idx = self.get_best_action(q)
        print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
        print("It took {} seconds for this step".format(time.time() - start_time))
        
        return Action.INDEX_TO_ACTION[action_idx], None, low_level_action

    def step(self, world_state, belief, SEARCH_DEPTH=5, SEARCH_TIME=1, KB_SEARCH_DEPTH=3, debug=False):
        '''
        The goal is to obtain the possible obtained value of a low-level action and select the one with the highest.
        '''
        start_time = time.time()
        est_next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)

        # update sim human
        curr_human_kb = self.sim_human_model.get_knowledge_base(world_state)
        curr_kb_key = self.get_kb_key(curr_human_kb)

        ## Reason over all belief over human's subtask
        for curr_subtask, belief_prob in belief.items():
            if belief_prob > 0.2 or all(np.array(list((belief.values()))) < 0.072):
                ## obtain human's next action based on the curr_subtask
                next_human_las = self.get_human_traj_robot_stays(world_state, self.subtask_dict[curr_subtask])
                next_human_la = next_human_las[0]

                # perform one step rollout
                world_state_to_la = {}
                for la in Action.ALL_ACTIONS:
                    joint_motion_action = (la, next_human_la) if self.agent_index == 0 else (next_human_la, la)
                    # get one step roll out world state and its corresponding high-level state representation
                    new_pos, new_ori = self.mdp.compute_new_positions_and_orientations(world_state.players, joint_motion_action)
                    rollout_world_state = self.jmp.derive_state(world_state, tuple(zip(*[new_pos, new_ori])), [joint_motion_action])
                    if rollout_world_state not in world_state_to_la.keys():
                        world_state_to_la[rollout_world_state] = [la]
                    else:
                        world_state_to_la[rollout_world_state].append(la)
                
                # Compute value of each one step state
                one_step_state_info_dict = {} # one_step_world_state: [[list of (next_high_state, value of next_high_state, depended next kb, next kb prob)], [list of (one_step_cost from one_step_world_state to next_high_state_world_state)]]
                for one_step_world_state, las in world_state_to_la.items():
                    if len(one_step_world_state.order_list) == 0:
                        return las[0], None, None
                    if one_step_world_state not in one_step_state_info_dict.keys():
                        if self.debug: print('one_step_world_state:', one_step_world_state.players_pos_and_or)
                        ### Map one step world state to a high-level state representation to obtain next high-level state
                        one_step_robot_world_str = self.world_state_to_mdp_state_key(one_step_world_state, one_step_world_state.players[0], one_step_world_state.players[1], RETURN_NON_SUBTASK=True)

                        # one step human subtask (either same subtask or changes due to a change in human's kb)
                        one_step_human_subtasks = [curr_subtask]
                        one_step_human_kb = self.sim_human_model.get_knowledge_base(one_step_world_state, rollout_kb=curr_human_kb)
                        one_step_human_kb_key = self.get_kb_key(one_step_human_kb)
                        if self.debug: print('one step human kb key:', one_step_human_kb_key)
                        
                        human_held_obj = 'None' if one_step_world_state.players[1-self.agent_index].held_object is None else one_step_world_state.players[1-self.agent_index].held_object.name
                        if one_step_human_kb_key != curr_kb_key or next_human_la == 'interact':
                            one_step_human_subtasks, _ = self.det_kb_based_human_subtask_state(curr_subtask, one_step_human_kb_key, kb_key=True, human_obj=human_held_obj)

                        one_step_states_keys = ['_'.join([one_step_robot_world_str, one_step_human_subtask]) for one_step_human_subtask in one_step_human_subtasks]
                        if self.debug: print('one_step_human_subtasks:', one_step_human_subtasks)
                        
                        ### Get next possible kbs (roll out for n number of steps)
                        rollout_kbs_and_probs = self.roll_out_for_kb(one_step_world_state, one_step_human_kb, search_depth=KB_SEARCH_DEPTH, other_agent_plan=next_human_las[1:], explore_interact=False)
                        if self.debug: print('rollout_kbs_and_probs:', rollout_kbs_and_probs)
                        
                        ### For each possible one step state and kb, get next high-level state and its value
                        one_step_state_key_info_dict = {}
                        for one_step_state_key in one_step_states_keys:
                            rollout_kb_state_key_cost = {}
                            for rollout_kb_key, [rollout_kb_prob, one_step_to_rollout_world_cost, rollout_world_state] in rollout_kbs_and_probs.items():
                                if self.debug: print('rollout_world_state:', rollout_world_state.players_pos_and_or)
                                if self.debug: print('rollout_kb_key:', rollout_kb_key)
                                rollout_states_keys = [(one_step_human_kb_key, one_step_state_key, 0)]
                                if rollout_kb_key != one_step_human_kb_key: #and self.jmp.is_valid_joint_motion_pair(one_step_world_state.players_pos_and_or, rollout_world_state.players_pos_and_or):
                                    # check if need to update human's subtask state based on next kb
                                    rollout_states_keys = []
                                    ### TODO: check if this plan cost should be just robot 
                                    
                                    # _, one_step_to_rollout_world_cost = self.joint_action_cost(one_step_world_state, rollout_world_state.players_pos_and_or, PLAN_COST='robot')
                                    human_held_obj = 'None' if rollout_world_state.players[1-self.agent_index].held_object is None else rollout_world_state.players[1-self.agent_index].held_object.name

                                    rollout_human_subtasks, _ = self.det_kb_based_human_subtask_state(self.state_dict[one_step_state_key][-1], rollout_kb_key, kb_key=True, human_obj=human_held_obj)
                                    for rollout_human_subtask in rollout_human_subtasks:
                                        rollout_states_keys.append((rollout_kb_key, self.world_state_to_mdp_state_key(rollout_world_state, rollout_world_state.players[0], rollout_world_state.players[1], subtask=rollout_human_subtask), one_step_to_rollout_world_cost))

                                ## get next state based on transition function and average the subtasks' value
                                rollout_state_key_cost = {}
                                for _, rollout_state_key, rollout_cost in rollout_states_keys:
                                    rollout_state_idx = self.state_idx_dict[rollout_state_key]
                                    # next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]
                                    next_mdp_state_idx_arr = np.where(self.optimal_non_subtask_s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]

                                    # get value of the next states and pick the best performing one
                                    max_cost_next_state_idx = next_mdp_state_idx_arr[0]
                                    next_states_dep_ori_state_kb_prob = {} # next_state_key : [rollout_cost, next_kb_prob, values]
                                    for next_state_idx in next_mdp_state_idx_arr:
                                        next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)] = [rollout_cost, [], 0]

                                        # map next states to world to get starting world state for the rollout
                                        next_state_world_states = self.mdp_state_to_world_state(rollout_state_idx, next_state_idx, rollout_world_state, with_argmin=True, cost_mode='average')
                                        for next_state_world_state in next_state_world_states:
                                            # compute next state reward to rollout n step based on assuming optimal path planning 
                                            ### TODO: check the cost if it has to be positive, reward - steps
                                            cost = self.compute_V(next_state_world_state[0], self.get_key_from_value(self.state_idx_dict, next_state_idx), search_depth=((SEARCH_DEPTH-1) if (las[0] == 'interact' or next_human_la == 'interact') else SEARCH_DEPTH), search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True, debug=debug)
                                            rollout_to_next_world_cost = next_state_world_state[1]

                                            # log the cost and one step cost
                                            next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)][1].append((cost, rollout_to_next_world_cost))
                                            estimate_cost = cost + (cost/((rollout_to_next_world_cost + rollout_cost)*200))
                                            next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1] = estimate_cost

                                            # log the max cost state idx
                                            if estimate_cost > next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)][-1]:
                                                max_cost_next_state_idx = next_state_idx

                                    if self.debug: 
                                        print('all next_mdp_state_idx_arr:', next_states_dep_ori_state_kb_prob)
                                        print('[Best rollout state key --> next world states] ', rollout_state_key, ':', self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx), next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)], end='\n\n')

                                    rollout_state_key_cost[rollout_state_key] = [self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx), next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)][-1]]

                                tmp_v = np.array(list(rollout_state_key_cost.values()))
                                # avg_rollout_state_key_value = np.average(np.array(tmp_v[:,-1], dtype=float))
                                max_rollout_state_key = list(rollout_state_key_cost.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                                rollout_kb_state_key_cost[rollout_kb_key] = (max_rollout_state_key, list(rollout_state_key_cost.keys()), list(rollout_state_key_cost.values()), rollout_state_key_cost[max_rollout_state_key][-1]) # (kb state key, next state key, next state value)
                                
                                if self.debug: 
                                    print('average all rollout kb subtasks:', rollout_state_key_cost)
                                    print('[Max rollout kb with subtask] ', rollout_kb_key, max_rollout_state_key, rollout_state_key_cost[max_rollout_state_key][-1], end='\n\n')

                            tmp_v = np.array(list(rollout_kb_state_key_cost.values()), dtype=object)
                            max_rollout_kb_key = list(rollout_kb_state_key_cost.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                            one_step_state_key_info_dict[one_step_state_key] = (max_rollout_kb_key, rollout_kb_state_key_cost[max_rollout_kb_key][0], rollout_kb_state_key_cost[max_rollout_kb_key][1], rollout_kb_state_key_cost[max_rollout_kb_key][2], rollout_kb_state_key_cost[max_rollout_kb_key][-1]) # (kb key, kb state key, next state key, next state value)
                            
                            if self.debug: 
                                print('all rollout kb:', rollout_kb_state_key_cost)
                                print('[Best rollout kb for one step] ', one_step_state_key, max_rollout_kb_key, rollout_kb_state_key_cost[max_rollout_kb_key][-1], end='\n\n')

                        tmp_v = np.array(list(one_step_state_key_info_dict.values()), dtype=object)
                        # avg_la_state_key = np.average(np.array(tmp_v[:,-1], dtype=float))
                        max_la_state_key = list(one_step_state_key_info_dict.keys())[np.argmax(np.array(tmp_v[:,-1]))]
                        one_step_state_info_dict[one_step_world_state] = (max_la_state_key, list(one_step_state_key_info_dict.keys()), list(one_step_state_key_info_dict.values()), one_step_state_key_info_dict[max_la_state_key][-1])
                        
                        if self.debug: 
                            print('[Max one step state key based on one step subtask] ', max_la_state_key, ':', one_step_state_key_info_dict)
                            print('----------', end='\n\n')

                for one_step_world_state, v in one_step_state_info_dict.items():
                    for la in world_state_to_la[one_step_world_state]:
                        est_next_state_v[self.subtask_idx_dict[curr_subtask]][Action.ACTION_TO_INDEX[la]] = v[-1]

                # pick the la with highest value next state
                tmp_v = np.array(list(one_step_state_info_dict.values()), dtype=object)
                max_one_step_world_state = list(one_step_state_info_dict.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                max_la_actions = world_state_to_la[max_one_step_world_state]
                
                if self.debug: print('[Best la actions based on value] ', max_la_actions, max_one_step_world_state, one_step_state_info_dict)

        q = self.compute_Q(list(belief.values()), est_next_state_v, action_cost)
        action_idx = self.get_best_action(q)
        
        if self.debug: 
            print('q value =', q)
            print('next_state_value:', est_next_state_v)
            print('action_cost:', action_cost)       
            print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
            print("It took {} seconds for this step".format(time.time() - start_time))
        

        return Action.INDEX_TO_ACTION[action_idx], None, None

    def compute_Q(self, b, v, c, knowledge_gap=0, gamma=0.9):
        '''
        P(H_a|L_a) [vector]: high-level subtask conditioned on low-level action
        T(KB'|KB, L_a) [2D matrix]: transition of the knowledge base
        B(S|o, KB') [2D matrix]: the belief distribution of state S conditioned on the current observation and new knowledge base
        T(S'|S, H_a) [3D matrix=(H_a, S, S')]: the transition matrix of current state to next state given high-level subtask
        V(S') [one value]: value of the next state
        c(S', L_a) [one value]: the cost of reaching the next state given the current low-level action
        KB_cost(KB', o) [one value]: the cost of the knowledge difference the human has with the actual world status
        
        Q value is computed with the equation:
        Q(S', L_a) = [ V(S') * [B(S|o) @ (P(H_a|L_a) * T(S'|S, H_a, KB')*P(KB'|KB, L_a))]# + KB_cost(KB', o)]
        '''
        print('b =', b)
        print('v =', v)
        print('c =', c)
        
        return b@((v*gamma)+c)+knowledge_gap
    
    def init_mdp(self):
        self.init_actions()
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_s_kb_trans_matrix()
        self.init_sprim_s_kb_trans_matrix()
        self.init_optimal_s_kb_trans_matrix()
        self.init_optimal_non_subtask_s_kb_trans_matrix()
    
    
# class AbstractQMDPPlanner(HumanSubtaskQMDPPlanner):
#     def __init__(self, mdp, mlp_params, abstract_file, abstract_trans_file, map_world_to_state_file, state_dict={}, state_idx_dict={}, action_dict={}, action_idx_dict={}, transition_matrix=None, reward_matrix=None, policy_matrix=None, value_matrix=None, num_states=0, num_rounds=0, epsilon=0.01, discount=0.8):
#         super().__init__(mdp, mlp_params, state_dict, state_idx_dict, action_dict, action_idx_dict, transition_matrix, reward_matrix, policy_matrix, value_matrix, num_states, num_rounds, epsilon, discount)

#         self.abstract_states = self.read_abstract_state(abstract_file)
#         self.abstract_trans = self.read_abstract_trans(abstract_trans_file)
#         self.subtask_dict = self.decode_abstract_states()
#         self.action_dict = self.abstract_states.copy()
#         self.map_world_to_state = pickle.load(open(os.path.join(os.getcwd(), map_world_to_state_file, 'rb')))
#         #### NEXT AR: link state_idx_dict with abstract state. Only need to deal with world info transfer to abstract state.


#     def read_abstract_state(abstract_file):
#         abstract_states = json.load(abstract_file)
#         return abstract_states

#     def read_abstract_trans(abstract_trans_file):
#         abstract_trans = np.load(abstract_trans_file)
#         return abstract_trans

#     def decode_abstract_states(self):
#         abstract_sub_states = {}
#         for k,v in self.abstract_states.items():
#             states = v.split('|')
#             sub_state_dict = {}
#             for s in states:
#                 tmp = s.split(':')
#                 sub_state_dict[tmp[0]] = tmp[1]
#             abstract_sub_states[k] = sub_state_dict.copy()

#         return abstract_sub_states

#     def map_world_to_data_column(self, state, player, other_player, subtask):
#         player_obj = None; other_player_obj = None
#         if player.held_object is not None:
#             player_obj = player.held_object.name
#         if other_player.held_object is not None:
#             other_player_obj = other_player.held_object.name

#         order_str = None if len(state.order_list) == 0 else state.order_list[0]
#         for order in state.order_list[1:]:
#             order_str = order_str + '_' + str(order)

#         num_item_in_pot = 0
#         if state.objects is not None and len(state.objects) > 0:
#             for obj_pos, obj_state in state.objects.items():
#                 if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
#                     num_item_in_pot = obj_state.state[1]

#         d = {}
#         d['num_item_in_pot'] = num_item_in_pot
#         d['agent_hold_onion'] = 1.00 if player_obj == 'onion' else 0.00
#         d['agent_hold_soup'] = 1.00 if player_obj == 'soup' else 0.00
#         d['agent_hold_dish'] = 1.00 if player_obj == 'dish' else 0.00

#         d['onion_loc0_agent_pos'] = np.min(np.sum(self.mdp.get_onion_dispenser_locations() - player.position), axis=1)

#         d['pot_loc0_agent_pos'] = np.min(np.sum(self.mdp.get_pot_locations() - player.position), axis=1)

#         d['dish_loc0_agent_pos'] = np.min(np.sum(self.mdp.get_dish_dispenser_locations() - player.position), axis=1)

#         d['serving_loc0_agent_pos'] = np.min(np.sum(self.mdp.get_serving_locations() - player.position), axis=1)

#         return pd.DataFrame.from_dict(d)
    
# # Next AR: solve as new prob. 1. update belief 2. compute value 3. compute policy
#     def belief_update(self, world_state, agent_player, soup_finish, human_player, belief_vector, prev_dist_to_feature, greedy=False):
#         '''
#         Map the observation into the abstract state through the classificiation network.
#         The probability of the classification becomes the belief distribution.
#         '''
#         # get input to classification
#         ### NEXT AR: map world to data column for belief update
#         world_df = self.map_world_to_data_column(world_state)

#         # classify
#         subtask_belief = self.map_world_to_state.predict(world_df)

#         # prediction results matching belief
#         return subtask_belief

#     def map_abstract_to_world(self, abstract_idx):


#     def mdp_action_state_to_world_state(self, action_idx, ori_abstract_idx, ori_world_state, with_argmin=False):
#         new_world_state = ori_world_state.deepcopy()
#         # NEXT AR: include data for map_action_to_location for human

#         possible_agent_motion_goals, AI_WAIT = self.map_abstract_to_world(ori_abstract_idx)
#         possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[mdp_state_obj[-1]][0], self.action_dict[mdp_state_obj[-1]][1], p0_obj=mdp_state_obj[-2], player_idx=1) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)

#     def get_successor_states(self, start_world_state, start_state_key, debug=False):

#         next_abs_state_idx_arr = np.where(self.abstract_trans[:, self.abstract_states[start_state_key]] > 0.000001)

#         for next_abs_state_idx in next_abs_state_idx_arr:
#             next_world_state, cost = self.mdp_action_state_to_world_state(next_abs_state_idx, next_abs_state_idx, start_world_state)
#             successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))

#     # def compute_V(subtask_belief):
        

#     def step(self, curr_world, curr_abs_states, belief):
#         """
#         Compute plan cost that starts from the next qmdp state defined as next_state_v().
#         Compute the action cost of excuting a step towards the next qmdp state based on the
#         current low level state information.
#         action = abstract action (defined by the name of the end abs state)
#         """
        
#         for curr_abs_idx, curr_abs in enumerate(curr_abs_states):
#             abs_action_idxs = np.where(self.abstract_trans[curr_abs, :] > 0.000001)


#             for abs_action_idx in abs_action_idxs:
#                 next_abs_state_idx = abs_action_idx
#                 # cost of current abstract state to next
#                 ## map abstract to world for cost
#                 after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(abs_action_idx, curr_abs_idx, curr_world, with_argmin=True)

#                 next_abs_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_abs_state_idx), search_depth=100)

#                 # cost of current world to abstract state
#                 ## map abstract to world,
#                 joint_action, one_step_cost = self.joint_action_cost(curr_world, after_action_world_state.players_pos_and_or)

#                 next_state_v[i, action_idx] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
#                 # print(next_state_v[i, action_idx])

#                 ## compute one step cost with joint motion considered
#                 action_cost[i, action_idx] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

#         # compute Q and best action


#         return action_idx


class HumanMediumLevelPlanner(object):
    def __init__(self, mdp, ml_action_manager, goal_preference, adaptiveness):
        self.mdp = mdp
        self.ml_action_manager = ml_action_manager
        
        self.sub_goals = {'Onion cooker':0, 'Soup server':1}
        self.adaptiveness = adaptiveness
        self.goal_preference = np.array(goal_preference)
        self.prev_goal_dstb = self.goal_preference

    def get_state_trans(self, obj, num_item_in_pot, order_list):
        ml_goals, curr_p = self.human_ml_motion_goal(obj, num_item_in_pot, order_list)
        
        next_states = []
        for i, ml_goal in enumerate(ml_goals):
            WAIT = ml_goal[4]
            min_distance = np.Inf
            if not WAIT:
                start_locations = self.start_location_from_object(obj)
                min_distance = self.ml_action_manager.motion_planner.min_cost_between_features(start_locations, ml_goal[1])
            else:
                min_distance = 1.0
            next_states.append([ml_goal[0], ml_goal[2], ml_goal[3], 1.0/min_distance, curr_p[i]])
        
        next_states = np.array(next_states, dtype=object)

        return next_states

    def human_ml_motion_goal(self, obj, num_item_in_pot, order_list):
        """ 
        Get the human's motion goal based on its held object. The return can be multiple location since there can be multiple same feature tiles.

        Return: next object, list(motion goals)
        """
        ml_logic_goals = self.logic_ml_action(obj, num_item_in_pot, order_list)

        curr_p = ((1.0-self.adaptiveness)*self.prev_goal_dstb + self.adaptiveness*ml_logic_goals)   
        # print(self.adaptiveness, self.prev_goal_dstb, ml_logic_goals, curr_p)
        task = np.random.choice(len(self.sub_goals), p=curr_p)
        self.prev_goal_dstb = curr_p

        ml_goals = []
        ml_goals.append(self.onion_cooker_ml_goal(obj, num_item_in_pot, order_list))
        ml_goals.append(self.soup_server_ml_goal(obj, num_item_in_pot, order_list))
        ml_goals = np.array(ml_goals, dtype=object)

        return ml_goals, curr_p

    def onion_cooker_ml_goal(self, obj, num_item_in_pot, order_list):
        """
        Player action logic as an onion cooker.

        Return: a list of motion goals
        """
        motion_goal = []; next_obj = ''; WAIT = False
        if obj == 'None':
            motion_goal = self.mdp.get_onion_dispenser_locations()
            next_obj = 'onion'
        elif obj == 'onion':
            motion_goal = self.mdp.get_pot_locations()
            next_obj = 'None'
            num_item_in_pot += 1
        else:
            # drop the item in hand
            motion_goal = self.mdp.get_counter_locations()
            next_obj = 'None'
            # next_obj = obj
            # WAIT = True

        if num_item_in_pot > self.mdp.num_items_for_soup:
            num_item_in_pot = self.mdp.num_items_for_soup

        return next_obj, motion_goal, num_item_in_pot, order_list, WAIT

    def soup_server_ml_goal(self, obj, num_item_in_pot, order_list):
        motion_goal = []; WAIT = False; next_obj = ''
        if obj == 'None':
            motion_goal = self.mdp.get_dish_dispenser_locations()
            next_obj = 'dish'
        elif obj == 'dish' and num_item_in_pot == self.mdp.num_items_for_soup:
            motion_goal = self.mdp.get_pot_locations()
            next_obj = 'soup'
            num_item_in_pot = 0
        elif obj == 'dish' and num_item_in_pot != self.mdp.num_items_for_soup:
            motion_goal = None
            next_obj = obj
            WAIT = True
        elif obj == 'soup':
            motion_goal = self.mdp.get_serving_locations()
            order_list = [] if len(order_list) <= 1 else order_list[1:]
            next_obj = 'None'
        else:
            # drop the item in hand
            motion_goal = self.mdp.get_counter_locations()
            next_obj = 'None'
            # next_obj = obj
            # WAIT = True

        if num_item_in_pot > self.mdp.num_items_for_soup:
            num_item_in_pot = self.mdp.num_items_for_soup

        return next_obj, motion_goal, num_item_in_pot, order_list, WAIT
    
    def logic_ml_action(self, player_obj, num_item_in_pot, order_list):
        """
        """
        env_pref = np.zeros(len(self.sub_goals))

        if player_obj == 'None':

            if num_item_in_pot == self.mdp.num_items_for_soup:
                env_pref[1] += 1
            else:
                next_order = None
                if len(order_list) > 1:
                    next_order = order_list[1]

                if next_order == 'onion':
                    env_pref[0] += 1
                elif next_order == 'tomato':
                    # env_pref[self.sub_goals['Tomato cooker']] += 1
                    pass
                elif next_order is None or next_order == 'any':
                    env_pref[0] += 1
                    # env_pref[self.sub_goals['Tomato cooker']] += 1

        else:
            if player_obj == 'onion':
                env_pref[0] += 1

            elif player_obj == 'tomato':
                # env_pref[self.sub_goals['Tomato cooker']] += 1
                pass

            elif player_obj == 'dish':
                env_pref[1] += 1

            elif player_obj == 'soup':
                env_pref[1] += 1
            else:
                raise ValueError()

        if np.sum(env_pref) > 0.0:
            env_pref = env_pref/np.sum(env_pref)
        else:
            env_pref = np.ones((len(env_pref)))/len(env_pref)

        return env_pref

    def start_location_from_object(self, obj):
        """ 
        Calculate the starting location based on the object in the human's hand. The feature tile bellowing to the held object will be used as the start location.

        Return: list(starting locations)
        """
        if obj == 'None':
            # default to have dropped item
            start_locations = self.mdp.get_pot_locations() + self.mdp.get_serving_locations()
        elif obj == 'onion':
            start_locations = self.mdp.get_onion_dispenser_locations()
        elif obj == 'tomato':
            start_locations = self.mdp.get_tomato_dispenser_locations()
        elif obj == 'dish':
            start_locations = self.mdp.get_dish_dispenser_locations()
        elif obj == 'soup':
            start_locations = self.mdp.get_pot_locations()
        else:
            ValueError()

        return start_locations

# TODO: change MdpPlanner to MdpPlanner(object) not relied on mediumlevelplanner
class MdpPlanner(MediumLevelPlanner):

    def __init__(self, mdp, mlp_params, ml_action_manager=None, \
        state_dict = {}, state_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.98):

        super().__init__(mdp, mlp_params, ml_action_manager=ml_action_manager)
        
        self.state_idx_dict = state_idx_dict
        self.state_dict = state_dict

        self.num_joint_action = (Action.NUM_ACTIONS)# * Action.NUM_ACTIONS)
        self.num_states = num_states
        self.num_rounds = num_rounds
        self.planner_name = 'mdp'
        self.agent_index = 0

        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((self.num_joint_action, MAX_NUM_STATES, MAX_NUM_STATES), dtype=float)
        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros((self.num_joint_action, MAX_NUM_STATES), dtype=float)
        self.policy_matrix = policy_matrix if policy_matrix is not None else np.zeros((MAX_NUM_STATES), dtype=int)
        self.value_matrix = value_matrix if value_matrix is not None else np.zeros((MAX_NUM_STATES), dtype=float)
        self.epsilon = epsilon
        self.discount = discount

    @staticmethod
    def from_mdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            mdp_planner = pickle.load(f)
            mdp = mdp_planner[0]
            params = mdp_planner[1]
            mlp_action_manager = mdp_planner[2]
            
            state_idx_dict = mdp_planner[3]
            state_dict = mdp_planner[4]

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            policy_matrix = mdp_planner[5]
            # value_matrix = mdp_planner.value_matrix
            
            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds
            
            return MdpPlanner(mdp, params, mlp_action_manager, state_dict, state_idx_dict, policy_matrix=policy_matrix)
    
    @staticmethod
    def from_pickle_or_compute(mdp, other_agent, other_agent_index, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):
        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'mdp' + '.pkl'

        if force_compute_all:
            mdp_planner = MdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(other_agent, other_agent_index, filename)
            return mdp_planner
        
        try:
            mdp_planner = MdpPlanner.from_mdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp_policy(other_agent, other_agent_index, filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = MdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(other_agent, other_agent_index, filename)
            return mdp_planner

        if info:
            print("Loaded MdpPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner

    def get_state_dict_length(self):
        return len(self.state_dict)

    def gen_state_dict_key(self, state):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)
        player0 = state.players[0]
        player1 = state.players[1]

        p0_obj = None
        p1_obj = None
        if player0.held_object is not None:
            p0_obj = player0.held_object.name
        if player1.held_object is not None:
            p1_obj = player1.held_object.name

        obj_str = str(None)
        if state.objects is not None and len(state.objects) > 0:
            obj_state = list(state.objects.values())[0]
            obj_str = str(obj_state.name)+','+str(obj_state.state)

        state_str = \
        str(player0.position)+','+str(player0.orientation)+','+str(p0_obj)+','+ \
        str(player1.position)+','+str(player1.orientation)+','+str(p1_obj)+','+ \
        obj_str+','+ \
        str(len(state.order_list))
        # print('obj_str =', obj_str)

        return state_str

    def get_joint_action_array():
        joint_actions = []
        for i0, a0 in enumerate(Action.ALL_ACTIONS):
            for i1, a1 in enumerate(Action.ALL_ACTIONS):
                joint_actions.append((a0, a1))

        return joint_actions

    def rev_joint_action_idx(self, joint_action_idx, agent_idx):
        joint_actions = self.get_joint_action_array()
        return joint_actions[joint_action_idx][agent_idx]

    def overload_trans_matrix(self):
        return len(self.state_dict) >= MAX_NUM_STATES-1
    def search_branches(self, start_states_strs, start_states, other_agent, actions=Action.ALL_ACTIONS):
        successors = {}; new_successors = {}
        init_num_states = len(self.state_idx_dict)
        # print('init_num_states =', init_num_states)
        
        for start_str, start_obj in zip(start_states_strs, start_states):
            if not self.mdp.is_terminal(start_obj):
                other_agent_action, _ = other_agent.action(start_obj)
                for a_idx, action in enumerate(actions):
                    successor_state, _, sparse_reward, shaped_reward = self.embedded_mdp_step(start_obj, action, other_agent_action, other_agent.agent_index) # self.mdp.get_state_transition(start_obj, joint_action)

                    parent_state_idx = self.state_idx_dict[start_str]
                    add_state_str = self.gen_state_dict_key(successor_state)
                    total_reward = sum(sparse_reward+shaped_reward)

                    if add_state_str not in self.state_dict.keys() and successor_state.order_list is not None:
                        self.state_idx_dict[add_state_str] = self.get_state_dict_length()
                        self.state_dict[add_state_str] = successor_state
                        add_state_idx = self.state_idx_dict[add_state_str]

                        # if add_state_idx >= transition_matrix.shape[-1]:
                        #     add_trans_array = np.array((self.num_joint_action,transition_matrix.shape[-1]+1), dtype=float)
                        #     add_trans_array[ja_idx][add_state_idx]
                        #     transition_matrix = np.append(transition_matrix, np.atleast_3d(add_trans_array))


                        self.transition_matrix[a_idx][parent_state_idx][add_state_idx] = 1.0
                        self.reward_matrix[a_idx][parent_state_idx] += total_reward

                        successors[add_state_str] = successor_state
                    # else:
                    #     successors[add_state_str] = successor_state
                    # successors = np.append(successors, np.array([[add_state_str, successor_state]]), axis=0)

                    if self.overload_trans_matrix():
                        print('State numbers reaches matrix maximum limit.')
                        return
            else:
                print('Reached end of one branch.')

        #dfs
        # print('successors =', len(successors), '; new_states =', len(self.state_idx_dict) - init_num_states)
        if len(self.state_idx_dict) - init_num_states > 0:
            # print('len(successors) =', len(successors))
            sub_start_dict = {}

            # sub_start_states_str = random.sample(list(successors.keys()), min(25, len(successors)))
            # for key in sub_start_states_str:
            #     sub_start_dict[key] = successors[key]
            #     successors.pop(key)

            if len(sub_start_dict) <= 0:
                sub_start_dict = successors

            self.search_branches(sub_start_dict.keys(), sub_start_dict.values(), other_agent)

        return    

    def init_all_states(self, start_dict, other_agent): # joint_actions=get_joint_action_array()):

        # find successor states from all start states with all actions avaliable to two agents
        # successors = np.empty((0,2), dtype=object)

        
        remaining_start_dict = start_dict.copy()
        print('start of remaining_start_dict =', len(remaining_start_dict))

        while True:
            sub_start_states_str = random.sample(list(remaining_start_dict.keys()), min(100, len(remaining_start_dict)))
            sub_start_dict = {}
            for key in sub_start_states_str:
                sub_start_dict[key] = remaining_start_dict[key]
                remaining_start_dict.pop(key)
            
            self.search_branches(sub_start_dict.keys(), sub_start_dict.values(), other_agent)

            print('remaining_start_dict =', len(remaining_start_dict))

            if len(remaining_start_dict) <= 0 or self.overload_trans_matrix():
                break

        # print('max of transition matrix (should not exceed 1) =', self.transition_matrix[self.transition_matrix>1.0])

        return

    def init_all_start_states(self, start_states):
        for state in start_states:
            self.state_idx_dict[self.gen_state_dict_key(state)] = len(self.state_dict)
            self.state_dict[self.gen_state_dict_key(state)] = state

        return list(self.state_dict.keys()).copy(), list(self.state_dict.values()).copy(), self.state_dict.copy()

    def get_all_start_states(self):
        players_start_states = self.mdp.get_valid_joint_player_positions_and_orientations()
        start_states = [OvercookedState.from_players_pos_and_or(players_start_state, self.mdp.start_order_list) for players_start_state in players_start_states]
        # self.start_dict = {gen_state_dict_key(state):state for state in start_states}
        print('Number of start states =', len(start_states))
        return start_states

    def get_standard_start_states(self):
        start_state = self.mdp.get_standard_start_state()
        state_str = self.gen_state_dict_key(start_state)
        self.state_dict[state_str] = start_state
        self.state_idx_dict[state_str] = 0

        return list(self.state_dict.keys()).copy(), list(self.state_dict.values()).copy(), self.state_dict.copy()

    def embedded_mdp_step(self, state, action, other_agent_action, other_agent_index):
        if other_agent_index == 0:
            joint_action = (other_agent_action, action)
        else:
            joint_action = (action, other_agent_action)
        if not self.mdp.is_terminal(state):
            results, sparse_reward, shaped_reward, _ = self.mdp.get_state_transition(state, joint_action)
            successor_state = results
        else:
            print("Tried to find successor of terminal")
            assert False, "state {} \t action {}".format(state, action)
            successor_state = state
        return successor_state, joint_action, sparse_reward, shaped_reward

    def one_step_lookahead(self, next_state, joint_action, sparse_reward, shaped_reward):
        # print('##### one_step_lookahead() #####')
        """
        NOTE: Sparse reward is given only when soups are delivered, 
        shaped reward is given only for completion of subgoals 
        (not soup deliveries).
        """
        next_state_str = self.gen_state_dict_key(next_state)
        reward = sparse_reward + shaped_reward

        # add new state into state dictionaries
        if next_state_str not in self.state_dict and next_state.order_list is not None:
            self.state_dict[next_state_str] = next_state
            self.value_dict[next_state_str] = 0.0
            self.policy_dict[next_state_str] = Action.ALL_ACTIONS[0] #default

        prob = 1.0
        v = prob * (reward + self.discount_factor * self.value_dict[next_state_str])

        return v

    def bellman_operator(self, V=None):

        if V is None:
            V = self.value_matrix

        Q = np.zeros((self.num_joint_action, self.num_states))
        for a in range(self.num_joint_action):
            Q[a] = self.reward_matrix[a][:self.num_states] + self.discount * self.transition_matrix[a,:self.num_states,:self.num_states].dot(V[:self.num_states])

            # print(list(Q.max(axis=0)))
            # tmp = input()

        return Q.max(axis=0), Q.argmax(axis=0)

    @staticmethod
    def get_span(arr):
        print('in get span arr.max():', arr.max(), ' - arr.min():', arr.min(), ' = ', (arr.max()-arr.min()))
        return arr.max()-arr.min()

    def log_value_iter(self, iter_count):
        self.num_rounds = iter_count
        output_filename = self.mdp.layout_name+'_'+self.planner_name+'_'+str(self.num_rounds)+".pkl"
        output_mdp_path = os.path.join(PLANNERS_DIR, output_filename)
        self.save_policy_to_file(output_mdp_path)

        return

    def value_iteration(self, other_agent, filename):

        # computation of threshold of variation for V for an epsilon-optimal policy
        if self.discount < 1.0:
            thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            thresh = self.epsilon

        iter_count = 0
        while True:
            V_prev = self.value_matrix[:self.num_states].copy()

            self.value_matrix, self.policy_matrix = self.bellman_operator()

            variation = self.get_span(self.value_matrix-V_prev)
            print('Variation =',  variation, ', Threshold =', thresh)

            if variation < thresh:
                self.log_value_iter(iter_count)
                break
            elif iter_count % LOGUNIT == 0:
                self.log_value_iter(iter_count)
            else:
                pass
            
            iter_count += 1
            
        return

    def save_policy_to_file(self, filename):
        with open(filename, 'wb') as output:
            mdp_plan = [self.mdp, self.params, self.ml_action_manager, self.state_idx_dict, self.state_dict, self.policy_matrix]
            pickle.dump(mdp_plan, output, pickle.HIGHEST_PROTOCOL)
        
    def save_to_file(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def compute_mdp_policy(self, other_agent, other_agent_index, filename):
        start = timer() 

        ALL_START = False
        # initiate all state, transition, reward for array operations
        start_states = None; start_states_strs = None
        if ALL_START == True:
            start_states = self.get_all_start_states()
            start_states_strs, start_states, start_dict = self.init_all_start_states(start_states)
        else:
            start_states_strs, start_states, start_dict = self.get_standard_start_states()

        other_agent.agent_index = other_agent_index
        self.agent_index = 1 - other_agent.agent_index

        self.init_all_states(start_dict, other_agent)
        self.num_states = len(self.state_dict)
        print('Total states =', self.num_states)

        self.value_iteration(other_agent, filename)

        print("Policy Probability Distribution = ")
        # print(self.policy_matrix.tolist(), '\n')
        print(self.policy_matrix.shape)

        print("without GPU:", timer()-start)

        # self.save_to_file(output_mdp_path)
        return 


class SoftmaxMdpPlanner(MdpPlanner):

    def __init__(self, mdp, mlp_params, ml_action_manager=None, \
        state_dict = {}, state_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
        num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.98):
        super().__init__(mdp, mlp_params, ml_action_manager, state_dict, state_idx_dict, transition_matrix, reward_matrix, policy_matrix, value_matrix, num_states, num_rounds, epsilon, discount)
        self.planner_name = 'softmax_mdp'


    @staticmethod
    def from_pickle_or_compute(mdp, other_agent, other_agent_index, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):
        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_mdp.pkl'

        if force_compute_all:
            mdp_planner = SoftmaxMdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(other_agent, other_agent_index, filename)
            return mdp_planner
        
        try:
            mdp_planner = SoftmaxMdpPlanner.from_mdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp_policy(other_agent, other_agent_index, filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = SoftmaxMdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(other_agent, other_agent_index, filename)
            return mdp_planner

        if info:
            print("Loaded MdpPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner

    @staticmethod
    def softmax(values, temperature):
        # return math.log(np.sum(np.exp(x), axis=0))
        return np.exp(values * temperature) / np.sum(np.exp(values * temperature))

    def get_boltzmann_action_idx(self, values, temperature):
        """Chooses index based on softmax probabilities obtained from value array"""
        values = np.array(values)
        softmax_probs = self.softmax(values, temperature)
        action_idx = np.random.choice(len(values), p=softmax_probs)
        return action_idx

    def search_branches(self, start_states_strs, start_states, other_agent, actions=Action.ALL_ACTIONS):
        successors = {}; new_successors = {}
        init_num_states = len(self.state_idx_dict)
        print('init_num_states =', init_num_states)
        
        for start_str, start_obj in zip(start_states_strs, start_states):
            if not self.mdp.is_terminal(start_obj):
                other_agent_action, _ = other_agent.action(start_obj)
                for a_idx, action in enumerate(actions):
                    successor_state, _, sparse_reward, shaped_reward = self.embedded_mdp_step(start_obj, action, other_agent_action, other_agent.agent_index) # self.mdp.get_state_transition(start_obj, joint_action)

                    parent_state_idx = self.state_idx_dict[start_str]
                    add_state_str = self.gen_state_dict_key(successor_state)
                    total_reward = sum(sparse_reward+shaped_reward)

                    if add_state_str not in self.state_dict.keys() and successor_state.order_list is not None:
                        self.state_idx_dict[add_state_str] = self.get_state_dict_length()
                        self.state_dict[add_state_str] = successor_state
                        add_state_idx = self.state_idx_dict[add_state_str]

                        # if add_state_idx >= transition_matrix.shape[-1]:
                        #     add_trans_array = np.array((self.num_joint_action,transition_matrix.shape[-1]+1), dtype=float)
                        #     add_trans_array[ja_idx][add_state_idx]
                        #     transition_matrix = np.append(transition_matrix, np.atleast_3d(add_trans_array))
                        self.transition_matrix[a_idx][parent_state_idx][add_state_idx] = 1.0
                        self.reward_matrix[a_idx][parent_state_idx] += total_reward

                        successors[add_state_str] = successor_state
                    # else:
                    #     successors[add_state_str] = successor_state
                    # successors = np.append(successors, np.array([[add_state_str, successor_state]]), axis=0)

                    if self.overload_trans_matrix():
                        print('State numbers reaches matrix maximum limit.')
                        return

        #dfs
        print('successors =', len(successors), '; new_states =', len(self.state_idx_dict) - init_num_states)
        if len(self.state_idx_dict) - init_num_states > 0:
            print('len(successors) =', len(successors))
            sub_start_dict = {}

            # sub_start_states_str = random.sample(list(successors.keys()), min(100, len(successors)))
            # for key in sub_start_states_str:
            #     sub_start_dict[key] = successors[key]
            #     successors.pop(key)

            if len(sub_start_dict) <= 0:
                sub_start_dict = successors

            self.search_branches(sub_start_dict.keys(), sub_start_dict.values(), other_agent)

        return 
        
    def bellman_operator(self, V=None):

        if V is None:
            V = self.value_matrix

        Q = np.zeros((self.num_joint_action, self.num_states))
        for a in range(self.num_joint_action):
            Q[a] = self.reward_matrix[a][:self.num_states] + self.discount * self.transition_matrix[a,:self.num_states,:self.num_states].dot(V[:self.num_states])

            # print(list(Q.max(axis=0)))
            # tmp = input()

        # softmax action selection for policy

        policy = np.array([self.get_boltzmann_action_idx(q,10) for q in Q.T])

        return Q.max(axis=0), policy # Q.argmax(axis=0)


    def value_iteration(self, other_agent, filename):

        # computation of threshold of variation for V for an epsilon-optimal policy
        if self.discount < 1.0:
            thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            thresh = self.epsilon

        iter_count = 0
        while True:
            V_prev = self.value_matrix[:self.num_states].copy()

            self.value_matrix, self.policy_matrix = self.bellman_operator()

            variation = self.get_span(self.value_matrix-V_prev)
            print('Variation =',  variation, ', Threshold =', thresh)

            if variation < thresh:
                self.log_value_iter(iter_count)
                break
            elif iter_count % LOGUNIT == 0:
                self.log_value_iter(iter_count)
            else:
                pass
            
            iter_count += 1
            
        return
