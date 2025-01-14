import itertools, math, copy
import numpy as np
import random, torch
from collections import defaultdict
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import Heuristic
from overcooked_ai_py.planning.search import SearchTree
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState


class Agent(object):

    def action(self, state):
        """
        Should return an action, and an action info dictionary.
        If collecting trajectories of the agent with OvercookedEnv, the action
        info data will be included in the trajectory data under `ep_infos`.

        This allows agents to optionally store useful information about them
        in the trajectory for further analysis.
        """
        return NotImplementedError()

    def actions(self, states, agent_indices):
        """
        A multi-state version of the action method. This enables for parallized
        implementations that can potentially give speedups in action prediction. 

        Args:
            states (list): list of OvercookedStates for which we want actions for
            agent_indices (list): list to inform which agent we are requesting the action for in each state

        Returns:
            [(action, action_info), (action, action_info), ...]: the actions and action infos for each state-agent_index pair
        """
        return NotImplementedError()

    @staticmethod
    def a_probs_from_action(action):
        action_idx = Action.ACTION_TO_INDEX[action]
        return np.eye(Action.NUM_ACTIONS)[action_idx]

    @staticmethod
    def check_action_probs(action_probs, tolerance=1e-4):
        """Check that action probabilities sum to ≈ 1.0"""
        probs_sum = sum(action_probs)
        assert math.isclose(probs_sum, 1.0, rel_tol=tolerance), "Action probabilities {} should sum up to approximately 1 but sum up to {}".format(list(action_probs), probs_sum)

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        pass


class AgentGroup(object):
    """
    AgentGroup is a group of N agents used to sample 
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        self.agents = agents
        self.n = len(self.agents)
        self.reset()
        for i, agent in enumerate(self.agents):
            agent.set_agent_index(i)

        if not all(a0 is not a1 for a0, a1 in itertools.combinations(agents, 2)):
            assert allow_duplicate_agents, "All agents should be separate instances, unless allow_duplicate_agents is set to true"

    def joint_action(self, state):
        actions_and_probs_n = tuple(a.action(state) for a in self.agents)
        return actions_and_probs_n

    def set_mdp(self, mdp):
        for a in self.agents:
            a.set_mdp(mdp)

    def reset(self):
        for a in self.agents:
            a.reset()


class AgentPair(AgentGroup):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.

    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, *agents, allow_duplicate_agents=False): 
        super().__init__(*agents, allow_duplicate_agents=allow_duplicate_agents)
        assert self.n == 2
        self.a0, self.a1 = self.agents

        if type(self.a0) is CoupledPlanningAgent and type(self.a1) is CoupledPlanningAgent:
            print("If the two planning agents have same params, consider using CoupledPlanningPair instead to reduce computation time by a factor of 2")

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play,
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_and_infos_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_and_infos_1 = self.a1.action(state)
            joint_action_and_infos = (action_and_infos_0, action_and_infos_1)
            return joint_action_and_infos
        else:
            return super().joint_action(state)


class CoupledPlanningPair(AgentPair):
    """
    Pair of identical coupled planning agents. Enables to search for optimal
    action once rather than repeating computation to find action of second agent
    """

    def __init__(self, agent):
        super().__init__(agent, agent, allow_duplicate_agents=True)

    def joint_action(self, state):
        # Reduce computation by half if both agents are coupled planning agents
        joint_action_plan = self.a0.mlp.get_low_level_action_plan(state, self.a0.heuristic, delivery_horizon=self.a0.delivery_horizon, goal_info=True)

        if len(joint_action_plan) == 0:
            return ((Action.STAY, {}), (Action.STAY, {}))

        joint_action_and_infos = [(a, {}) for a in joint_action_plan[0]]
        return joint_action_and_infos


class NNPolicy(object):
    """
    This is a common format for NN-based policies. Once one has wrangled the intended trained neural net
    to this format, one can then easily create an Agent with the AgentFromPolicy class.
    """

    def __init__(self):
        pass

    def multi_state_policy(states, agent_indices):
        """
        A function that takes in multiple OvercookedState instances and their respective agent indices and returns action probabilities.
        """
        raise NotImplementedError()

    def multi_obs_policy(states):
        """
        A function that takes in multiple preprocessed OvercookedState instatences and returns action probabilities.
        """
        raise NotImplementedError()


class AgentFromPolicy(Agent):
    """
    This is a useful Agent class backbone from which to subclass from NN-based agents.
    """
    
    def __init__(self, policy):
        """
        Takes as input an NN Policy instance
        """
        self.policy = policy
        self.reset()
        
    def action(self, state):
        return self.actions([state], [self.agent_index])[0]

    def actions(self, states, agent_indices):
        action_probs_n = self.policy.multi_state_policy(states, agent_indices)
        actions_and_infos_n = []
        for action_probs in action_probs_n:
            action = Action.sample(action_probs)
            actions_and_infos_n.append((action, {"action_probs": action_probs}))
        return actions_and_infos_n

    def actions_from_observations(self, obs):
        """
        An action method that takes in states in post-processed form, and returns respective actions.
        """
        return self.policy.multi_obs_policy(obs)


class RandomAgent(Agent):
    """
    An agent that randomly picks motion actions.
    NOTE: Does not perform interact actions, unless specified
    """

    def __init__(self, sim_threads=None, all_actions=False, custom_wait_prob=None):
        self.sim_threads = sim_threads
        self.all_actions = all_actions
        self.custom_wait_prob = custom_wait_prob
    
    def get_model_name(self):
        return 'random'

    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        legal_actions = list(Action.MOTION_ACTIONS)
        if self.all_actions:
            legal_actions = Action.ALL_ACTIONS
        legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
        action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)

        if self.custom_wait_prob is not None:
            stay = Action.STAY
            if np.random.random() < self.custom_wait_prob:
                return stay, {"action_probs": Agent.a_probs_from_action(stay)}
            else:
                action_probs = Action.remove_indices_and_renormalize(action_probs, [Action.ACTION_TO_INDEX[stay]])

        return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]


class StayAgent(Agent):

    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads
    
    def action(self, state):
        a = Action.STAY
        return a, {}

    def direct_action(self, obs):
        return [Action.ACTION_TO_INDEX[Action.STAY]] * self.sim_threads


class FixedPlanAgent(Agent):
    """
    An Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = plan
        self.i = 0
    
    def action(self, state):
        if self.i >= len(self.plan):
            state.players[self.agent_index].active_log += [0]
            return Action.STAY, {}

        curr_action = self.plan[self.i]
        self.i += 1

        if curr_action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]

        return curr_action, {}

    def reset(self):
        super().reset()
        self.i = 0


class CoupledPlanningAgent(Agent):
    """
    An agent that uses a joint planner (mlp, a MediumLevelPlanner) to find near-optimal
    plans. At each timestep the agent re-plans under the assumption that the other agent
    is also a CoupledPlanningAgent, and then takes the first action in the plan.
    """

    def __init__(self, mlp, delivery_horizon=2, heuristic=None):
        self.mlp = mlp
        self.mlp.failures = 0
        self.heuristic = heuristic if heuristic is not None else Heuristic(mlp.mp).simple_heuristic
        self.delivery_horizon = delivery_horizon

    def action(self, state):
        try:
            joint_action_plan = self.mlp.get_low_level_action_plan(state, self.heuristic, delivery_horizon=self.delivery_horizon, goal_info=True)
        except TimeoutError:
            print("COUPLED PLANNING FAILURE")
            self.mlp.failures += 1
            return Direction.ALL_DIRECTIONS[np.random.randint(4)]
        return (joint_action_plan[0][self.agent_index], {}) if len(joint_action_plan) > 0 else (Action.STAY, {})


class EmbeddedPlanningAgent(Agent):
    """
    An agent that uses A* search to find an optimal action based on a model of the other agent,
    `other_agent`. This class approximates the other agent as being deterministic even though it
    might be stochastic in order to perform the search.
    """

    def __init__(self, other_agent, mlp, env, delivery_horizon=2, logging_level=0):
        """mlp is a MediumLevelPlanner"""
        self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mlp = mlp
        self.env = env
        self.h_fn = Heuristic(mlp.mp).simple_heuristic
        self.logging_level = logging_level

    def action(self, state):
        start_state = state.deepcopy()
        order_list = start_state.order_list if start_state.order_list is not None else ["any", "any"]
        start_state.order_list = order_list[:self.delivery_horizon]
        other_agent_index = 1 - self.agent_index
        initial_env_state = self.env.state
        self.other_agent.env = self.env

        expand_fn = lambda state: self.mlp.get_successor_states_fixed_other(state, self.other_agent, other_agent_index)
        goal_fn = lambda state: len(state.order_list) == 0
        heuristic_fn = lambda state: self.h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, max_iter_count=50000)

        try:
            ml_s_a_plan, cost = search_problem.A_star_graph_search(info=True)
        except TimeoutError:
            print("A* failed, taking random action")
            idx = np.random.randint(5)
            return Action.ALL_ACTIONS[idx]

        # Check estimated cost of the plan equals
        # the sum of the costs of each medium-level action
        assert sum([len(item[0]) for item in ml_s_a_plan[1:]]) == cost

        # In this case medium level actions are tuples of low level actions
        # We just care about the first low level action of the first med level action
        first_s_a = ml_s_a_plan[1]

        # Print what the agent is expecting to happen
        if self.logging_level >= 2:
            self.env.state = start_state
            for joint_a in first_s_a[0]:
                print(self.env)
                print(joint_a)
                self.env.step(joint_a)
            print(self.env)
            print("======The End======")

        self.env.state = initial_env_state

        first_joint_action = first_s_a[0][0]
        if self.logging_level >= 1:
            print("expected joint action", first_joint_action)
        action = first_joint_action[self.agent_index]
        return action, {}


class GreedyHumanModel(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do

    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example forced_coordination.layout,
    in which an individual agent cannot complete the task on their own.
    """

    def __init__(self, mlp, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True):
        self.mlp = mlp
        self.mdp = self.mlp.mdp

        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        self.hl_boltzmann_rational = hl_boltzmann_rational  # For choices among high level goals of same type
        self.ll_boltzmann_rational = ll_boltzmann_rational  # For choices about low level motion

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if it's in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck

        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_chosen_action = None

    def get_model_name(self):
        return 'greedy'

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state, eps = 0, chosen_subtask=None, return_path=False):
        possible_motion_goals = self.ml_action(state, chosen_subtask=chosen_subtask)

        #from IPython import embed
        #embed()

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        if return_path:
            chosen_goal, chosen_action, action_probs, path = self.choose_motion_goal(start_pos_and_or, possible_motion_goals, return_path=return_path)
        else:
            chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, possible_motion_goals)

        if self.ll_boltzmann_rational and chosen_goal[0] == start_pos_and_or[0]:
            chosen_action, action_probs = self.boltzmann_rational_ll_action(start_pos_and_or, chosen_goal)

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            human_changed_world = False
            if self.prev_state is not None:
                i_pos = Action.move_in_direction(state.players[self.agent_index].position, state.players[self.agent_index].orientation)
                if self.prev_state.has_object(i_pos) and state.has_object(i_pos):
                    obj0 = self.prev_state.get_object(i_pos).state
                    obj1 = state.get_object(i_pos).state
                    if obj0 != obj1:
                        human_changed_world = True
                elif self.prev_state.has_object(i_pos) or state.has_object(i_pos):
                    human_changed_world = True

            if self.prev_state is not None and (state.players_pos_and_or[self.agent_index] == self.prev_state.players_pos_and_or[self.agent_index] and (state.players[self.agent_index].held_object == self.prev_state.players[self.agent_index].held_object) and (self.prev_chosen_action !='interact' or (self.prev_chosen_action == 'interact' and not human_changed_world))):
                # if self.prev_state is not None and self.prev_state == state:#and state.players_pos_and_or == self.prev_state.players_pos_and_or:# and self.prev_chosen_action != 'interact':
                    if self.agent_index == 0:
                        joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                    elif self.agent_index == 1:
                        joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                    else:
                        raise ValueError("Player index not recognized")

                    unblocking_joint_actions = []
                    for j_a in joint_actions:
                        new_state, _, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
                        if new_state.player_positions != self.prev_state.player_positions:
                            unblocking_joint_actions.append(j_a)

                    if len(unblocking_joint_actions) > 0:
                        chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]
                    else:
                        chosen_action = Action.STAY
                    action_probs = self.a_probs_from_action(chosen_action)

                    state.players[self.agent_index].stuck_log += [1]
                    self.prev_state = None
            else:
                state.players[self.agent_index].stuck_log += [0]
                self.prev_state = state

            # NOTE: Assumes that calls to the action method are sequential
            # self.prev_state = state
            self.prev_chosen_action = chosen_action

        #eps-greedy
        if random.random() < eps:
          chosen_action = (Action.ALL_ACTIONS[np.random.randint(6)],{})[0]

        if chosen_action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]

        print('greedy human chosen action:', chosen_action, action_probs)

        if return_path:
            return chosen_action, {"action_probs": action_probs}, path
        
        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals, return_path=False):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        return_plans = []
        chosen_goal = None
        chosen_goal_action = None
        action_probs = None
        
        if self.hl_boltzmann_rational:
            possible_plans = [self.mlp.mp.get_plan(start_pos_and_or, goal) for goal in motion_goals]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
            return_plans = possible_plans[goal_idx][0]
        else:
            if return_path:
                chosen_goal, chosen_goal_action, return_plans = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals, return_path=return_path)
            else:
                chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
            if chosen_goal_action is not None:
                action_probs = self.a_probs_from_action(chosen_goal_action)
        
        if return_path:
            return chosen_goal, chosen_goal_action, action_probs, return_plans
        
        return chosen_goal, chosen_goal_action, action_probs
        

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals, return_path=False):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        action_plan = []
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        if return_path:
            return best_goal, best_action, action_plan
        return best_goal, best_action

    def boltzmann_rational_ll_action(self, start_pos_and_or, goal, inverted_costs=False):
        """
        Computes the plan cost to reach the goal after taking each possible low level action.
        Selects a low level action boltzmann rationally based on the one-step-ahead plan costs.

        If `inverted_costs` is True, it will make a boltzmann "irrational" choice, exponentially
        favouring high cost plans rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlp.mp.get_plan(new_pos_and_or, goal)
            sign = (-1) ** int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx], action_probs

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order

                if next_order == 'onion':
                    motion_goals = am.pickup_onion_actions(counter_objects)
                elif next_order == 'tomato':
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals


class GreedySteakHumanModel(GreedyHumanModel):
    def __init__(self, mlp, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True):
        super().__init__(mlp, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck)

    def ml_action(self, state, chosen_subtask=None):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        sink_status = self.mlp.mdp.get_sink_status(state)
        chopping_board_status = self.mlp.mdp.get_chopping_board_status(state)
        pot_states_dict = self.mlp.mdp.get_pot_states(state)
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order
        motion_goals = []

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']

            steak_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
            other_has_hot_plate = other_player.has_object() and other_player.get_object().name == 'hot_plate'
            other_has_steak = other_player.has_object() and other_player.get_object().name == 'steak'
            other_has_meat = other_player.has_object() and other_player.get_object().name == 'meat'
            other_has_onion = other_player.has_object() and other_player.get_object().name == 'onion'
            other_has_plate = other_player.has_object() and other_player.get_object().name == 'plate'

            garnish_ready = len(chopping_board_status['ready']) > 0
            chopping = len(chopping_board_status['full']) > 0
            board_empty = len(chopping_board_status['empty']) > 0
            hot_plate_ready = len(sink_status['ready']) > 0
            rinsing = len(sink_status['full']) > 0
            sink_empty = len(sink_status['empty']) > 0

            if not steak_nearly_ready and state.num_orders_remaining > 0 and not other_has_meat:
                motion_goals += am.pickup_meat_actions(counter_objects)
            if not chopping and not garnish_ready and not other_has_onion:
                motion_goals += am.pickup_onion_actions(counter_objects)
            if chopping and not garnish_ready:
                motion_goals += am.chop_onion_on_board_actions(state)
            if not rinsing and not hot_plate_ready and not other_has_plate:
                motion_goals += am.pickup_plate_actions(counter_objects, state)
            if rinsing and not hot_plate_ready:
                motion_goals += am.heat_plate_in_sink_actions(state)
            if garnish_ready and hot_plate_ready:
                motion_goals += am.pickup_hot_plate_from_sink_actions(counter_objects,state)
            
            if len(motion_goals) == 0:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order
                if next_order == 'onion':
                    motion_goals += am.pickup_onion_actions(counter_objects)
                elif next_order == 'steak':
                    motion_goals += am.pickup_meat_actions(counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals += am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects) + am.pickup_meat_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                motion_goals = am.put_onion_on_board_actions(state)
            
            elif player_obj.name == 'meat':
                motion_goals = am.put_meat_in_pot_actions(pot_states_dict)

            elif player_obj.name == "plate":
                motion_goals = am.put_plate_in_sink_actions(counter_objects, state)

            elif player_obj.name == 'hot_plate':
                motion_goals = am.pickup_steak_with_hot_plate_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'steak':
                motion_goals = am.add_garnish_to_steak_actions(state)

            elif player_obj.name == 'dish':
                motion_goals = am.deliver_dish_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        print(motion_goals)
        return motion_goals
    

class limitVisionHumanModel(GreedyHumanModel):
    def __init__(self, mlp, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True, explore=False, vision_limit=True, vision_bound=120, kb_update_delay=0, debug=False):
        super().__init__(mlp, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp,
                 auto_unstuck)
        self.explore = explore
        self.vision_limit = vision_limit
        self.vision_bound = vision_bound
        self.kb_update_delay_track = [{}, {}, {}]
        self.kb_update_delay = kb_update_delay
        self.init_knowledge_base(start_state)
        self.debug = debug

    def init_knowledge_base(self, start_state):
        self.knowledge_base = {}
        for obj in start_state.objects.values():
            key = self.knowledge_base_key(obj)
            self.knowledge_base[key] = obj
        self.knowledge_base['pot_states'] = self.mlp.mdp.get_pot_states(start_state)
        self.knowledge_base['other_player'] = start_state.players[1]
        
        self.kb_update_delay_track[0]['human_holding'] = [None, 0]
        self.kb_update_delay_track[1]['other_player'] = start_state.players[1].deepcopy()
        self.kb_update_delay_track[2] = {}

    def knowledge_base_key(self, object):
        key = '_'.join((str(object.position[0]), str(object.position[1]), str(object.name)))
        return key

    # def get_vision_bound(self, state, half_bound=60):
    #     player = state.players[self.agent_index]

    #     # get the two points first by assuming facing north
    #     vision_width = np.radians(90-half_bound)
    #     player_back = [player.position[0], player.position[1]]

    #     ori = Direction.DIRECTION_TO_INDEX[player.orientation]
    #     if ori == 0: # north
    #         # right_pt[0] = math.ceil(right_pt[0])
    #         # left_pt[0] = math.ceil(left_pt[0])
    #         player_back[1] += 1
    #     elif ori == 2: # east
    #         # right_pt[1] = math.floor(right_pt[1])
    #         # left_pt[1] = math.floor(left_pt[1])
    #         player_back[0] -= 1
    #     elif ori == 1: # south
    #         # right_pt[0] = math.floor(right_pt[0])
    #         # left_pt[0] = math.floor(left_pt[0])
    #         player_back[1] -= 1
    #     elif ori == 3: # west
    #         # right_pt[1] = math.ceil(right_pt[1])
    #         # left_pt[1] = math.ceil(left_pt[1])
    #         player_back[0] += 1
    
    #     right_pt = player_back + np.array([-math.cos(vision_width), math.sin(vision_width)])
    #     left_pt = player_back + np.array([math.cos(vision_width), math.sin(vision_width)])

    #     # angle based on the agent's facing
    #     # theta = np.radians(0)
        
    #     # c, s = np.cos(theta), np.sin(theta)
    #     # R = np.array(((c, -s), (s, c)))
    #     # right_pt = np.matmul(R,right_pt-player.position)+player.position
    #     # left_pt = np.matmul(R,left_pt-player.position)+player.position

    #     return right_pt, left_pt

    # def in_bound(self, loc, right_pt, left_pt, state):
    #     '''
    #     Use cross product to see if the point is on the left or right side of the vision bound edges.
    #     '''
    #     if not self.vision_limit:
    #         return True

    #     player = state.players[self.agent_index]
    #     right_in_bound = False
    #     left_in_bound = False
    #     thresh = 1e-9
    #     player_back = [player.position[0], player.position[1]]
    #     # angle based on the agent's facing
    #     theta = None
    #     ori = Direction.DIRECTION_TO_INDEX[player.orientation]
    #     if ori == 1: # south
    #         theta = np.radians(0)
    #         bount_theta = np.radians(0)
    #         player_back[1] -= 1
    #     elif ori == 2: # east
    #         theta = np.radians(-90)
    #         bount_theta = np.radians(180)
    #         player_back[0] -= 1
    #     elif ori == 0: # north
    #         theta = np.radians(0)
    #         bount_theta = np.radians(180)
    #         player_back[1] += 1
    #     elif ori == 3: # west
    #         theta = np.radians(-270)
    #         bount_theta = np.radians(180)
    #         player_back[0] += 1

    #     c, s = np.cos(theta), np.sin(theta)
    #     R = np.array(((c, -s), (s, c)))
    #     rot_loc = np.matmul(R,np.array(loc)-player_back)+player_back

    #     rot_left_pt = np.matmul(np.array(((np.cos(bount_theta), -np.sin(bount_theta)), (np.sin(bount_theta), np.cos(bount_theta)))), np.array(left_pt)-player_back)+player_back
    #     rot_right_pt = np.matmul(np.array(((np.cos(bount_theta), -np.sin(bount_theta)), (np.sin(bount_theta), np.cos(bount_theta)))), np.array(right_pt)-player_back)+player_back

    #     # check right bound
    #     right_val = ((rot_right_pt[0] - player_back[0])*(rot_loc[1] - player_back[1]) - (rot_right_pt[1] - player_back[1])*(rot_loc[0] - player_back[0]))
    #     if right_val >= thresh: # above of line
    #         right_in_bound = False
    #     elif right_val <= -thresh: # below of line
    #         right_in_bound = True
    #     else: # on the line
    #         right_in_bound = True

    #     # check left bound
    #     left_val = ((rot_left_pt[0] - player_back[0])*(rot_loc[1] - player_back[1]) - (rot_left_pt[1] - player_back[1])*(rot_loc[0] - player_back[0]))
    #     if left_val >= thresh: # above of line
    #         left_in_bound = True
    #     elif left_val <= -thresh: # below of line
    #         left_in_bound = False
    #     else: # on the line
    #         left_in_bound = True

    #     if (rot_loc == player_back).all():
    #         return False
        
    #     return (left_in_bound and right_in_bound)

    def in_bound(self, state, loc, vision_bound=120/2, move_back=False):
        if vision_bound == 0:
            return True
        
        player = state.players[self.agent_index]
        center_pt = [player.position[0], player.position[1]]

        ori = Direction.DIRECTION_TO_INDEX[player.orientation]
        if ori == 0: # north
            if move_back: center_pt[1] += 1
            if loc[1] == player.position[1] and (loc[0] == player.position[0]-1 or loc[0] == player.position[0]+1): return True
            rot_angel = np.radians(180)
        elif ori == 2: # east
            if move_back: center_pt[0] -= 1
            if loc[0] == player.position[0] and (loc[1] == player.position[1]-1 or loc[1] == player.position[1]+1): return True
            rot_angel = np.radians(270)
        elif ori == 1: # south
            if move_back: center_pt[1] -= 1
            if loc[1] == player.position[1] and (loc[0] == player.position[0]-1 or loc[0] == player.position[0]+1): return True
            rot_angel = np.radians(0)
        elif ori == 3: # west
            if move_back: center_pt[0] += 1
            if loc[0] == player.position[0] and (loc[1] == player.position[1]-1 or loc[1] == player.position[1]+1): return True
            rot_angel = np.radians(90)

        c, s = np.cos(rot_angel), np.sin(rot_angel)
        R = np.array(((c, -s), (s, c)))

        shifted_loc = np.array(loc)-np.array(center_pt)
        y_flip_loc = [shifted_loc[0], shifted_loc[1]*-1]

        rot_loc = np.matmul(R,y_flip_loc)
        
        y = -np.abs(rot_loc[0]*np.cos(np.radians(vision_bound)))
        if y >= rot_loc[1]:
            return True
        
        return False

    def update(self, state):
        # right_pt, left_pt = self.get_vision_bound(state, half_bound=self.vision_bound)
        valid_pot_pos = []

        for obj in state.objects.values():
            if self.in_bound(state, obj.position, vision_bound=self.vision_bound/2):
                key = self.knowledge_base_key(obj)
                self.knowledge_base[key] = obj

                # update the pot states based on the knowledge base
                if obj.name == 'soup':
                    valid_pot_pos.append(obj.position)
                    self.knowledge_base['pot_states'] = self.mlp.mdp.get_pot_states(state, pots_states_dict=self.pot_states, valid_pos=valid_pot_pos)

        # check if other player is in vision
        other_player = state.players[1 - self.agent_index]
        if self.in_bound(state, other_player.position, vision_bound=self.vision_bound/2):
            # print('Other agent in bound')
            self.knowledge_base['other_player'] = other_player

    def get_knowledge_base(self, state):
        # right_pt, left_pt = self.get_vision_bound(state, half_bound=self.vision_bound/2)
        valid_pot_pos = []
        new_knowledge_base = self.knowledge_base.copy()
        for obj in state.objects.values():
            if self.in_bound(state, obj.position, vision_bound=self.vision_bound/2):
                key = self.knowledge_base_key(obj)
                new_knowledge_base[key] = obj

                # update the pot states based on the knowledge base
                if obj.name == 'soup':
                    valid_pot_pos.append(obj.position)
                    new_knowledge_base['pot_states'] = self.mlp.mdp.get_pot_states(state, pots_states_dict=self.pot_states, valid_pos=valid_pot_pos)

        # check if other player is in vision
        other_player = state.players[1 - self.agent_index]
        if self.in_bound(state, other_player.position, vision_bound=self.vision_bound/2):
            # print('Other agent in bound')
            new_knowledge_base['other_player'] = other_player

        return new_knowledge_base

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """

        self.update(state)
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.knowledge_base['pot_states']
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order

                if next_order == 'onion':
                    motion_goals = am.pickup_onion_actions(counter_objects)
                elif next_order == 'tomato':
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order == 'steak':
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
            
            elif player_obj.name == 'meat':
                motion_goals = am.put_meat_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                motion_goals = am.deliver_dish_actions()

            elif player_obj.name == 'hot_plate':
                motion_goals = am.pickup_steak_with_hot_plate_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            if self.explore: # explore to expand the vision.
                # get four directions to explore
                for o in Direction.ALL_DIRECTIONS:
                    motion_goals.append((player.position, o))
                motion_goals.remove(player.pos_and_or)
                random.shuffle(motion_goals)
                motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)][0] # directly return on specific motion goal as the interact plan will always cost
                assert len(motion_goals) != 0
            else: # get to the closest key object location
                motion_goals = am.go_to_closest_feature_actions(player)
                motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
                assert len(motion_goals) != 0

        return motion_goals


class SteakLimitVisionHumanModel(limitVisionHumanModel):
    def __init__(self, mlp, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, debug=False):
        super().__init__(mlp, start_state, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck, explore, vision_limit=vision_limit, vision_bound=vision_bound, kb_update_delay=kb_update_delay, debug=debug)
        self.robot_aware = robot_aware
        self.prev_chosen_subtask = None
        self.kb_log = []

    def deepcopy(self, world_state):
        new_human_model = SteakLimitVisionHumanModel(self.mlp, world_state, auto_unstuck=self.auto_unstuck, explore=self.explore, vision_limit=self.vision_limit)
        
        for k, v in self.knowledge_base.items():
            new_human_model.knowledge_base[k] = v.deepcopy()

        return new_human_model
    
    def kb_track_deepcopy(self, kb_track):
        new_kb_track = {}
        for k, v in kb_track.items():
            if type(v) == list:
                if v[0] is not None and v[0].position is not None:
                    new_kb_track[k] = [v[0].deepcopy(), v[1]]
                else:
                    new_kb_track[k] = [None, v[1]]
            else:
                new_kb_track[k] = v.deepcopy()
        return new_kb_track

    def init_knowledge_base(self, start_state):
        self.knowledge_base = {}
        for obj in start_state.all_objects_list:
            self.knowledge_base[obj.id] = obj

        self.knowledge_base['pot_states'] = self.mlp.mdp.get_pot_states(start_state)
        self.knowledge_base['sink_states'] = self.mlp.mdp.get_sink_status(start_state)
        self.knowledge_base['chop_states'] = self.mlp.mdp.get_chopping_board_status(start_state)
        self.knowledge_base['other_player'] = start_state.players[0]

        self.kb_update_delay_track[0]['human_holding'] = [None, self.kb_update_delay]
        self.kb_update_delay_track[1]['other_player'] = start_state.players[0].deepcopy()
        self.kb_update_delay_track[2] = {}

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

    def update_kb_log(self):
        self.kb_log += [self.get_kb_key(self.knowledge_base)]
        
    def get_kb_key(self, kb):
        num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        kb_key = '.'.join([str(num_item_in_pot), str(chop_time), str(wash_time), str(robot_obj)])
        return kb_key
    
    def get_knowledge_base(self, state, rollout_kb=None, rollout_info=[]):
        return self.update(state, rollout_kb=rollout_kb, rollout_info=rollout_info)

    def update_kb_key_object(self, obj, tmp_kb, prev_kb):
        if 'steak' in obj.name:
            if obj.position in self.mlp.mdp.get_pot_locations():
                _, _, cooking_time = obj.state
                if cooking_time < self.mlp.mdp.steak_cooking_time:
                    if obj.position in tmp_kb['pot_states']['empty']:
                        tmp_kb['pot_states']['empty'].remove(obj.position)
                    if obj.position in tmp_kb['pot_states'][obj.name]['empty']:
                        tmp_kb['pot_states'][obj.name]['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['pot_states'][obj.name]['cooking']:
                        tmp_kb['pot_states'][obj.name]['cooking'].append(obj.id)
                    if obj.id in tmp_kb['pot_states'][obj.name]['ready']:
                        tmp_kb['pot_states'][obj.name]['ready'].remove(obj.id)
                else:
                    if obj.position in tmp_kb['pot_states']['empty']:
                        tmp_kb['pot_states']['empty'].remove(obj.position)
                    if obj.position in tmp_kb['pot_states'][obj.name]['empty']:
                        tmp_kb['pot_states'][obj.name]['empty'].remove(obj.position)
                    if obj.id in tmp_kb['pot_states'][obj.name]['cooking']:
                        tmp_kb['pot_states'][obj.name]['cooking'].remove(obj.id)
                    if obj.id not in tmp_kb['pot_states'][obj.name]['ready']:
                        tmp_kb['pot_states'][obj.name]['ready'].append(obj.id)
            else:
                if obj.id in prev_kb.keys():
                    # NOTE: this only works if we have one pot
                    empty_pot_loc = self.mlp.mdp.get_pot_locations()[0] # prev_kb[obj.id].position
                    #if obj.id in tmp_kb['pot_states'][obj.name]['ready'] or obj.id in tmp_kb['pot_states'][obj.name]['cooking']:
                    if empty_pot_loc not in tmp_kb['pot_states'][obj.name]['empty']:
                        tmp_kb['pot_states'][obj.name]['empty'].append(empty_pot_loc)
                    if empty_pot_loc not in tmp_kb['pot_states']['empty']:
                        tmp_kb['pot_states']['empty'].append(empty_pot_loc)

                    if obj.id in tmp_kb['pot_states'][obj.name]['ready']:
                        tmp_kb['pot_states'][obj.name]['ready'].remove(obj.id)
                    if obj.id in tmp_kb['pot_states'][obj.name]['cooking']:
                        tmp_kb['pot_states'][obj.name]['cooking'].remove(obj.id)

        elif 'garnish' in obj.name:
            chop_time = obj.state
            if obj.position in self.mlp.mdp.get_chopping_board_locations():
                if chop_time >= 0 and chop_time < self.mlp.mdp.chopping_time:
                    if obj.position in tmp_kb['chop_states']['empty']:
                        tmp_kb['chop_states']['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['chop_states']['full']:
                        tmp_kb['chop_states']['full'].append(obj.id)
                    if obj.id in tmp_kb['chop_states']['ready']:
                        tmp_kb['chop_states']['ready'].remove(obj.id)
                elif chop_time >= self.mlp.mdp.chopping_time:
                    if obj.position in tmp_kb['chop_states']['empty']:
                        tmp_kb['chop_states']['empty'].remove(obj.position)
                    if obj.id in tmp_kb['chop_states']['full']:
                        tmp_kb['chop_states']['full'].remove(obj.id)
                    if obj.id not in tmp_kb['chop_states']['ready']:
                        tmp_kb['chop_states']['ready'].append(obj.id)
                else:
                    print(tmp_kb['chop_states'])
                    print(obj)
                    raise ValueError()
            else:
                if obj.id in prev_kb.keys():
                    # NOTE: this only works if we have one chopping board
                    empty_board_loc = self.mlp.mdp.get_chopping_board_locations()[0] #prev_kb[obj.id].position
                    # if obj.id in tmp_kb['chop_states']['ready'] or obj.id in tmp_kb['chop_states']['full']:
                    if empty_board_loc not in tmp_kb['chop_states']['empty']:
                        tmp_kb['chop_states']['empty'].append(empty_board_loc)
                    if obj.id in tmp_kb['chop_states']['ready']:
                        tmp_kb['chop_states']['ready'].remove(obj.id)
                    if obj.id in tmp_kb['chop_states']['full']:
                        tmp_kb['chop_states']['full'].remove(obj.id)

        elif 'hot_plate' in obj.name:
            wash_time = obj.state
            if obj.position in self.mlp.mdp.get_sink_locations():
                if wash_time >= 0 and wash_time < self.mlp.mdp.wash_time:
                    if obj.position in tmp_kb['sink_states']['empty']:
                        tmp_kb['sink_states']['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['sink_states']['full']:
                        tmp_kb['sink_states']['full'].append(obj.id)
                    if obj.id in tmp_kb['sink_states']['ready']:
                        tmp_kb['sink_states']['ready'].remove(obj.id)
                elif wash_time >= self.mlp.mdp.wash_time:
                    if obj.position in tmp_kb['sink_states']['empty']:
                        tmp_kb['sink_states']['empty'].remove(obj.position)
                    if obj.id in tmp_kb['sink_states']['full']:
                        tmp_kb['sink_states']['full'].remove(obj.id)
                    if obj.id not in tmp_kb['sink_states']['ready']:
                        tmp_kb['sink_states']['ready'].append(obj.id)
                else:
                    print(tmp_kb['sink_states'])
                    print(obj)
                    raise ValueError()
            else:
                if obj.id in prev_kb.keys():
                    ## NOTE: this only works when we have one sink
                    empty_sink_loc = self.mlp.mdp.get_sink_locations()[0] # prev_kb[obj.id].position
                    # if obj.id in tmp_kb['sink_states']['ready'] or obj.id in tmp_kb['sink_states']['full']:
                    if empty_sink_loc not in tmp_kb['sink_states']['empty']:
                        tmp_kb['sink_states']['empty'].append(empty_sink_loc)
                    if obj.id in tmp_kb['sink_states']['ready']:
                        tmp_kb['sink_states']['ready'].remove(obj.id)
                    if obj.id in tmp_kb['sink_states']['full']:
                        tmp_kb['sink_states']['full'].remove(obj.id)
        return tmp_kb

    def update(self, state, rollout_kb=None, rollout_info=None):
        # right_pt, left_pt = self.get_vision_bound(state, half_bound=self.vision_bound/2)
        other_player = state.players[1 - self.agent_index]

        if rollout_kb is not None:
            [rollout_track, rollout_untrack, rollout_remove] = rollout_info
            prev_kb = copy.deepcopy(rollout_kb)
            prev_track = self.kb_track_deepcopy(rollout_track)
            prev_untrack = self.kb_track_deepcopy(rollout_untrack)
            tmp_kb = copy.deepcopy(rollout_kb)
            tmp_track = self.kb_track_deepcopy(rollout_track)
            tmp_untrack = self.kb_track_deepcopy(rollout_untrack)
            prev_remove_obj_id_list = self.kb_track_deepcopy(rollout_remove)
            tmp_remove_obj_id_list = self.kb_track_deepcopy(rollout_remove)
        else:
            prev_kb = copy.deepcopy(self.knowledge_base) # copy.deepcopy(self.knowledge_base)
            prev_track = self.kb_track_deepcopy(self.kb_update_delay_track[0])
            prev_untrack = self.kb_track_deepcopy(self.kb_update_delay_track[1])
            tmp_kb = self.knowledge_base
            tmp_track = self.kb_update_delay_track[0]
            tmp_untrack = self.kb_update_delay_track[1]
            prev_remove_obj_id_list = self.kb_track_deepcopy(self.kb_update_delay_track[2])
            tmp_remove_obj_id_list = self.kb_update_delay_track[2]

        ### Update the tracking
        new_track, new_untrack = {}, {}
        other_player_dropped_obj_id = -1
        human_pickup_obj_id = -1
        other_player_dropped_obj_count = -1
        all_object_lists_id = {o.id:o for o in state.all_objects_list}
        # prev_track is a dictionary with object id as key and a list of [obj, consecutive seen count]

        for obj_key in set([o2 for o2 in all_object_lists_id] + [o1 for o1 in prev_track.keys()]):
            if obj_key in all_object_lists_id: obj = all_object_lists_id[obj_key]
            else: obj = prev_track[obj_key][0]

            if obj_key == 'human_holding':
                obj = state.players[self.agent_index].held_object

            if obj is not None:
                pos = other_player.position if obj_key == 'other_player' else obj.position
                obj_in_bound = (self.in_bound(state, pos, vision_bound=self.vision_bound/2))
                if obj_in_bound:
                    if 'other_player' == obj_key: #in prev_track.keys() and obj == prev_track['other_player'][0]:
                        obj_count = min(self.kb_update_delay, prev_track['other_player'][1] + 1)
                        tmp_track['other_player'] = [other_player.deepcopy(), obj_count]

                        # if the agent picked up an item, and that item was in untrack, we move it to tmp_remove_obj_id_list
                        if prev_track['other_player'][0].held_object == None and state.players[1-self.agent_index].held_object != None and state.players[1-self.agent_index].held_object.id in prev_untrack.keys():
                            tmp_remove_obj_id_list[state.players[1-self.agent_index].held_object.id] = [prev_untrack[state.players[1-self.agent_index].held_object.id].deepcopy(), obj_count]                  
                        
                        # if the agent drops item, remove the item from player's hand
                        if prev_track['other_player'][0].held_object != None and state.players[1-self.agent_index].held_object == None:
                            # if obj.id not in tmp_remove_obj_id_list.keys():
                            
                            new_obj = None
                            # set the newly dropped inbound object to have the same count as the threshold
                            if prev_track['other_player'][0].held_object.name == 'meat' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'onion' and state.all_objects_by_type['garnish']:
                                new_obj = state.all_objects_by_type['garnish'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'plate' and state.all_objects_by_type['hot_plate']:
                                new_obj = state.all_objects_by_type['hot_plate'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'hot_plate' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'steak' and state.all_objects_by_type['dish']:
                                new_obj = state.all_objects_by_type['dish'][-1]
                                for tmp_o_untrack_key, tmp_o_untrack in prev_track.items():
                                    if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish' and tmp_o_untrack[0].id not in tmp_remove_obj_id_list.keys():
                                        tmp_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), obj_count]
                            if new_obj != None:
                                tmp_remove_obj_id_list[prev_track['other_player'][0].held_object.id] = [prev_track['other_player'][0].held_object.deepcopy(), obj_count]
                                # flag the new_object and make sure the obj_count is not reset when dropped
                                other_player_dropped_obj_id = new_obj.id
                                other_player_dropped_obj_count = obj_count
                                tmp_track[new_obj.id] = [new_obj, obj_count]

                    elif human_pickup_obj_id == obj.id:
                        tmp_remove_obj_id_list[obj.id] = [obj.deepcopy(), self.kb_update_delay]

                    # is player object
                    elif (obj_key in all_object_lists_id) or (obj_key == 'human_holding' and obj.id in all_object_lists_id):
                        obj_count = 1
                        if obj.id in prev_track.keys(): obj_count = min(self.kb_update_delay, prev_track[obj.id][1] + 1)
                        elif obj.id in tmp_track.keys():
                            obj_count = tmp_track[obj.id][1]
                        elif other_player_dropped_obj_id == obj.id:
                            obj_count = other_player_dropped_obj_count
                            
                        if obj.position == state.players[self.agent_index].position:
                            obj_count = self.kb_update_delay
                            tmp_track['human_holding'] = [state.players[self.agent_index].held_object.deepcopy(), self.kb_update_delay]
                        tmp_track[obj.id] = [all_object_lists_id[obj.id].deepcopy(), obj_count]

                    # previous human held now dropped and changed object name
                    elif prev_track['human_holding'][0] != None and obj.id == prev_track['human_holding'][0].id and obj != None:
                        # if obj.id not in tmp_remove_obj_id_list.keys():
                        tmp_remove_obj_id_list[obj.id] = [obj.deepcopy(), self.kb_update_delay]

                        if obj.name == 'dish':
                            tmp_track['human_holding'] = [None, self.kb_update_delay]
                        else:
                            # set the newly dropped inbound object to have the same count as the threshold
                            if obj.name == 'meat':
                                new_obj = state.all_objects_by_type['steak'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay]
                            elif obj.name == 'onion':
                                new_obj = state.all_objects_by_type['garnish'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay]
                            elif obj.name == 'plate':
                                new_obj = state.all_objects_by_type['hot_plate'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay]
                            elif obj.name == 'hot_plate':
                                new_obj = None
                                for tmp_steak in state.all_objects_by_type['steak']:
                                    if self.in_bound(state, tmp_steak.position, vision_bound=self.vision_bound/2):
                                        new_obj = tmp_steak # can overwrite since the last item is the newest object
                                tmp_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay]
                            elif obj.name == 'steak':
                                new_obj = state.all_objects_by_type['dish'][-1]
                                tmp_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay]
                                for tmp_o_untrack_key, tmp_o_untrack in prev_track.items():
                                    if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish':
                                        human_pickup_obj_id = tmp_o_untrack[0].id
                                        tmp_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), self.kb_update_delay]
                                        
                            tmp_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay]

                    else:
                        # if obj_key != 'human_holding':
                        if obj_key not in tmp_remove_obj_id_list.keys():
                            tmp_remove_obj_id_list[obj_key] = [obj.deepcopy(), 0]

                else:
                    if 'other_player' == obj_key: # and obj_id == prev_track['other_player'][0].id:
                        tmp_untrack['other_player'] = prev_track['other_player'][0].deepcopy()
                        del tmp_track['other_player']
                    elif obj_key in prev_track.keys():
                        tmp_untrack[obj_key] = prev_track[obj_key][0].deepcopy()
                        del tmp_track[obj_key]

        if 'other_player' not in tmp_track.keys() and self.in_bound(state, other_player.position, vision_bound=self.vision_bound/2):
            tmp_track['other_player'] = [other_player.deepcopy(), 1]

            # if the agent picked up an item
            if 'other_player' in prev_untrack.keys() and state.players[1-self.agent_index].held_object != None and state.players[1-self.agent_index].held_object.id in prev_untrack.keys() and prev_untrack['other_player'].held_object == None and state.players[1-self.agent_index].held_object != None:
                tmp_remove_obj_id_list[state.players[1-self.agent_index].held_object.id] = [prev_untrack[state.players[1-self.agent_index].held_object.id].deepcopy(), 0]

            ## if the agent dropped an item
            # if 'other_player' in prev_track.keys() and prev_track['other_player'][0].held_object != None and prev_track['other_player'][0].held_object == None:
            #     # if obj.id not in tmp_remove_obj_id_list.keys():
            #     tmp_remove_obj_id_list[prev_track['other_player'][0].held_object.id] = [prev_track['other_player'][0].held_object.deepcopy(), 0]

        for obj_id, obj in prev_untrack.items():
            obj_in_bound = self.in_bound(state, obj.position, vision_bound=self.vision_bound/2)
            if obj_in_bound:
                if obj_id in all_object_lists_id.keys() and obj_id in tmp_track.keys():
                    # means is not in sight but still exists
                    del tmp_untrack[obj_id]
                elif obj_id == 'other_player' and obj_id in tmp_track.keys():
                    # new position inbound
                    del tmp_untrack['other_player']
                else:
                    if obj_id not in tmp_remove_obj_id_list.keys():
                        tmp_remove_obj_id_list[obj_id] = [obj.deepcopy(), 0]

        ### Update the KB with the removed object list
        del_remove_list = []
        for obj_id in tmp_remove_obj_id_list.keys():
            if obj_id in tmp_track.keys(): del tmp_track[obj_id]
            if obj_id in tmp_untrack.keys(): del tmp_untrack[obj_id]

            if self.in_bound(state, tmp_remove_obj_id_list[obj_id][0].position, vision_bound=self.vision_bound/2):
                tmp_remove_obj_id_list[obj_id][1] += 1

            if tmp_remove_obj_id_list[obj_id][1] >= self.kb_update_delay:
                if obj_id != 'other_player' and obj_id in tmp_kb.keys():
                    tmp_kb[obj_id].position = None
                    tmp_kb = self.update_kb_key_object(tmp_kb[obj_id], tmp_kb, prev_kb)
                else:
                    # do nothing
                    # tmp_kb['other_player'].position = None
                    # tmp_kb['other_player'].held_object = None
                    pass
                ## Remove the object from all tracking
                if obj_id != 'other_player' and obj_id in tmp_kb.keys(): del tmp_kb[obj_id]

                del_remove_list.append(obj_id)
        
        for obj_id in del_remove_list:
            del tmp_remove_obj_id_list[obj_id]
            
        ### Update the KB with the updated tracking list
        for obj_id, [obj, obj_count] in tmp_track.items():
            if obj_count >= self.kb_update_delay and obj is not None and obj_id != 'human_holding':
                ## Update the object in KB
                tmp_kb[obj_id] = obj
                if obj_id != 'other_player':
                    tmp_kb = self.update_kb_key_object(obj, tmp_kb, prev_kb)
                else:
                    tmp_kb['other_player'] = obj
        
        ## print out knowledge base
        if rollout_kb is None and self.debug:
            print('tmp_kb:')
            for k, v in tmp_kb.items():
                print(k, ':', v)
            print('\ntmp_track:')
            for k, v in tmp_track.items():
                print(k, ':', v)
            print('\ntmp_untrack:')
            for k, v in tmp_untrack.items():
                print(k, ':', v)
            print('')
            for k, v in tmp_remove_obj_id_list.items():
                print(k, ':', v)
            print('')

        return tmp_kb, [tmp_track, tmp_untrack, tmp_remove_obj_id_list]
    
    def get_human_traj(self, world_state, human_subtask_obj):
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

    def ml_action(self, state, chosen_subtask = None):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        self.update(state)
        self.update_kb_log()
        
        player = state.players[self.agent_index]
        other_player = self.knowledge_base['other_player']
        am = self.mlp.ml_action_manager
        curr_pos_and_or = player.pos_and_or

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        sink_status = self.knowledge_base['sink_states']
        chopping_board_status = self.knowledge_base['chop_states']
        pot_states_dict = self.knowledge_base['pot_states']
        # for k, o in self.knowledge_base.items():
        #     if k not in ['sink_states', 'chop_states', 'pot_states', 'other_player']:
        #         if o.position in self.mlp.mdp.get_counter_locations():
        #             counter_objects[o.name] = [o.position]
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        empty_pot = pot_states_dict['empty']
        ready_steaks = pot_states_dict['steak']['ready'] 
        cooking_steaks = pot_states_dict['steak']['cooking']

        garnish_ready = len(chopping_board_status['ready']) > 0
        chopping = len(chopping_board_status['full']) > 0
        board_empty = len(chopping_board_status['empty']) > 0
        hot_plate_ready = (len(sink_status['ready']) > 0)# or (len(counter_objects['hot_plate']) > 0)
        rinsing = len(sink_status['full']) > 0
        sink_empty = len(sink_status['empty']) > 0
        motion_goals = []
        
        steak_nearly_ready = len(ready_steaks) > 0 or len(cooking_steaks) > 0# or len(counter_objects['steak']) > 0
        other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
        other_has_hot_plate = other_player.has_object() and other_player.get_object().name == 'hot_plate'
        other_has_steak = other_player.has_object() and other_player.get_object().name == 'steak'
        other_has_meat = other_player.has_object() and other_player.get_object().name == 'meat'
        other_has_onion = other_player.has_object() and other_player.get_object().name == 'onion'
        other_has_plate = other_player.has_object() and other_player.get_object().name == 'plate'

        if chosen_subtask is not None:
            if chosen_subtask == 'pickup_meat':
                motion_goals += am.pickup_meat_actions(counter_objects)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'pickup_onion':
                motion_goals += am.pickup_onion_actions(counter_objects)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'pickup_plate':
                motion_goals += am.pickup_plate_actions(counter_objects, state)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'pickup_hot_plate':
                motion_goals += am.pickup_hot_plate_from_sink_actions(counter_objects, state)#, knowledge_base=self.knowledge_base)
                if len(motion_goals) == 0:
                    if other_has_plate:
                        motion_goals += self.mlp.mdp.get_sink_status(state)['full']
                        if len(motion_goals) == 0:
                            tmp_goal = self.mlp.mdp.get_sink_status(state)['empty']
                            motion_goals += self.go_to_closest_feature_or_counter_to_goal(self.mlp.motion_planner.motion_goals_for_pos[tmp_goal], tmp_goal)
            elif chosen_subtask == 'pickup_steak':
                motion_goals = am.pickup_steak_with_hot_plate_actions(self.mlp.mdp.get_pot_states(state))#, only_nearly_ready=True, knowledge_base=self.knowledge_base)
                if len(motion_goals) == 0:
                    if other_has_meat:
                        motion_goals = self.mlp.mdp.get_pot_states(state)['steak']['cooking'] + self.mlp.mdp.get_pot_states(state)['steak']['partially_full']
                        if len(motion_goals) == 0:
                            tmp_goal = self.mlp.mdp.get_pot_states(state)['empty']
                            motion_goals += self.go_to_closest_feature_or_counter_to_goal(self.mlp.motion_planner.motion_goals_for_pos[tmp_goal], tmp_goal)
            elif chosen_subtask == 'pickup_garnish':
                motion_goals = am.add_garnish_to_steak_actions(state)#, knowledge_base=self.knowledge_base)
                if len(motion_goals) == 0:
                    if other_has_onion:
                        motion_goals = self.mlp.mdp.get_chopping_board_status(state)['full']
                        if len(motion_goals) == 0:
                            tmp_goal = self.mlp.mdp.get_chopping_board_status(state)['empty']
                            motion_goals += self.go_to_closest_feature_or_counter_to_goal(self.mlp.motion_planner.motion_goals_for_pos[tmp_goal], tmp_goal)
            elif chosen_subtask == 'drop_meat':
                motion_goals = am.put_meat_in_pot_actions(self.mlp.mdp.get_pot_states(state))#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'drop_onion':
                motion_goals = am.put_onion_on_board_actions(state)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'drop_plate':
                motion_goals = am.put_plate_in_sink_actions(counter_objects, state)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'chop_onion':
                motion_goals += am.chop_onion_on_board_actions(state)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'heat_hot_plate':
                motion_goals += am.heat_plate_in_sink_actions(state)#, knowledge_base=self.knowledge_base)
            elif chosen_subtask == 'deliver_dish':
                motion_goals = am.deliver_dish_actions()

            motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            
        else:
            if not player.has_object():

                # if 'dish' in counter_objects.keys():
                #     motion_goals = counter_objects['dish']
                # elif garnish_ready and not other_has_steak and 'steak' in counter_objects.keys():
                #     motion_goals += counter_objects['steak']
                # elif steak_nearly_ready and not other_has_hot_plate and 'hot_plate' in counter_objects.keys():
                #     motion_goals = counter_objects['hot_plate']
                
                if chopping and not garnish_ready and self.prev_chosen_subtask in ['drop_onion', 'chop_onion']:
                    motion_goals += am.chop_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'chop_onion'
                elif rinsing and not hot_plate_ready and self.prev_chosen_subtask in ['drop_plate', 'heat_hot_plate']:
                    motion_goals += am.heat_plate_in_sink_actions(state, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'heat_hot_plate'
                elif not steak_nearly_ready and state.num_orders_remaining > 0 and not other_has_meat:
                    motion_goals += am.pickup_meat_actions(counter_objects, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_meat'
                elif not chopping and not garnish_ready and not other_has_onion:
                    motion_goals += am.pickup_onion_actions(counter_objects, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_onion'
                elif not rinsing and not hot_plate_ready and not other_has_plate:
                    motion_goals += am.pickup_plate_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_plate'
                elif (garnish_ready or chopping or other_has_onion) and hot_plate_ready and steak_nearly_ready:# and not (other_has_hot_plate):# or other_has_steak)
                    motion_goals += am.pickup_hot_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_hot_plate'
                # elif (garnish_ready or other_has_onion) and steak_nearly_ready and other_has_plate and not (other_has_hot_plate or other_has_steak) and not hot_plate_ready:
                #     motion_goals += am.pickup_hot_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                #     chosen_subtask = 'pickup_hot_plate'
                    if len(motion_goals) == 0:
                        if other_has_plate:
                            motion_goals += self.knowledge_base['sink_states']['full']
                            if len(motion_goals) == 0:
                                motion_goals += self.knowledge_base['sink_states']['empty']

                if len(motion_goals) == 0:
                    motion_goals += am.pickup_plate_actions(counter_objects, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_plate'
            else:
                player_obj = player.get_object()

                if player_obj.name == 'onion':
                    motion_goals = am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'drop_onion'
                elif player_obj.name == 'meat':
                    motion_goals = am.put_meat_in_pot_actions(pot_states_dict, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'drop_meat'
                elif player_obj.name == "plate":
                    motion_goals = am.put_plate_in_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                    # if (hot_plate_ready or rinsing) and garnish_ready and (steak_nearly_ready):
                    chosen_subtask = 'drop_plate'
                elif player_obj.name == 'hot_plate':
                    motion_goals = am.pickup_steak_with_hot_plate_actions(pot_states_dict, only_nearly_ready=True, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_steak'
                    if len(motion_goals) == 0:
                        if other_has_meat:
                            motion_goals = pot_states_dict['steak']['cooking'] + pot_states_dict['steak']['partially_full']
                            if len(motion_goals) == 0:
                                tmp_goal = pot_states_dict['empty']
                                motion_goals += self.go_to_closest_feature_or_counter_to_goal(self.mlp.motion_planner.motion_goals_for_pos[tmp_goal], tmp_goal)
                                

                elif player_obj.name == 'steak':
                    # would go towards chopping board if it does know the human picked up the onion
                    motion_goals = am.add_garnish_to_steak_actions(state, knowledge_base=self.knowledge_base)
                    chosen_subtask = 'pickup_garnish'
                    if len(motion_goals) == 0:
                        if other_has_onion:
                            motion_goals = self.knowledge_base['chop_states']['full']
                            if len(motion_goals) == 0:
                                tmp_goal = self.knowledge_base['chop_states']['empty']
                                motion_goals += self.go_to_closest_feature_or_counter_to_goal(self.mlp.motion_planner.motion_goals_for_pos[tmp_goal], tmp_goal)

                elif player_obj.name == 'dish':
                    motion_goals = am.deliver_dish_actions()
                    chosen_subtask = 'deliver_dish'
                # else:
                #     motion_goals += am.place_obj_on_counter_actions(state)

            motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        
            if len(motion_goals) == 0:
                if self.explore: # explore to expand the vision.
                    if player.has_object():
                        motion_goals += am.place_obj_on_counter_actions(state)
                    else:
                        motion_goals += am.pickup_plate_actions(counter_objects, knowledge_base=self.knowledge_base)

                        # get four directions to explore
                        for o in Direction.ALL_DIRECTIONS:
                            if o != player.orientation:
                                motion_goals.append(self.mdp._move_if_direction(player.position, player.orientation, o))
                        if player.pos_and_or in motion_goals:
                            motion_goals.remove(player.pos_and_or)
                    random.shuffle(motion_goals)
                    motion_goals = [[mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)][0]] # directly return on specific motion goal as the interact plan will always cost

                    assert len(motion_goals) != 0
                else: # get to the closest key object location
                    if player.has_object():
                        motion_goals += am.place_obj_on_counter_actions(state)
                    else:
                        motion_goals += am.pickup_plate_actions(counter_objects, knowledge_base=self.knowledge_base)
                        # motion_goals += am.go_to_closest_feature_actions(player)
                    motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
                    assert len(motion_goals) != 0

        self.prev_chosen_subtask = chosen_subtask
        # print('SteakLimitVisionHumanModel\'s motion_goals:', motion_goals)
        state.players[self.agent_index].subtask_log += [chosen_subtask]
        return motion_goals

class oneGoalHumanModel(Agent):
    """
    This human only executes one goal throughout the game.
    Goals include: deliver onion to pot, deliver soup
    
    Note: may not work if the onion can not be directly deliveried to the pot
    """
    def __init__(self, ml_action_manager, one_goal=None, auto_unstuck=False):
        self.ml_action_manager = ml_action_manager
        self.mdp = self.ml_action_manager.mdp
        self.mp = ml_action_manager.motion_planner

        # The one goal this agent excutes
        self.one_goal = one_goal # 'Onion cooker' or 'Soup server'

        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state):
        # possible_motion_goals = self.ml_action(state)
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # identify agent's goal
        sub_goals = {0:'Onion cooker', 1:'Soup server'}
        one_goal_motion_goals = []; WAIT = False
        if self.one_goal == sub_goals[0]:
            one_goal_motion_goals, WAIT = self.onion_cooker_ml_action(state)

        elif self.one_goal == sub_goals[1]:
            one_goal_motion_goals, WAIT = self.soup_server_ml_action(state)

        chosen_goal = []; chosen_action = []; action_probs = []
        if not WAIT:
            chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, one_goal_motion_goals)
            state.players[self.agent_index].active_log += [1]

        else: # if action is to stay at the same place
            # chosen_goal = one_goal_motion_goals[0]
            chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)
            state.players[self.agent_index].active_log += [0]

        if self.auto_unstuck:
            chosen_action, action_probs = self.resolve_stuck(state, chosen_action, action_probs)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state
            
        return chosen_action, {"action_probs": action_probs}

    def get_curr_env_status(self, state):
        counter_objects = self.mdp.get_counter_objects_dict(state, list(self.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if curr_order == 'any':
            ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
            cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        else:
            ready_soups = pot_states_dict[curr_order]['ready']
            cooking_soups = pot_states_dict[curr_order]['cooking']

        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0

        return counter_objects, pot_states_dict, ready_soups, cooking_soups, soup_nearly_ready

    def resolve_stuck(self, state, chosen_action, action_probs):
        # HACK: if two agents get stuck, select an action at random that would
        # change the player positions if the other player were not to move
        if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
            # if self.agent_index == 0:
            #     joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
            # elif self.agent_index == 1:
            #     joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
            # else:
            #     raise ValueError("Player index not recognized")
            joint_actions = list(itertools.product(Action.MOTION_ACTIONS, Action.MOTION_ACTIONS))
            unblocking_joint_actions = []
            for j_a in joint_actions:
                new_state, _, _, _ = self.mdp.get_state_transition(state, j_a)
                if new_state.player_positions != self.prev_state.player_positions:
                    unblocking_joint_actions.append(j_a)

            if len(unblocking_joint_actions) > 0:
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]
            else:
                chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)
            
            state.players[self.agent_index].stuck_log += [1]
        
        else:
            state.players[self.agent_index].stuck_log += [0]


        return chosen_action, action_probs        

    def onion_cooker_ml_action(self, state):
        WAIT = False
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]

        counter_objects = self.mdp.get_counter_objects_dict(state, list(self.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():
            motion_goals = self.ml_action_manager.pickup_onion_actions(counter_objects)

        else:
            player_obj = player.get_object()
            if player_obj.name == 'onion':
                motion_goals = self.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)
            else:
                motion_goals = self.ml_action_manager.place_obj_on_counter_actions(state)

        # motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        if (len(motion_goals)) == 0:
            WAIT = True

        return motion_goals, WAIT

    def soup_server_ml_action(self, state):
        WAIT = False
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]

        counter_objects = self.mdp.get_counter_objects_dict(state, list(self.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if curr_order == 'any':
            ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
            cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        else:
            ready_soups = pot_states_dict[curr_order]['ready']
            cooking_soups = pot_states_dict[curr_order]['cooking']

        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

        if not player.has_object():
            motion_goals = self.ml_action_manager.pickup_dish_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'dish' and soup_nearly_ready:
                motion_goals = self.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'dish' and not soup_nearly_ready:
                motion_goals = self.ml_action_manager.wait_actions(player)
                return motion_goals, True # because wait is not valid start goal pair

            elif player_obj.name == 'soup':
                motion_goals = self.ml_action_manager.deliver_soup_actions()

            else:
                motion_goals = self.ml_action_manager.place_obj_on_counter_actions(state)

        # motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        if (len(motion_goals)) == 0:
            WAIT = True

        return motion_goals, WAIT

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal, and returns the plan and the corresponding first action on that plan.
        """
        chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        # print(start_pos_and_or, motion_goals, chosen_goal, chosen_goal_action)
        action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]

        counter_objects = self.mdp.get_counter_objects_dict(state, list(self.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                motion_goals = self.ml_action_manager.pickup_dish_actions(counter_objects)
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order

                if next_order == 'onion':
                    motion_goals = self.ml_action_manager.pickup_onion_actions(counter_objects)
                elif next_order == 'tomato':
                    motion_goals = self.ml_action_manager.pickup_tomato_actions(counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = self.ml_action_manager.pickup_onion_actions(counter_objects) + self.ml_action_manager.pickup_tomato_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                motion_goals = self.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = self.ml_action_manager.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                motion_goals = self.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = self.ml_action_manager.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            motion_goals = self.ml_action_manager.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals


class biasHumanModel(oneGoalHumanModel):
    """
    The human has a preference distribution over being a type of one-goal human model. Distribution is pre-defined once the model is initiated. The final task the human chooses to execute depends on the initial distribution and the environment.
    """

    def __init__(self, ml_action_manager, goal_preference, adaptiveness, auto_unstuck=False):
        
        # A list containing probability of executing each task. The total probability should sum up to 1.
        self.goal_preference = np.array(goal_preference)
        self.adaptiveness = adaptiveness
        self.prev_goal_dstb = self.goal_preference
        self.sub_goals = {'Onion cooker':0, 'Soup server':1}
        self.prev_goal = None

        super().__init__(ml_action_manager, auto_unstuck=auto_unstuck)

        # print('Initial human task preference: {:.3f}% onion cooker, {:.3f}% soup server'.format(self.prev_goal_dstb[0]*100, self.prev_goal_dstb[1]*100))

    def action(self, state):
        start_pos_and_or = state.players_pos_and_or[self.agent_index]
        
        # identify agent's goal
        ml_logic_goals = self.logic_ml_action(state)
        # print(self.adaptiveness)
        curr_p = ((1.0-self.adaptiveness)*self.prev_goal_dstb + self.adaptiveness*ml_logic_goals)

        # print('Game logistics: {:.3f}% onion cooker, {:.3f}% soup server'.format(ml_logic_goals[0]*100, ml_logic_goals[1]*100))
        # print('Human task preference: {:.3f}% onion cooker, {:.3f}% soup server'.format(curr_p[0]*100, curr_p[1]*100))

        task = np.random.choice(len(self.sub_goals), p=curr_p)
        self.prev_goal_dstb = curr_p
        # print('Chosen task:', list(self.sub_goals.keys())[task], '\n')

        one_goal_motion_goals = []; WAIT = False
        if task == self.sub_goals['Onion cooker']:
            one_goal_motion_goals, WAIT = self.onion_cooker_ml_action(state)

        elif task == self.sub_goals['Soup server']:
            one_goal_motion_goals, WAIT = self.soup_server_ml_action(state)

        else:
            ValueError()

        chosen_goal = self.prev_goal; chosen_action = []; action_probs = []
        if not WAIT:
            chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, one_goal_motion_goals)
            self.prev_goal = chosen_goal
            state.players[self.agent_index].active_log += [1]
        else: # if action is to stay at the same place
            chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)
            state.players[self.agent_index].active_log += [0]

        if self.auto_unstuck:
            chosen_action, action_probs = self.resolve_stuck(state, chosen_action, action_probs)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state


        if state.players[self.agent_index].position == chosen_goal[0] and chosen_action == Action.INTERACT:
            # reset the task prob choice to the player's interest
            self.prev_goal_dstb = self.goal_preference

        return chosen_action, {"action_probs": action_probs}

    def logic_ml_action(self, state):
        """
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]

        counter_objects = self.mdp.get_counter_objects_dict(state, list(self.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        env_pref = np.zeros(len(self.sub_goals))

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                env_pref[self.sub_goals['Soup server']] += 1
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order

                if next_order == 'onion':
                    env_pref[self.sub_goals['Onion cooker']] += 1
                elif next_order == 'tomato':
                    # env_pref[self.sub_goals['Tomato cooker']] += 1
                    pass
                elif next_order is None or next_order == 'any':
                    env_pref[self.sub_goals['Onion cooker']] += 1
                    # env_pref[self.sub_goals['Tomato cooker']] += 1

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                env_pref[self.sub_goals['Onion cooker']] += 1

            elif player_obj.name == 'tomato':
                # env_pref[self.sub_goals['Tomato cooker']] += 1
                pass

            elif player_obj.name == 'dish':
                env_pref[self.sub_goals['Soup server']] += 1

            elif player_obj.name == 'soup':
                env_pref[self.sub_goals['Soup server']] += 1
            else:
                raise ValueError()

        if np.sum(env_pref) > 0.0:
            env_pref = env_pref/np.sum(env_pref)
        else:
            env_pref = np.fill(1.0/len(env_pref))

        return env_pref


class MdpPlanningAgent(Agent):

    def __init__(self, other_agent, mdp_planner, env, delivery_horizon=1, logging_level=0):
        self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mdp_planner = mdp_planner
        self.env = env
        self.logging_level = logging_level
        
    def action(self, state):
        state_str = self.mdp_planner.gen_state_dict_key(state)

        if state_str not in self.mdp_planner.state_idx_dict:
            # print('State = ', state_str, ';\nNot in dictionary. Action = North')
            action = Action.ALL_ACTIONS[0]#random.choice(Action.ALL_ACTIONS)
        else:
            action_idx = self.mdp_planner.policy_matrix[self.mdp_planner.state_idx_dict[state_str]]
            action = Action.INDEX_TO_ACTION[action_idx]

        return action, {}


class MediumMdpPlanningAgent(Agent):

    def __init__(self, mdp_planner, delivery_horizon=1, logging_level=0, auto_unstuck=False):
        # self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mdp_planner = mdp_planner
        self.logging_level = logging_level
        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def get_pot_status(self, state):
        pot_states = self.mdp_planner.mdp.get_pot_states(state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        return nearly_ready_pots

    def get_ml_states(self, state):
        num_item_in_pot = 0; pot_pos = []
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                # print(obj_state)
                if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]
                    pot_pos = obj_pos

        # print(pot_pos, num_item_in_pot)
        state_str = self.mdp_planner.gen_state_dict_key(state, state.players[1], num_item_in_pot, state.players[0])

        # print('State = ', state_str)

        return state_str

    def action(self, state):
        pot_states = self.mdp_planner.mdp.get_pot_states(state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        state_str = self.get_ml_states(state)
        action = []; chosen_action = []
        if state_str not in self.mdp_planner.state_idx_dict:
            # print('State = ', state_str, ';\nNot in dictionary. Action = North')
            action = Action.ALL_ACTIONS[0]#random.choice(Action.ALL_ACTIONS)
            state.players[self.agent_index].active_log += [0]
        
        else:
            # retrieve medium level action from policy
            action_idx = self.mdp_planner.policy_matrix[self.mdp_planner.state_idx_dict[state_str]]
            keys = list(self.mdp_planner.action_idx_dict.keys())
            vals = list(self.mdp_planner.action_idx_dict.values())
            action_object_pair = self.mdp_planner.action_dict[keys[vals.index(action_idx)]]
            # print(self.mdp_planner.state_idx_dict[state_str], action_idx, action_object_pair)

            # map back the medium level action to low level action
            possible_motion_goals = self.mdp_planner.map_action_to_location(state, state_str, action_object_pair[0], action_object_pair[1])

            # initialize
            action = Action.STAY
            minimum_cost = 100000.0
            # print(state)
            # print('possible_motion_goals =', possible_motion_goals)

            for possible_location in possible_motion_goals:
                motion_goal_locations = self.mdp_planner.mp.motion_goals_for_pos[possible_location]
                for motion_goal_location in motion_goal_locations:
                    if self.mdp_planner.mp.is_valid_motion_start_goal_pair((state.players[1].position, state.players[1].orientation), motion_goal_location):
                        action_plan, _, cost = self.mdp_planner.mp._compute_plan((state.players[1].position, state.players[1].orientation), motion_goal_location)
                        if cost < minimum_cost:
                            minimum_cost = cost
                            action = action_plan[0]

            # print(action)
        action_probs = self.a_probs_from_action(action)
        if self.auto_unstuck:
            action, action_probs = self.resolve_stuck(state, action, action_probs)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state

        if action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]

        return action, {"action_probs": action_probs}

    def resolve_stuck(self, state, chosen_action, action_probs):
        # HACK: if two agents get stuck, select an action at random that would
        # change the player positions if the other player were to move
        if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
            joint_actions = list(itertools.product(Action.MOTION_ACTIONS, Action.MOTION_ACTIONS))
            unblocking_joint_actions = []
            for j_a in joint_actions:
                new_state, _, _, _ = self.mdp_planner.mdp.get_state_transition(state, j_a)
                if new_state.player_positions != self.prev_state.player_positions:
                    unblocking_joint_actions.append(j_a)

            if len(unblocking_joint_actions) > 0:
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]
            else:
                chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)
        
            state.players[self.agent_index].stuck_log += [1]
        
        else:
            state.players[self.agent_index].stuck_log += [0]
        
        return chosen_action, action_probs


# class AbstractPlanningAgent(Agent):
#     def __init__(self, mdp_planner, greedy=False, other_agent=None, delivery_horizon=1, logging_level=0, auto_unstuck=False, low_level_action_flag=True):
#         self.delivery_horizon = delivery_horizon

#     def action(self, state, track_belief=False):
#         '''
#         Update current observation of the world and perform next action.
#         Input: world state
#         Output: 
#         '''
#         num_item_in_pot = 0
#         # update the observation's pot status by looking at the pot status in the world
#         if state.objects is not None and len(state.objects) > 0:
#             for obj_pos, obj_state in state.objects.items():
#                 # print(obj_state)
#                 if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
#                     num_item_in_pot = obj_state.state[1]

#         self.belief = self.mdp_planner.belief_update(state, num_item_in_pot)
#         action = self.mdp_planner.step(state)

class MediumQMdpPlanningAgent(Agent):
    def __init__(self, mdp_planner, greedy=False, other_agent=None, delivery_horizon=1, logging_level=0, auto_unstuck=False, low_level_action_flag=True, vision_limit=False):
        '''
        AI agent index is 0. Human index is 1.
        '''

        # self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mdp_planner = mdp_planner
        self.logging_level = logging_level
        self.auto_unstuck = auto_unstuck
        self.other_agent = other_agent
        self.greedy_known = greedy
        self.low_level_action_flag = low_level_action_flag
        self.vision_limit = vision_limit
        self.prev_stuck = False
        self.prev_action_info = [np.full((len(self.mdp_planner.subtask_dict)), 1.0/len(self.mdp_planner.subtask_dict), dtype=float), None, None, None]
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_dist_to_feature = {}
        self.belief = np.full((len(self.mdp_planner.subtask_dict)), 1.0/len(self.mdp_planner.subtask_dict), dtype=float)
        self.track_belief = [np.full((len(self.mdp_planner.subtask_dict)), 1.0/len(self.mdp_planner.subtask_dict), dtype=float)]
        self.prev_action_info = [np.full((len(self.mdp_planner.subtask_dict)), 1.0/len(self.mdp_planner.subtask_dict), dtype=float), None, None, None]
        self.prev_chosen_action = None

    def mdp_action_to_low_level_action(self, state, state_strs, action_object_pair):
        # map back the medium level action to low level action
        ai_agent_obj = state.players[0].held_object.name if state.players[0].held_object is not None else 'None'
        # print(ai_agent_obj)
        possible_motion_goals, WAIT = self.mdp_planner.map_action_to_location(state, state_strs[0], action_object_pair[0], action_object_pair[1], p0_obj=ai_agent_obj, player_idx=0)

        # initialize
        action = Action.STAY
        minimum_cost = 100000.0
        # print(state)
        # print('possible_motion_goals =', possible_motion_goals)
        if not WAIT:
            for possible_location in possible_motion_goals:
                motion_goal_locations = self.mdp_planner.mp.motion_goals_for_pos[possible_location]
                for motion_goal_location in motion_goal_locations:
                    if self.mdp_planner.mp.is_valid_motion_start_goal_pair((state.players[0].position, state.players[0].orientation), motion_goal_location):
                        action_plan, _, cost = self.mdp_planner.mp._compute_plan((state.players[0].position, state.players[0].orientation), motion_goal_location)
                        if cost < minimum_cost:
                            minimum_cost = cost
                            action = action_plan[0]
        return action

    def action(self, state, track_belief=False, RECOMPUTE_SUBTASK=0.01, debug=False):
        LOW_LEVEL_ACTION = self.low_level_action_flag
        # update the belief in a state by the result of observations
        observed_info, robot_agent_obj, human_agent_obj = self.mdp_planner.observe(state, state.players[0], state.players[1])
        curr_state_str = '_'.join([str(s) for s in (observed_info + [robot_agent_obj] + [human_agent_obj])])
        [prev_belief, prev_state_str, prev_action, prev_action_object_pair] = self.prev_action_info
        action, action_object_pair = prev_action, prev_action_object_pair

        # reset belief once there is a subtask completed (aka, change in the robot's holding or human holding or observed world)
        prev_max_belief = list(self.mdp_planner.subtask_dict.keys())[np.argmax(self.belief)]
        if curr_state_str != prev_state_str:
            self.belief = np.full((len(self.mdp_planner.subtask_dict)), 1.0/len(self.mdp_planner.subtask_dict), dtype=float)

        tmp_belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(state, state.players[0], observed_info, state.players[1], self.belief, self.prev_dist_to_feature, greedy=self.greedy_known, vision_limit=self.vision_limit, prev_max_belief=prev_max_belief)

        # do not update belief based on actions trying to resolve stuck
        if not self.prev_stuck:
            self.belief = tmp_belief
            
        # do not recompute next subtask if the belief change and observed_info is the same
        if debug: print('belief delta =', np.sum(np.abs(self.belief-prev_belief)))
        belief_delta = np.sum(np.abs(self.belief-prev_belief))
        
        # map abstract to low-level state
        mdp_state_keys = self.mdp_planner.world_to_state_keys(state, state.players[0], state.players[1], self.belief)

        # if belief_delta >= RECOMPUTE_SUBTASK or curr_state_str != prev_state_str or LOW_LEVEL_ACTION:
        
        # compute in low-level the action and cost
        # action, action_object_pair, _ = self.mdp_planner.old_step(state, mdp_state_keys, self.belief, self.agent_index, low_level_action=LOW_LEVEL_ACTION, observation=observed_info)
        
        action, action_object_pair, _, q = self.mdp_planner.step(state, dict(zip(list(self.mdp_planner.subtask_dict.keys()), self.belief)))

        if not LOW_LEVEL_ACTION:
            action = self.mdp_action_to_low_level_action(state, mdp_state_keys, action_object_pair)

        if debug: print('action =', action, '; action_object_pair =', action_object_pair)
        action_probs = self.a_probs_from_action(action)
        if self.auto_unstuck:
            action, action_probs, self.prev_stuck = self.resolve_stuck(state, action, action_probs)
            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state

        if action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]
        
        if debug: 
            print('\nState =', state)
            print('Subtasks:', self.mdp_planner.subtask_dict.keys())
            print('Belief =', self.belief)
            print('Max belief =', list(self.mdp_planner.subtask_dict.keys())[np.argmax(self.belief)])
            print('Action =', action, '\n')

        print('Subtasks:', self.mdp_planner.subtask_dict.keys())
        print('Belief =', self.belief)
        print('Max belief =', list(self.mdp_planner.subtask_dict.keys())[np.argmax(self.belief)])
        print('Action =', action, '\n')
            
        if track_belief:
            self.track_belief.append(self.belief)
        self.prev_action_info = [self.belief.copy(), curr_state_str, action, action_object_pair]

        return action, action_probs, q

    def resolve_stuck(self, state, chosen_action, action_probs):
        # HACK: if two agents get stuck and neither performing a pick or drop action, select an action at random that would
        # change the player positions if the other player were to move
        stuck_flag = False
        action_changed_world = False
        if self.prev_state is not None:
            i_pos = Action.move_in_direction(state.players[self.agent_index].position, state.players[self.agent_index].orientation)
            if self.prev_state.has_object(i_pos) and state.has_object(i_pos):
                obj0 = self.prev_state.get_object(i_pos).state
                obj1 = state.get_object(i_pos).state
                if obj0 != obj1:
                    action_changed_world = True
            elif self.prev_state.has_object(i_pos) or state.has_object(i_pos):
                action_changed_world = True

        if self.prev_state is not None and self.prev_state == state: #and (state.players_pos_and_or == self.prev_state.players_pos_and_or and (state.players[self.agent_index].held_object == self.prev_state.players[self.agent_index].held_object) and (self.prev_chosen_action !='interact' or (self.prev_chosen_action == 'interact' and not action_changed_world))):
        # if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or and state.players[0].held_object == self.prev_state.players[0].held_object and state.players[1].held_object == self.prev_state.players[1].held_object:
            # print('Resolving stuck...')
            joint_actions = list(itertools.product(Action.MOTION_ACTIONS, Action.MOTION_ACTIONS))
            unblocking_joint_actions = []
            for j_a in joint_actions:
                new_state, _, _, _ = self.mdp_planner.mdp.get_state_transition(state, j_a)
                if new_state.player_positions != self.prev_state.player_positions:
                    unblocking_joint_actions.append(j_a)

            if len(unblocking_joint_actions) > 0:
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]
            else:
                chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)
        
            state.players[self.agent_index].stuck_log += [1]
            stuck_flag = True
        else:
            state.players[self.agent_index].stuck_log += [0]
        
        self.prev_chosen_action = chosen_action
        
        return chosen_action, action_probs, stuck_flag


class HRLTrainingAgent(MediumQMdpPlanningAgent):
    def __init__(self, mdp, mdp_planner, other_agent=None, delivery_horizon=1, logging_level=0, auto_unstuck=False, qnet=None):
        super().__init__(mdp_planner, other_agent=other_agent, delivery_horizon=delivery_horizon, logging_level=logging_level, auto_unstuck=auto_unstuck)

        self.mdp = mdp

        self.state_idx_dict = None
        self.state_dict = None
        self.action_dict = None
        self.action_idx_dict = None
        self.subtask_dict = None
        self.env_items = self.encode_env()
        self.init_state_space()
        self.qnet = qnet

    def init_states(self, state_dict=None, state_idx_dict=None, order_list=None):
        # player_obj, num_item_in_pot, order_list

        if state_idx_dict is None:
            objects = ['onion', 'soup', 'dish', 'None'] # 'tomato'
            # common_actions = ['pickup', 'drop']
            # addition_actions = [('soup','deliver'), ('soup', 'pickup'), ('dish', 'pickup'), ('None', 'None')]
            # obj_action_pair = list(itertools.product(objects, common_actions)) + addition_actions

            state_keys = []; state_obj = []; tmp_state_obj = []

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

    def init_human_info_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pot, order list, agent 1 (usually human) holding object
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list) 

        # add [p1_obj] to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'soup', 'dish', 'None'] # 'tomato'
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for ori_key, ori_value in original_state_dict.items():
                new_key = ori_key+'_'+obj
                new_obj = original_state_dict[ori_key]+[obj]
                self.state_dict[new_key] = new_obj # update value
                self.state_idx_dict[new_key] = len(self.state_idx_dict)

        # print('init_human_info_states dict =', self.state_dict)

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
    
    def init_state_space(self):
        self.init_human_info_states(order_list=self.mdp.start_order_list)
        self.init_actions()

    def encode_env(self):
        onion_loc = self.mdp.get_onion_dispenser_locations()[0]
        pot_loc = self.mdp.get_pot_locations()[0]
        dish_loc = self.mdp.get_dish_dispenser_locations()[0]
        serving = self.mdp.get_serving_locations()[0]

        # create 4x2 matrix
        env_items = [[onion_loc[0], onion_loc[1]],
                    [pot_loc[0], pot_loc[1]],
                    [dish_loc[0], dish_loc[1]],
                    [serving[0], serving[1]]]

        return np.array(env_items)

    ### transition subtask actions to low level actions ###
    def mdp_action_to_low_level_action(self, state, state_strs, action_object_pair):
        # map back the medium level action to low level action
        ai_agent_obj = state.players[0].held_object.name if state.players[0].held_object is not None else 'None'
        # print(ai_agent_obj)
        possible_motion_goals, WAIT = self.mdp_planner.map_action_to_location(state, state_strs[0], action_object_pair[0], action_object_pair[1], p0_obj=ai_agent_obj, player_idx=0, counter_drop=False, state_dict=self.state_dict)

        # initialize
        action = Action.STAY
        minimum_cost = 100000.0
        # print(state)
        # print('possible_motion_goals =', possible_motion_goals)
        if not WAIT:
            for possible_location in possible_motion_goals:
                motion_goal_locations = self.mdp_planner.mp.motion_goals_for_pos[possible_location]
                for motion_goal_location in motion_goal_locations:
                    if self.mdp_planner.mp.is_valid_motion_start_goal_pair((state.players[0].position, state.players[0].orientation), motion_goal_location):
                        action_plan, _, cost = self.mdp_planner.mp._compute_plan((state.players[0].position, state.players[0].orientation), motion_goal_location)
                        if cost < minimum_cost:
                            minimum_cost = cost
                            action = action_plan[0]
        return action

    def lstate_to_hstate(self, state, to_str=False):
        # hstate: subtask state and key object position
        objects = {'onion': 0, 'soup': 1, 'dish': 2, 'None': 3}

        player = state.players[0]
        other_player = state.players[1]
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

        state_vector = np.array([objects[str(player_obj)], num_item_in_pot, len(state.order_list[1:]), objects[str(other_player_obj)]])
        
        if to_str:
            state_strs = str(player_obj)+'_'+str(num_item_in_pot)+'_'+ order_str + '_' + str(other_player_obj)
            return np.array(state_vector), state_strs

        return np.array(state_vector)

    
    def haction_to_laction(self, state, state_strs, a_idx):
        laction = self.mdp_action_to_low_level_action(state, [state_strs], list(self.mdp_planner.subtask_dict.values())[a_idx])

        return laction

    def action(self, state, q=None, track_belief=False):
        if q is None:
            q = self.qnet
        h_state, h_state_strs = self.lstate_to_hstate(state, to_str=True)
        h_env_state = np.concatenate((h_state.reshape(4,1), self.env_items), axis=1)
        a = q.sample_action(torch.from_numpy(h_env_state).float(), 0)
        action = self.haction_to_laction(state, h_state_strs, a)

        return action, {}


class QMDPAgent(Agent):
    def __init__(self, mlp, env, delivery_horizon=2, heuristic=None):
        self.mlp = mlp
        self.env = env
        self.mlp.failures = 0
        self.h_fn = Heuristic(mlp.mp).simple_heuristic
        self.delivery_horizon = delivery_horizon
    def action(self, state):
        #joint_action_plan = self.mlp.get_low_level_action_plan(state, self.heuristic, delivery_horizon=self.delivery_horizon, goal_info=True)
        start_state = state.deepcopy()
        order_list = start_state.order_list if start_state.order_list is not None else ["any", "any"]
        start_state.order_list = order_list[:self.delivery_horizon]
        initial_env_state = self.env.state
        expand_fn = lambda state: self.mlp.get_successor_states(state)
        goal_fn = lambda state: len(state.order_list) == 0
        heuristic_fn = lambda state: self.h_fn(state)
        succ_states = self.mlp.get_successor_states(start_state)
        succ_costs = [] # V value
        plans_to_succs = []
        imm_costs = [] # R value
        search_problem = SearchTree(succ_states[4][1], goal_fn, expand_fn, heuristic_fn)
        ml_plan, cost = search_problem.A_star_graph_search(info=True)

class HumanPlayer(Agent):
    """Agent to be controlled by human player.
    """
    def __init__(self,):
        self.prev_state = None

    def update_logs(self, state, action, curr_s=None):
        """Update necessary logs for calculating bcs."""
        state.players[self.agent_index].subtask_log += [curr_s]
        if action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]

        # get stuck if the players position and orientations do not change.
        if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
            state.players[self.agent_index].stuck_log += [1]
        else:
            state.players[self.agent_index].stuck_log += [0]

        self.prev_state = state

    def reset(self):
        super().reset()
        self.prev_state = None

class RLTrainingAgent(Agent):
    """
    Placeholder Agent used for RL training.
    """
    def __init__(self):
        pass

    def reset(self):
        super().reset()