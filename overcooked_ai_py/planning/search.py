import heapq, time
import numpy as np
import scipy.sparse

from overcooked_ai_py.utils import MergePlanError


class SearchTree(object):
    """
    A class to help perform tree searches of various types. Once a goal state is found, returns a list of tuples
    containing (action, state) pairs. This enables to recover the optimal action and state path.
    
    Args:
        root (state): Initial state in our search
        goal_fn (func): Takes in a state and returns whether it is a goal state
        expand_fn (func): Takes in a state and returns a list of (action, successor, action_cost) tuples
        heuristic_fn (func): Takes in a state and returns a heuristic value
    """

    def __init__(self, root, goal_fn, expand_fn, heuristic_fn, max_iter_count=10e8, debug=False):
        self.debug = debug
        self.root = root
        self.is_goal = goal_fn
        self.expand = expand_fn
        self.heuristic_fn = heuristic_fn
        self.max_iter_count = max_iter_count

    def A_star_graph_search(self, info=False, time_limit=10e8):
        """
        Performs a A* Graph Search to find a path to a goal state
        """
        if info: print('A_star_graph_search')
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = PriorityQueue()

        root_node = SearchNode(self.root, action=None, parent=None, action_cost=0, debug=self.debug)
        pq.push(root_node, self.estimated_total_cost(root_node))
        while not pq.isEmpty():
            curr_node = pq.pop()
            iter_count += 1

            if self.debug and iter_count % 1000 == 0:
                print([p[0] for p in curr_node.get_path()])
                print(iter_count)

            curr_state = curr_node.state
            # print(iter_count, curr_state, curr_node.backwards_cost)
            # print(iter_count, curr_state.num_orders_remaining)

            if curr_state in seen:
                continue

            seen.add(curr_state)
            if iter_count > self.max_iter_count:
                print("Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")
            
            elapsed_time = time.time() - start_time
            if self.is_goal(curr_state) or elapsed_time > time_limit:
                if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(
                    elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
                return curr_node.get_path(), curr_node.backwards_cost
            
            successors = self.expand(curr_state)

            for action, child, cost in successors:
                child_node = SearchNode(child, action, parent=curr_node, action_cost=cost, debug=self.debug)
                pq.push(child_node, self.estimated_total_cost(child_node))

        print("Path for last node expanded: ", [p[0] for p in curr_node.get_path()])
        print("State of last node expanded: ", curr_node.state)
        print("Successors for last node expanded: ", self.expand(curr_state))
        raise TimeoutError("A* graph search was unable to find any goal state.")
    
    def coupled_A_star_graph_search(self, info=False, time_limit=10e8, path_limit=100):
        """
        Performs a A* Graph Search to find a path to a goal state
        """
        if info: print('A_star_graph_search')
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = PriorityQueue()
        kb_prob_track = {}

        root_node = SearchNode(self.root, action=None, parent=None, action_cost=0, debug=self.debug)
        pq.push(root_node, self.estimated_total_cost(root_node))
        while not pq.isEmpty():
            curr_node = pq.pop()
            iter_count += 1

            if self.debug and iter_count % 1000 == 0:
                print([p[0] for p in curr_node.get_path()])
                print(iter_count)

            curr_state = curr_node.state
            # print(iter_count, curr_state, curr_node.backwards_cost)
            # print(iter_count, curr_state.num_orders_remaining)

            if curr_state in seen:
                continue
            
            seen.add(curr_state)
            if iter_count > self.max_iter_count:
                print("Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")
            
            elapsed_time = time.time() - start_time
            if self.is_goal(curr_state) or elapsed_time > time_limit or curr_node.cost >= path_limit:
                if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(
                    elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
                return curr_node.get_path(), curr_node.backwards_cost
            
            successors = self.expand(curr_state, depth=curr_node.cost)

            for action, child, cost in successors:
                child_node = SearchNode(child, action, parent=curr_node, action_cost=cost, debug=self.debug)
                pq.push(child_node, self.estimated_total_cost(child_node))

        print("Path for last node expanded: ", [p[0] for p in curr_node.get_path()])
        print("State of last node expanded: ", curr_node.state)
        print("Successors for last node expanded: ", self.expand(curr_state))
        raise TimeoutError("A* graph search was unable to find any goal state.")

    def bounded_A_star_graph_search(self, qmdp_root=None, info=False, cost_limit=10e8, gamma=0, time_limit=10e8, debug=False):
        """
        Performs a A* Graph Search to find a path to a goal state
        """
        if info: print('A_star_graph_search')
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = PriorityQueue()
        pq_min_depth = cost_limit

        root_node = SearchNode(self.root, action=qmdp_root, parent=None, action_cost=0, debug=self.debug, gamma=gamma)
        pq.push(root_node, self.estimated_total_cost(root_node, gamma=gamma))
        # print('\n\n')
        while not pq.isEmpty():
            curr_node = pq.pop()
            iter_count += 1
            # if curr_node.depth < pq_min_depth:
            #     pq_min_depth = curr_node.depth

            if self.debug and iter_count % 1000 == 0 and debug:
                print([p[0] for p in curr_node.get_path()])
                print(iter_count)

            curr_state = curr_node.state
            curr_qmdp_state = curr_node.action

            if curr_qmdp_state in seen:
                continue

            seen.add(curr_qmdp_state)
            if iter_count > self.max_iter_count:
                print("Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")
            
            elapsed_time = time.time() - start_time
            if self.is_goal(curr_qmdp_state) or curr_node.depth > cost_limit: #or elapsed_time > time_limit 
                if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(
                    elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
                tmp_node = curr_node
                if debug:
                    print(curr_node.action, curr_node.gamma_cost, curr_node.state.players_pos_and_or, end='/')
                    while (tmp_node.parent is not None):
                        print(tmp_node.parent.action, tmp_node.parent.gamma_cost, tmp_node.parent.state.players_pos_and_or, end='/')
                        tmp_node = tmp_node.parent
                    print('')

                if gamma:
                    return curr_node.state, curr_node.gamma_cost, False, curr_node.action
                else:
                    return curr_node.state, curr_node.backwards_cost, False
            
            successors = self.expand(curr_state, curr_qmdp_state)
            # print('length of successors = {}'.format(len(successors)))
            # pq_min_depth = curr_node.depth

            for qmdp_state, child, cost in successors:
                # print(qmdp_state, child, cost)
                child_node = SearchNode(child, qmdp_state, parent=curr_node, action_cost=cost, debug=self.debug, gamma=gamma)
                # est_total_cost = self.estimated_total_cost(child_node)
                pq.push(child_node, self.estimated_total_cost(child_node, gamma=gamma))

        print("Path for last node expanded: ", [p[0] for p in curr_node.get_path()])
        print("State of last node expanded: ", curr_node.state)
        print("Successors for last node expanded: ", self.expand(curr_state, curr_qmdp_state))
        raise TimeoutError("A* graph search was unable to find any goal state.")

    def estimated_total_cost(self, node, gamma=False):
        """
        Calculates the estimated total cost of going from node to goal
        
        Args:
            node (SearchNode): node of the state we are interested in
        
        Returns:
            float: h(s) + g(s), where g is the total backwards cost
        """
        if gamma:
            return node.gamma_cost + self.heuristic_fn(node.state)
        
        return node.backwards_cost + self.heuristic_fn(node.state)

    def bfs(self, kb_key_root=None, info=False, time_limit=10e8, path_limit=100, search_depth=10):
        """
        Performs a bfs
        """
        if info: print('BFS')
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = []
        kb_prob_track = {}

        root_node = SearchNode(self.root, action=kb_key_root, parent=None, action_cost=0, debug=self.debug)
        pq.append([root_node, self.estimated_total_cost(root_node)])
        root_flag = True
        while not len(pq) == 0:
            [curr_node, _] = pq.pop()
            iter_count += 1

            if self.debug and iter_count % 1000 == 0:
                print([p[0] for p in curr_node.get_path()])
                print(iter_count)

            curr_state = curr_node.state
            curr_kb_key = curr_node.action
            # print(iter_count, curr_state, curr_node.backwards_cost)
            # print(iter_count, curr_state.num_orders_remaining)
            
            if not root_flag:
                if curr_kb_key not in kb_prob_track.keys():
                    kb_prob_track[curr_kb_key] = 1
                else:
                    kb_prob_track[curr_kb_key] += 1

            # if curr_state in seen:
            #     continue
            
            # seen.add(curr_state)

            if iter_count > self.max_iter_count:
                print("Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")
            
            elapsed_time = time.time() - start_time
            # if len(pq) == 0:# or curr_node.depth > path_limit:#or elapsed_time > time_limit: # the exit condition is set to be larger than search_depth is due to the fact that this is a FIFO queue, hence when the layer explored first exceeds the serach_depth, then it means all nodes of the search_depth layer are explored. 
            #     if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(
            #         elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
                
            #     for k in kb_prob_track.keys():
            #         kb_prob_track[k] /= len(seen)

            #     return curr_node.get_path(), curr_node.backwards_cost, kb_prob_track 

            if curr_node.depth < search_depth:
                successors = self.expand(curr_state, depth=curr_node.depth)

                for kb_key, child, cost in successors:
                    child_node = SearchNode(child, kb_key, parent=curr_node, action_cost=cost, debug=self.debug)
                    if child_node.depth <= search_depth:
                        pq.append([child_node, self.estimated_total_cost(child_node)])

            root_flag = False

        if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
        
        total_visited_seen = sum(kb_prob_track.values())
        for k in kb_prob_track.keys():
            kb_prob_track[k] /= total_visited_seen

        return curr_node.get_path(), curr_node.backwards_cost, kb_prob_track 
    
        print("Path for last node expanded: ", [p[0] for p in curr_node.get_path()])
        print("State of last node expanded: ", curr_node.state)
        print("Successors for last node expanded: ", self.expand(curr_state, depth=curr_state.depth))
        raise TimeoutError("A* graph search was unable to find any goal state.")

    def bfs_track_path(self, get_kb_key, kb_root=None, info=False, time_limit=10e8, path_limit=100, search_depth=10):
        """
        Performs a bfs
        """
        if info: print('BFS')
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = []
        kb_prob_track = {}

        root_node = SearchNode(self.root, action=kb_root, parent=None, action_cost=0, debug=self.debug)
        pq.append([root_node, self.estimated_total_cost(root_node)])
        # root_flag = True
        while not len(pq) == 0:
            [curr_node, _] = pq.pop()
            iter_count += 1

            if self.debug and iter_count % 1000 == 0:
                print([p[0] for p in curr_node.get_path()])
                print(iter_count)

            curr_state = curr_node.state
            curr_kb_key = get_kb_key(curr_node.action)
            # print(iter_count, curr_state, curr_node.backwards_cost)
            # print(iter_count, curr_state.num_orders_remaining)
            
            # if not root_flag:
            if curr_kb_key not in kb_prob_track.keys():
                    # we only track the largest probability when a change is seen so that later when we mulitply it on the state value, it gives the max possible value.
                kb_prob_track[curr_kb_key] = (1.0/(curr_node.depth+1), curr_node.depth, curr_node.state)
                # else:
                #     kb_prob_track[curr_kb_key] += 1

            if (str(curr_state), str(curr_node.action)) in seen:
                continue
            
            seen.add((str(curr_state), str(curr_node.action)))

            if iter_count > self.max_iter_count:
                print("Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")
            
            elapsed_time = time.time() - start_time
            # if len(pq) == 0:# or curr_node.depth > path_limit:#or elapsed_time > time_limit: # the exit condition is set to be larger than search_depth is due to the fact that this is a FIFO queue, hence when the layer explored first exceeds the serach_depth, then it means all nodes of the search_depth layer are explored. 
            #     if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(
            #         elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
                
            #     for k in kb_prob_track.keys():
            #         kb_prob_track[k] /= len(seen)

            #     return curr_node.get_path(), curr_node.backwards_cost, kb_prob_track 

            if curr_node.depth < search_depth:
                successors = self.expand(curr_state, curr_node.action, depth=curr_node.depth)

                for kb, child, cost in successors:
                    child_node = SearchNode(child, kb, parent=curr_node, action_cost=cost, debug=self.debug)
                    if child_node.depth <= search_depth:
                        pq.append([child_node, self.estimated_total_cost(child_node)])

            root_flag = False

        if info: print("Found goal after: \t{:.2f} seconds,   \t{} state expanded ({:.2f} unique) \t ~{:.2f} expansions/s".format(elapsed_time, iter_count, len(seen)/iter_count, iter_count/elapsed_time))
        
        return curr_node.get_path(), curr_node.backwards_cost, kb_prob_track 
    
        print("Path for last node expanded: ", [p[0] for p in curr_node.get_path()])
        print("State of last node expanded: ", curr_node.state)
        print("Successors for last node expanded: ", self.expand(curr_state, depth=curr_state.depth))
        raise TimeoutError("A* graph search was unable to find any goal state.")

class SearchNode(object):
    """
    A helper class that stores a state, action, and parent tuple and enables to restore paths
    
    Args:
        state (any): Game state corresponding to the node
        action (any): Action that brought to the current state
        parent (SearchNode): Parent SearchNode of the current SearchNode
        action_cost: Additional cost to get to this node from the parent
    """

    def __init__(self, state, action, parent, action_cost, debug=False, gamma=0.9):
        assert state is not None
        self.state = state
        # Action that led to this state
        self.action = action
        self.debug = debug
        self.discount_cost = 0.3
        self.gamma = gamma

        # Parent SearchNode
        self.parent = parent
        if parent != None:
            self.depth = self.parent.depth + 1
            self.backwards_cost = self.parent.backwards_cost + action_cost
            self.discount_cost = self.parent.discount_cost*(1.0-self.discount_cost) + action_cost*self.discount_cost
            # self.gamma_cost = self.parent.gamma_cost*(self.gamma) + action_cost
            self.gamma_cost = self.parent.gamma_cost + action_cost*(pow(self.gamma, self.depth))
        else:
            self.depth = 0
            self.backwards_cost = 0
            self.discount_cost = 0.0
            self.gamma_cost = 0.0

    def __lt__(self, other):
        return self.backwards_cost < other.backwards_cost

    def get_path(self):
        """
        Returns the path leading from the earliest parent-less node to the current
        
        Returns:
            List of tuples (action, state) where action is the action that led to the state.
            NOTE: The first entry will be (None, start_state).
        """
        path = []
        node = self
        while node is not None:
            path = [(node.action, node.state)] + path
            node = node.parent
        return path

class Graph(object):
    def __init__(self, dense_adjacency_matrix, encoder, decoder, debug=False):
        """
        Each graph node is distinguishable by a key, encoded by the encoder into 
        a index that corresponds to that node in the adjacency matrix defining the graph.

        Arguments:
            dense_adjacency_matrix: 2D array with distances between nodes
            encoder: Dictionary mapping each graph node key to the adj mtx index it corresponds to
            decoder: Dictionary mapping each adj mtx index to a graph node key
        """
        self.sparse_adjacency_matrix = scipy.sparse.csr_matrix(dense_adjacency_matrix)
        self.distance_matrix, self.predecessors = \
             scipy.sparse.csgraph.floyd_warshall(dense_adjacency_matrix, return_predecessors=True, overwrite=True)
        self._encoder = encoder
        self._decoder = decoder
        start_time = time.time()
        if debug: print("Computing shortest paths took {} seconds".format(time.time() - start_time))
        self._ccs = None

    @property
    def connected_components(self):
        if self._ccs is not None:
            return self._ccs
        else:
            self._ccs = self._get_connected_components()
            return self._ccs

    def dist(self, node1, node2):
        """
        Returns the calculated shortest distance between two nodes of the graph.
        Takes in as input the node keys.
        """
        idx1, idx2 = self._encoder[node1], self._encoder[node2]
        return self.distance_matrix[idx1][idx2]

    def get_children(self, node):
        """
        Returns a list of children node keys, given a node key.
        """
        edge_indx = self._get_children(self._encoder[node])
        nodes = [self._decoder[i] for i in edge_indx]
        return nodes

    def _get_children(self, node_index):
        """
        Returns a list of children node indices, given a node index.
        """
        assert node_index is not None
        # NOTE: Assuming successor costs are non-zero
        _, children_indices = self.sparse_adjacency_matrix.getrow(node_index).nonzero()
        return children_indices
        
    def get_node_path(self, start_node, goal_node):
        """
        Given a start node key and a goal node key, returns a list of
        node keys that trace a shortest path from start to goal.
        """
        start_index, goal_index = self._encoder[start_node], self._encoder[goal_node]
        index_path = self._get_node_index_path(start_index, goal_index)
        node_path = [self._decoder[i] for i in index_path]
        return node_path

    def _get_node_index_path(self, start_index, goal_index):
        """
        Given a start node index and a goal node index, returns a list of
        node indices that trace a shortest path from start to goal.
        """
        assert start_index is not None

        traceback_path = []
        cur_index = goal_index
        while cur_index != start_index:
            traceback_path.append(cur_index)

            if cur_index < 0:
                raise MergePlanError("INVALID CUR_INDEX IN TRACING PATH")

            cur_index = self.predecessors[start_index][cur_index]

        traceback_path.append(start_index)
        traceback_path.reverse()

        return traceback_path

    def _get_connected_components(self):
        num_ccs, cc_labels = scipy.sparse.csgraph.connected_components(self.sparse_adjacency_matrix)
        connected_components = [set() for _ in range(num_ccs)]
        for node_index, cc_index in enumerate(cc_labels):
            node = self._decoder[node_index]
            connected_components[cc_index].add(node)
        return connected_components

    def are_in_same_cc(self, node1, node2):
        idx1, idx2 = self._encoder[node1], self._encoder[node2]
        return self.predecessors[idx1,idx2] != -9999 or idx1 == idx2

class NotConnectedError(Exception):
    pass

class PriorityQueue:
    """Taken from UC Berkeley's CS188 project utils.

    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.

    Note that this PriorityQueue does not allow you to change the priority
    of an item. However, you may insert the same item multiple times with
    different priorities."""
    def  __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0
