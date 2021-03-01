import numpy as np
import heapq

from Solvers.Abstract_Solver import AbstractSolver, Statistics


class ValueIteration(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.V = np.zeros(env.nS)

    def train_episode(self):
        """
            Run a single episode of the Value Iteration Algorithm.

            Use:
                self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                    env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                    env.nS is a number of states in the environment.
                    env.nA is a number of actions in the environment.
                self.options.gamma: Gamma discount factor.
            """

        # Update each state...
        for s in range(self.env.nS):
            # Do a one-step lookahead to find the best action
            # Update the value function. Ref: Sutton book eq. 4.10.

            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            #Do a lookahead fuction to see the next state value
            A = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    A[a] += prob * (reward + self.options.gamma * self.V[next_state])
            #Do max opertation to find out the best action value 
            best_action_value = np.max(A)
            #Update the V for the given state
            self.V[s] = best_action_value

        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.

        Use:
            self.env.nA: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            #Define a state and policy variable
            s = state
            policy = np.zeros(self.env.nA)
            #Do a lookahead fuction to see the next state value
            A = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    A[a] += prob * (reward + self.options.gamma * self.V[next_state])
            #Figure out the best action from the action value using argmax operator
            best_action = np.argmax(A)
            #Set the best action policy
            policy[best_action] = 1
            return policy
        return policy_fn


class AsynchVI(ValueIteration):

    def __init__(self,env,options):
        super().__init__(env,options)
        # list of States to be updated by priority
        self.pq = PriorityQueue()
        # A mapping from each state to all states potentially leading to it in a single step
        self.pred = {}
        for s in range(self.env.nS):
            # Do a one-step lookahead to find the best action
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            self.pq.push(s, -abs(self.V[s]-best_action_value))
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def train_episode(self):
        """
        Run a *single* update for Asynchronous Value Iteration Algorithm (using prioritized sweeping).
        Updtae only one state, the one with the highest priority

        Use:
            self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            self.options.gamma: Gamma discount factor.
            self.pred[s]: a list of states leading to state s in one step with probability > 0
            self.pq: list of States to be updated by priority
        """

        #########################################################
        # YOUR IMPLEMENTATION HERE                              #
        # Choose state with the maximal value change potential  #
        # Do a one-step lookahead to find the best action       #
        # Update the value function. Ref: Sutton book eq. 4.10. #
        #########################################################
        # Choose state with the maximal value change potential
        high_Pstate = self.pq.pop()
        # Do a one-step lookahead to find the best action
        action = self.one_step_lookahead(high_Pstate)
        self.V[high_Pstate] = np.max(action)

        for sp in self.pred[high_Pstate]:
            temp = np.max(self.one_step_lookahead(sp))
            self.pq.update(sp, -abs(self.V[sp] - temp))
        
        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Asynchronous VI"

    def one_step_lookahead(self,state):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
