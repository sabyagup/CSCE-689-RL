import numpy as np

from Solvers.Abstract_Solver import AbstractSolver, Statistics


#env = FrozenLake-v0


class PolicyIteration(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.V = np.zeros(env.nS)
        # Start with a random policy
        self.policy = np.ones([env.nS, env.nA]) / env.nA

    def train_episode(self):
        """
            Run a single Policy iteration. Evaluate and improves the policy.

            Use:
                self.options.env: OpenAI gym environment.
                self.options.gamma: Gamma discount factor.
                self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
                new_state, reward, done, _ = self.step(action): To advance one step in the environment

            """

        # Evaluate the current policy
        self.policy_eval()

        # For each state...
        for s in range(self.env.nS):
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            # Reset the environment
            state = self.env.reset()

            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            A = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    A[a] += prob * (reward + self.options.gamma * self.V[next_state])
            best_action = np.argmax(A)
            self.policy[s] = np.eye(self.env.nA)[best_action]
            

        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Policy Iteration"

    def approx_policy_eval(self, theta=0.00001):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.

        Use:
            self.policy: [S, A] shaped matrix representing the policy.
            self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            self.options.gamma: Gamma discount factor.
        """
        # Start with a random (all 0) value function
        self.V = np.zeros(self.env.nS)
        while True:
            delta = 0
            # For each state, perform a "full backup"
            for s in range(self.env.nS):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(self.policy[s]):
                    # For each action, look at the possible next states...
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        # Calculate the expected value
                        v += action_prob * prob * (reward + self.options.gamma * self.V[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - self.V[s]))
                self.V[s] = v
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break

    def policy_eval(self):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.
        Use a linear system solver sallied by numpy (np.linalg.solve)

        Use:
            self.policy: [S, A] shaped matrix representing the policy.
            self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            self.options.gamma: Gamma discount factor.
            np.linalg.solve(a, b) # Won't work with discount factor = 0!
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        #pass
        self.V = np.zeros(self.env.nS)
        T = np.zeros((self.env.nS, self.env.nS))
        R = np.zeros(self.env.nS)
    
        for s in range(self.env.nS):
            # Look at the possible next actions
            for a, action_prob in enumerate(self.policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in self.env.P[s][a]:
                    # Solve for V by calculating the expected value of Transition Matrix and Reward Matrix using https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html#policy-evaluation-using-linear-algebra
                    T[s, next_state] += self.options.gamma*prob*action_prob
                    R[s] += reward*prob*action_prob
                    self.V = np.linalg.solve(np.identity(self.env.nS) - T, R)

    def create_greedy_policy(self):
        """
        Return the currently known policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        def policy_fn(state):
            Q_sa = np.zeros(self.env.nA)
            policy = np.zeros(self.env.nA)

            for action in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    Q_sa[action] += prob*(reward + self.options.gamma*self.V[next_state])
            
            policy[np.argmax(Q_sa)] = 1
            return policy

        return policy_fn
