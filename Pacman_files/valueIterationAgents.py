# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # start with all the states
        states = self.mdp.getStates()

        # Main loop. This is a for loop because
        # we're looping off of number of iterations
        # rather than delta in the book.
        for i in range(self.iterations):
            updated_values = self.values.copy() # Preserves values that are not changed

            for state in states:
                # Terminal states are done
                if self.mdp.isTerminal(state):
                    updated_values[state] = 0

                # Otherwise look at all actions and get the best one
                # aka V*(s) by way of Q*(s)
                else:
                    actions = self.mdp.getPossibleActions(state)
                    # handle no actions available
                    if actions == []:
                        updated_values[state] = 0
                    else:
                        q_values = [self.computeQValueFromValues(state,action) for action in actions]
                        updated_values[state] = max(q_values)
            self.values = updated_values 




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0

        # This is a set of (state, probability) tuples.
        transitions = self.mdp.getTransitionStatesAndProbs(state,action)
        
        for next_state, prob in transitions:
            reward = self.mdp.getReward(state,action,next_state)
            # We're iterating here, so Q = the sum of: probability (reward + discount * future values)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # handle terminal states
        if self.mdp.isTerminal(state):
            return None
        
        chosen_action = None
        value = -math.inf
        actions = self.mdp.getPossibleActions(state)

        # find the best value amongst available actions
        for action in actions:
            current_value = self.getQValue(state, action)
            if  current_value > value:
                value = current_value
                chosen_action = action
        return chosen_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # start with all the states
        states = self.mdp.getStates()

        # Main loop. This is a for loop because
        # we're looping off of number of iterations
        # rather than delta in the book.
        for i in range(self.iterations):
            # only select 1 state based on number of iterations.
            state = states[i % len(states)]

            # Terminal states are done
            if self.mdp.isTerminal(state):
                continue

            # Otherwise look at all actions and get the best one
            # aka V*(s) by way of Q*(s)
            else:
                actions = self.mdp.getPossibleActions(state)
                # handle no actions available
                if actions == []:
                    self.values[state] = 0
                else:
                    q_values = [self.computeQValueFromValues(state,action) for action in actions]
                    self.values[state] = max(q_values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computeMaxQValue(self, state):
        """Helper function to calculate max q_value"""
        actions = self.mdp.getPossibleActions(state)
        q_values = [self.computeQValueFromValues(state,action) for action in actions]
        return max(q_values)

    def runValueIteration(self):
        states = self.mdp.getStates()

        # compute predecessors of all states
        predecessors = {}   # can't use a counter here because we want a set of predecessors, not numbers
        for state in states:
            # ignore terminal states
            if not self.mdp.isTerminal(state):

                # get states, actions, and probabilities of successors
                # make this state a predecessor of each of those states.
                for action in self.mdp.getPossibleActions(state):
                    for child, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if child not in predecessors:
                            predecessors[child] = set()
                        predecessors[child].add(state)

        # initialize priority queue
        pq = util.PriorityQueue()

    
        # for each non-terminal state, place in priority queue based on error size.
        # Errors are negative so that larger errors rise to the top in the min heap.
        for state in states:
            if not self.mdp.isTerminal(state):
                q_max = self.computeMaxQValue(state)
                diff = abs(self.values[state] - q_max)
                pq.update(state, -diff)

        # iterate through value updates of each state via priority queue
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()

            # Update state value if not terminal.
            if not self.mdp.isTerminal(state):
                q_max = self.computeMaxQValue(state)
                self.values[state] = q_max
            
            # Add predecessors to queue for another update
            # IF they are not already in the queue and IF the error is greater than theta.
            for parent in predecessors[state]:
                q_max = self.computeMaxQValue(parent)
                diff = abs(self.values[parent] - q_max)
                if diff > self.theta:
                    pq.update(parent, -diff)




