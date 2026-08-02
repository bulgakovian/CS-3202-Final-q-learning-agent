# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # q_values is a counter with a (state, action) tuple for an index.
        self.q_values = util.Counter()



    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)

        # handle no legal actions
        if actions == []:
            return 0.0
        
        # Return Q* (best Q value from avaialble actions). 
        # Default is a safety valve I just learned; value returned when list empty.
        return max([self.getQValue(state, action) for action in actions], default=0)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        # handle no legal actions
        if actions == []:
            return None
        
        options = [] # list so that we can choose randomly if multiple options have same q-value
        value = self.computeValueFromQValues(state)
        for action in actions:
            if value == self.getQValue(state,action):
                options.append(action)
        
        if options == []:
            return None
        else:
            return random.choice(options)

        

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        actions = self.getLegalActions(state)
        # Handle no legal actions
        if actions == []:
            return None      

        # Random action at probability epsilon
        if util.flipCoin(self.epsilon):
            return random.choice(actions)

        # Otherwise use the policy
        # Find largest value
        max_value = -math.inf
        for action in actions:
            a_value = self.getQValue(state,action)
            if a_value >= max_value:
                max_value = a_value
        
        # Populate options with all max value actions
        options = []
        for action in actions:
            if self.getQValue(state,action) == max_value:
                options.append(action)
        return random.choice(options)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Computes new Q value
        # current value
        q = self.getQValue(state,action)
        # next value is max of existing values (0 catches empty lists)
        q_next = max([self.getQValue(nextState,next_action) for next_action in self.getLegalActions(nextState)], default = 0)
        self.q_values[(state,action)] = q + self.alpha * (reward + self.discount * q_next - q)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        # weights are a dictionary with feature name as key.
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # This line (which we use in the next function too) grabs a list of dictionary of features and their values.
        # It's a construct of the Berkely code and I don't quite get it.
        features = self.featExtractor.getFeatures(state,action)
        
        # This is basic linear algebra dot product: 
        # We're multiplying a vector of weights times values indexed by feature
        return sum([self.weights[feature] * value for feature, value in features.items()])

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state,action)

        # this is the formula given for Approximate Q learnining in question 10.
        # We don't actually learn about it until week 8, when the assignment is due.
        diff = (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))

        # For each feature, value pair, we update it with the formula in question 10.
        for feature, value in features.items():
            w = self.weights[feature]
            self.weights[feature] = w + self.alpha * diff * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
