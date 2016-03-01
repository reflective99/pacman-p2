# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    #print "Scores: ", scores
    bestScore = max(scores)
    #print "Best Score: ", bestScore
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    #print "Pacman's Current Pos: ", currentGameState.getPacmanPosition()
    newPos = successorGameState.getPacmanPosition()
    #print "New Pos: ", newPos
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    #print "New Ghost States: ", newGhostStates
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #print "New Scared Times: ", newScaredTimes

    "*** YOUR CODE HERE ***"
    newFoodList = newFood.asList()
    #print "New Food: ", newFood.asList()

    allGhostManDistances = []
    for ghost in newGhostStates:
        dist = util.manhattanDistance(ghost.getPosition(), newPos)
        allGhostManDistances.append(dist)

    minGhostDistance = min(allGhostManDistances)

    allFoodManDistances = []
    for food in newFoodList:
        if(newPos == food):
            continue
        else:
            dist = util.manhattanDistance(food, newPos)
            allFoodManDistances.append(dist)
    if(len(allFoodManDistances)==0):
        minFoodDistance = 100000000000000000
    else:
        minFoodDistance = min(allFoodManDistances)
    score = successorGameState.getScore()
    score += minGhostDistance/minFoodDistance

    #print "SCORE: ", score
    return score

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def terminalTest(self, gameState, actions, depth):
      if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
          return True
      else:
          return False


  def utility(self, gameState):
      return self.evaluationFunction(gameState)



  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    bestAction = self.minimaxDecision(gameState, 0, 0)
    return bestAction


  def minimaxDecision(self, gameState, depth, agent):
       minVal = float('-inf')
       actions = gameState.getLegalActions(agent)
       actions = [i for i in actions if i != Directions.STOP]
       bestAction = ''
       for action in actions:
           oldMin = minVal
           successorState = gameState.generateSuccessor(agent, action)
           #print "Min Val:", minVal
           minVal = max(minVal, self.minValue(successorState, 0, 1))
           if minVal > oldMin:
               bestAction = action
       #print "Min Val:", minVal
       return bestAction


  def maxValue(self, gameState, depth, agent):

      v = float('-inf')
      #print "NO RET MAX: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)
      actions = gameState.getLegalActions(agent)
      actions = [i for i in actions if i != Directions.STOP]
      if self.terminalTest(gameState, actions, depth):
          #print "MAX: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)
          return self.utility(gameState)

      for action in actions:
          successorState = gameState.generateSuccessor(agent, action)
          v = max(v, self.minValue(successorState, depth, agent+1))

      return v

  def minValue(self, gameState, depth, agent):

      v = float('inf')

      if agent == self.index:
          return self.maxValue(gameState, depth, agent)

      numGhosts = gameState.getNumAgents()-1
      actions = gameState.getLegalActions(agent)
      actions = [i for i in actions if i != Directions.STOP]

      if self.terminalTest(gameState, actions, depth):
       #print "MIN: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)
          return self.utility(gameState)

      for action in actions:
          successorState = gameState.generateSuccessor(agent, action)
          if agent < numGhosts:
              v = min(v, self.minValue(successorState, depth, agent + 1))
          if agent == numGhosts:
              v = min(v, self.maxValue(successorState, depth + 1, 0))
      return v


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def terminalTest(self, gameState, actions, depth):
      if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
          return True
      else:
          return False


  def utility(self, gameState):
      return self.evaluationFunction(gameState)



  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    currentDepth = 0
    currentAgent = self.index
    bestAction = self.alphaBetaSearch(gameState, 0, currentDepth)
    return bestAction

  def alphaBetaSearch(self, state, agent, depth):
      v = float('-inf')
      alpha = float('-inf')
      beta = float('inf')
      bestAction = ''
      actions = state.getLegalActions(agent)
      actions = [i for i in actions if i is not Directions.STOP]
      ghostIndex = 1
      for action in actions:
          successorState = state.generateSuccessor(0, action)
          oldV = v
          v = max(v, self.minValue(successorState, ghostIndex, depth, alpha, beta))
          if v > oldV:
              oldV = v
              bestAction = action

      print "Min Val:", v

      return bestAction



  def minValue(self, state, agent, depth, alpha, beta):
      #print "NO RET MIN: %d, D: %d, A: %d" % (self.evaluationFunction(state), depth, agent)
      actions = state.getLegalActions(agent)
      actions = [i for i in actions if i is not Directions.STOP]
      if agent == self.index:
          return self.maxValue(state, agent, depth, alpha, beta)

      if self.terminalTest(state, actions, depth):
          return self.utility(state)

      v = (float('inf'))
      numGhosts = state.getNumAgents()-1
      for action in actions:
          successorState = state.generateSuccessor(agent, action)
          if agent < numGhosts:
              v = min(v, self.minValue(successorState, agent + 1, depth, alpha, beta))
          if agent == numGhosts:
              v = min(v, self.maxValue(successorState, 0, depth + 1, alpha, beta))
          if v <= alpha:
              return v
          else:
              beta = min(beta, v)

      return v

  def maxValue(self, state, agent, depth, alpha, beta):
       #print "NO RET MAX: %d, D: %d, A: %d" % (self.evaluationFunction(state), depth, agent)
      v = float('-inf')
      #print "NO RET MAX: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)

      actions = state.getLegalActions(agent)
      actions = [i for i in actions if i is not Directions.STOP]

      if self.terminalTest(state, actions, depth):
          #print "MAX: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)
          return self.utility(state)

      for action in actions:
          successorState = state.generateSuccessor(agent, action)
          v = max(v, self.minValue(successorState, agent + 1, depth, alpha, beta))
          if v >= beta:
              return v
          else:
              alpha = max(alpha, v)

      return v



class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def terminalTest(self, gameState, actions, depth):
      if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
          return True
      else:
          return False

  def utility(self, gameState):
      return self.evaluationFunction(gameState)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    bestAction = self.expectiMaxDecision(gameState, 0, 0)
    return bestAction


  def expectiMaxDecision(self, gameState, depth, agent):
       minVal = float('-inf')
       actions = gameState.getLegalActions(agent)
       actions = [i for i in actions if i is not Directions.STOP]
       bestAction = ''
       for action in actions:
           oldVal = minVal
           successorState = gameState.generateSuccessor(agent, action)
           #print "Min Val:", minVal
           minVal = max(minVal, self.expectedValue(successorState, 0, 1))
           if minVal > oldVal:
               bestAction = action
       #print "Min Val:", minVal
       return bestAction


  def maxValue(self, gameState, depth, agent):

      v = float('-inf')
      #print "NO RET MAX: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)

      actions = gameState.getLegalActions(agent)
      actions = [i for i in actions if i is not Directions.STOP]
      if self.terminalTest(gameState, actions, depth):
          #print "MAX: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)
          return self.utility(gameState)

      for action in actions:
          successorState = gameState.generateSuccessor(agent, action)
          v = max(v, self.expectedValue(successorState, depth, agent + 1))
          # print "MaxValue for each successor", v, "D:", depth, "A:", agent

      return v

  def expectedValue(self, gameState, depth, agent):
      #print "expectedValue: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)

      v = 0

      if(agent == self.index):
          return self.maxValue(gameState, depth, agent)

      numGhosts = gameState.getNumAgents()-1
      actions = gameState.getLegalActions(agent)
      actions = [i for i in actions if i is not Directions.STOP]

      if self.terminalTest(gameState, actions, depth):
          #print "MIN: %d, D: %d, A: %d" % (self.evaluationFunction(gameState), depth, agent)
          return self.utility(gameState)

      ghostActions = len(gameState.getLegalActions(agent))

      for action in actions:
          successorState = gameState.generateSuccessor(agent, action)
          if agent < numGhosts:
              v += min(v, self.expectedValue(successorState, depth, agent + 1))/ghostActions
          if agent == numGhosts:
              v += min(v, self.maxValue(successorState, depth + 1, 0))/ghostActions

      return v


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
