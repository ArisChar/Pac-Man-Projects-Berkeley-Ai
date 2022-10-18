# multiAgents.py
# --------------
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = childGameState.getScore()
        foodlist = newFood.asList()
        foodDist = []

        # Get distance between pacman and all the food and store it in an array
        for food in foodlist:
            foodDist.append(manhattanDistance(newPos, food))

        # Get score of food
        if (len(foodDist) != 0):
            score += 1.0/min(foodDist)

        # Get score of ghost
        for ghost in newGhostStates:
            ghostPos =  ghost.getPosition()
            # If there is a ghost near that can hunt you run
            if manhattanDistance(newPos, ghostPos) <= 1 and ghost.scaredTimer == 0:
                score = float("-inf")

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # ------------------------------------------------------------------- #
        def minimax(gameState, agentIndex, depth):

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Get possible moves
            legalMoves = gameState.getLegalActions(agentIndex)
            # Get next agentsindex
            next = agentIndex + 1
            # list to store the score
            values = []

            # if Pacman
            if agentIndex == 0: 
                
                for move in legalMoves:
                    NextState = gameState.getNextState(agentIndex, move)
                    values.append(minimax(NextState, next,  depth))
                # Return biggest score from list
                return max(values)
            # if ghost   
            else:  

                for move in legalMoves:

                    NextState = gameState.getNextState(agentIndex, move)
                    # If it is the last ghost increase depth by one 
                    if (gameState.getNumAgents() == next):
                        next = 0
                        depth += 1
                        values.append(minimax(NextState, next,  depth))
                    # keep the same depth
                    else:
                        values.append(minimax(NextState, next,  depth))
                # Return smallest score from list
                return min(values)
        # ------------------------------------------------------------------- #

        legalMoves = gameState.getLegalActions(0)
        # List to store all [action,score] pair to then select the best action
        actions = []   
        
        for move in legalMoves:
            NextState = gameState.getNextState(0, move)
            actions.append((minimax(NextState, 1, 0), move))
        # Get  best action depending on score 
        action = max(actions, key = lambda x : x[0])[1]
        return action
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # ------------------------------------------------------------------- #
        
        def maxValue(gameState, agentIndex, depth, a, b):
            v = float("-inf")  
            legalMoves = gameState.getLegalActions(agentIndex)
            next = agentIndex + 1

            for move in legalMoves:
                NextState = gameState.getNextState(agentIndex, move)
                v = max(v , value(NextState, next,  depth, a, b))

                if v > b:
                    return v
                a = max(a, v)

            return v            
        # ------------------------------------------------------------------- #

        def minValue(gameState, agentIndex, depth, a, b):
            v = float("inf")  
            legalMoves = gameState.getLegalActions(agentIndex)
            next = agentIndex + 1

            for move in legalMoves:
                NextState = gameState.getNextState(agentIndex, move)
                if (gameState.getNumAgents() == next):
                    next = 0
                    depth += 1
                    v = min(v , value(NextState, next,  depth, a, b))
                else:
                    v = min(v , value(NextState, next,  depth, a, b))
                if v < a:
                    return v
                b = min(b, v)

            return v   
        # ------------------------------------------------------------------- #

        def value(gameState, agentIndex, depth, a, b):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0: 
                return maxValue(gameState, agentIndex, depth, a, b)
            else:
                return minValue(gameState, agentIndex, depth, a, b)
        # ------------------------------------------------------------------- #

        legalMoves = gameState.getLegalActions(0)
        actions = []   
        a = float("-inf") 
        b = float("inf") 

        for move in legalMoves:
            NextState = gameState.getNextState(0, move)
            actions.append((value(NextState, 1, 0, a, b), move))
            action = max(actions, key = lambda x : x[0])[1]
            bestValue = max(actions, key = lambda x : x[0])[0]
            a = max(a, bestValue)
            
        return action        

        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # ------------------------------------------------------------------- #
        def expectimax(gameState, agentIndex, depth):

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(agentIndex)
            next = agentIndex + 1
            values = []

            # if Pacman
            if agentIndex == 0: 

                for move in legalMoves:
                    NextState = gameState.getNextState(agentIndex, move)
                    values.append(expectimax(NextState, next,  depth))

                return max(values)                
            # if ghost   
            else: 

                for move in legalMoves:

                    NextState = gameState.getNextState(agentIndex, move)
                    if (gameState.getNumAgents() == next):
                        next = 0
                        depth += 1
                        values.append(expectimax(NextState, next,  depth))
                    else:
                        values.append(expectimax(NextState, next,  depth))

                total = sum(values)

                return total / len(legalMoves)
        # ------------------------------------------------------------------- #

        legalMoves = gameState.getLegalActions(0)
        actions = []   
        
        for move in legalMoves:
            NextState = gameState.getNextState(0, move)
            actions.append((expectimax(NextState, 1, 0), move))

        action = max(actions, key = lambda x : x[0])[1]
        return action
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()  
    foodlist = Food.asList()
    foodDist = []

    score = currentGameState.getScore()

    for food in foodlist:
        foodDist.append(manhattanDistance(Pos, food))

    # Get score of food
    if (len(foodDist) != 0):
        score += 10.0/min(foodDist)

    # Get score of ghost
    for ghost in GhostStates:

        ghostPos =  ghost.getPosition()
        ghostDist = manhattanDistance(Pos, ghostPos)
        # if you can't eat the ghost it is bad decrease score
        if ghost.scaredTimer == 0:
            score -= 10.0/(ghostDist + 1)
        # if you can eat the ghost it is good increase score
        else :
            score += 20.0/(ghostDist + 1)

    return score  
    
    # util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
