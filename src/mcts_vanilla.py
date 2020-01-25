# Credit to https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/ for providing pseudocode
from mcts_node import MCTSNode
from random import choice
from math import sqrt, log, inf

num_nodes = 100
explore_faction = 2.


# Calculate the upper confidence bound of a child node
def calc_ucb(node):
    # Pull every arm once (probably redundant)
    if node.visits == 0:
        return float('inf')

    exploit = float(node.wins / node.visits)
    
    explore = 0
    parent_log = float("-inf")
    if node.parent.visits != 0:
        parent_log = log(node.parent.visits)
        explore = explore_faction * sqrt(parent_log / node.visits)

    return exploit + explore



def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion are met.

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either '1' or '2'.

    Returns:        A node from which the next stage of the search can proceed.

    """
    # Tuple, leaf_node at 0, UCB at 1
    leaf_tuple = (None, None)

    # If untried actions exist, try one
    if node.untried_actions != []:
        child_node = expand_leaf(node, board, state)
        child_action = child_node.parent_action
        node.child_nodes[child_action] = child_node
        node.untried_actions.remove(child_action)
        leaf_tuple = (child_node, 0)

    # Find highest UCB child node
    else:
        for child_key in node.child_nodes.keys():
            child_node = node.child_nodes[child_key]
            child_ucb = calc_ucb(child_node)
            if (leaf_tuple is (None, None)) or (leaf_tuple[1] < child_ucb):
                leaf_tuple = (child_node, child_ucb)
        
    return leaf_tuple[0]


def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:    The added child node.

    """
    child_action = node.untried_actions[0]
    child_state = board.next_state(state, child_action)
    child_untried_actions = board.legal_actions(child_state)
    child_node = MCTSNode(parent=node, parent_action=child_action, action_list=child_untried_actions)
    return child_node


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    """
    if board.is_ended(state):
        return board.points_values(state)
    else:
        new_state = board.next_state(state, choice(board.legal_actions(state)))
        return rollout(board, new_state)


def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if node.parent is None:
        pass

    else:
        node.visits = node.visits + 1
        if won:
            node.wins = node.wins + 1
        backpropagate(node.parent, won)


def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    for step in range(num_nodes):
        # Copy the game for sampling a playthrough
        sampled_game = state

        # Start at root
        node = root_node

        # MCTS
        # Find leaf node
        leaf = traverse_nodes(node, board, sampled_game, identity_of_bot)
        # Move sampled_game to the state of leaf node
        simulation_result = rollout(board, sampled_game)
        won = False
        if simulation_result[identity_of_bot] == 1:
            won = True
        backpropagate(leaf, won)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_node = (None, None) # 0 is node, 1 is win rate
    for child_key in root_node.child_nodes:
        child_node = root_node.child_nodes[child_key]
        win_rate = child_node.wins / child_node.visits

        if best_node == (None, None) or win_rate > best_node[1]:
            best_node = (child_node, win_rate)

    return best_node[0].parent_action
