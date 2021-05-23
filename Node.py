from math import sqrt
import numpy as np


def ucb_score(parent, child):
    """
    The UCB score for an action that would transition between the parent and child
    Encourages exploration for less-visited nodes
    """
    prior_score = child.prior * sqrt(parent.visit_count) / (child.visit_count + 1)
    value_score = -child.value() if child.visit_count > 0 else 0
    return value_score + prior_score


class Node:
    __slots__ = ['prior', 'to_play', 'visit_count', 'value_sum', 'children', 'ENV', "Node_v"]

    def __init__(self, prior, to_play):
        self.to_play = to_play
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.ENV = None
        self.Node_v = np.vectorize(Node)

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self):
        """
        Select the action with the highest value score.
        """
        best_value, best_action = -np.inf, -1

        for action, child in self.children.items():
            score = child.value()
            if score > best_value:
                best_value, best_action = score, action

        return best_action, best_value

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score, best_action, best_child = -np.inf, -1, None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score, best_action, best_child = score, action, child

        return best_action, best_child

    def expand(self, env, to_play, action_probs):
        self.to_play = to_play
        self.ENV = env

        action_probs = np.reshape(action_probs, 4096)
        keys = np.nonzero(action_probs)[0]
        nodes = self.Node_v(action_probs.take(keys), self.to_play * -1)
        self.children.update(zip(keys, list(nodes)))

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "Game State: {} Prior: {} Count: {} Value: {}".format(self.ENV, prior, self.visit_count, self.value())
