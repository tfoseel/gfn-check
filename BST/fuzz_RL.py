import sys
import random
from collections import Counter
from bst import BinarySearchTree
from generators.state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn
from generators.RL import RLOracle
from generators.Random import RandomOracle

MAX_DEPTH = 5


def generate_tree(oracle, depth=0, min_value=-float('inf'), max_value=float('inf')):
    """
    Generate a Binary Search Tree (BST) while ensuring all intermediate steps are valid.

    Parameters:
    - oracle: The oracle object for selecting values.
    - depth: Current depth of the tree.
    - min_value: Minimum allowable value for the current node.
    - max_value: Maximum allowable value for the current node.

    Returns:
    - A valid BST.
    """
    # Sample the root value within the valid range
    value = random.choice(range(int(min_value) + 1, int(max_value)))
    tree = BinarySearchTree(value)

    # Decide whether to generate left and right subtrees
    if depth < MAX_DEPTH and random.choice([True, False]):
        # For left child, restrict max value to current node's value
        tree.left = generate_tree(oracle, depth + 1, min_value, value)

    if depth < MAX_DEPTH and random.choice([True, False]):
        # For right child, restrict min value to current node's value
        tree.right = generate_tree(oracle, depth + 1, value, max_value)

    return tree


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
    sizes = list()
    valid_set = set()
    trials = 100000
    for i in range(trials):
        print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), (len(valid_set) * 100 / valids) if valids != 0 else 0), end='\r')
        tree = generate_tree(oracle)
        if tree.valid():
            valids += 1
            if tree.__repr__() not in valid_set:
                valid_set.add(tree.__repr__())
                sizes.append(tree.depth())
                oracle.reward(unqiue_valid)
            else:
                oracle.reward(valid)
        else:
            oracle.reward(invalid)
    print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
        trials, valids, len(valid_set), len(valid_set) * 100 / valids), end='\r')
    print("\ndone!", file=sys.stderr)
    print(Counter(sizes))


if __name__ == '__main__':
    print("====Random====")
    oracle_r = RandomOracle()
    fuzz(oracle_r)
    print("====RL: Sequence====")
    oracle_s = RLOracle(sequence_ngram_fn(4), epsilon=0.25)
    fuzz(oracle_s, unqiue_valid=20, valid=0, invalid=-1)
    print("====RL: Tree====")
    oracle_t = RLOracle(parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    fuzz(oracle_t, unqiue_valid=20, valid=0, invalid=-1)
    print("====RL: Tree L/R====")
    oracle_lrt = RLOracle(
        left_right_parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    fuzz(oracle_lrt, unqiue_valid=20, valid=0, invalid=-1)
