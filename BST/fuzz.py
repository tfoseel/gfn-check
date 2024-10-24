import sys
import random
from collections import Counter
from bst import BinarySearchTree
from state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn
from generators.RL import RLOracle
from generators.Random import RandomOracle


MAX_DEPTH = 4


def generate_tree(oracle, depth=0):
    value = oracle.select(range(0, 11), 1)
    tree = BinarySearchTree(value)
    if depth < MAX_DEPTH and oracle.select([True, False], 2):
        tree.left = generate_tree(oracle, depth + 1)
    if depth < MAX_DEPTH and oracle.select([True, False], 3):
        tree.right = generate_tree(oracle, depth + 1)
    return tree


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
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
                oracle.reward(unqiue_valid)
            else:
                oracle.reward(valid)
        else:
            oracle.reward(invalid)
    sizes = [valid_tree.count("(") for valid_tree in valid_set]
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
