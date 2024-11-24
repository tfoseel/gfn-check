import sys
import random
from collections import Counter
from bst import BinarySearchTree
from generators.state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn
from generators.RL import RLOracle
from generators.Random import RandomOracle
from generators.GFN_flow_matching import GFNOracle
from tqdm import tqdm

MAX_DEPTH = 4
VALUES = range(0, 11)
LEFT = [True, False]
RIGHT = [True, False]


def generate_tree(oracle, depth=0):
    num_nodes = 0
    value = oracle.select(1)
    tree = BinarySearchTree(value)
    num_nodes += 1
    if depth < MAX_DEPTH and oracle.select(2):
        tree.left, l_num_nodes = generate_tree(oracle, depth + 1)
        num_nodes += l_num_nodes
    if depth < MAX_DEPTH and oracle.select(3):
        tree.right, r_num_nodes = generate_tree(oracle, depth + 1)
        num_nodes += r_num_nodes
    return tree, num_nodes


def fuzz(oracle, unique_valid=1, valid=1, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
    valid_set = set()
    invalid_set = set()
    trials = 10000
    for i in tqdm(range(trials)):
        tqdm.write("=========")
        tqdm.write("{} trials, {} valids, {} unique valids, {} unique invalids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), len(invalid_set), len(valid_set) * 100 / valids if valids != 0 else 0))
        tree, num_nodes = generate_tree(oracle)
        tqdm.write("Tree with {} nodes".format(num_nodes))
        if tree.valid():
            tqdm.write("\033[0;32m" + tree.__repr__() + "\033[0m")
        else:
            tqdm.write("\033[0;31m" + tree.__repr__() + "\033[0m")
        
        if tree.valid():
            valids += 1
            if tree.__repr__() not in valid_set:
                oracle.compute_flow_matching_loss(validity=True, uniqueness=True)
                valid_set.add(tree.__repr__())
            else:
                oracle.compute_flow_matching_loss(validity=False, uniqueness=False)
        else:
            oracle.compute_flow_matching_loss(validity=False, uniqueness=False)

        tqdm.write("{} trials, {} valids, {} unique valids, {} unique invalids, {:.2f}% unique valids".format(
            trials, valids, len(valid_set), len(invalid_set), len(valid_set) * 100 / (valids + 1)), end='\r')
    tqdm.write("\ndone!", file=sys.stderr)


if __name__ == '__main__':
    # print("====Random====")
    # oracle_r = RandomOracle()
    # fuzz(oracle_r)
    # print("====RL: Sequence====")
    # oracle_s = RLOracle(sequence_ngram_fn(4), epsilon=0.25)
    # fuzz(oracle_s, unqiue_valid=20, valid=0, invalid=-1)
    # print("====RL: Tree====")
    # oracle_t = RLOracle(parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    # fuzz(oracle_t, unqiue_valid=20, valid=0, invalid=-1)
    # print("====RL: Tree L/R====")
    # oracle_lrt = RLOracle(
    #     left_right_parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    # fuzz(oracle_lrt, unqiue_valid=20, valid=0, invalid=-1)
    print("====GFN====")
    oracle_g = GFNOracle(
        128, 128, [(VALUES, 1), (LEFT, 2), (RIGHT, 3)], transformer=False)
    fuzz(oracle_g, unique_valid=1, valid=1, invalid=10e-20)
