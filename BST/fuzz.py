import sys
import random
import argparse
from bst import BinarySearchTree
from generators.state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn
from generators.RL import RLOracle
from generators.Random import RandomOracle
from generators.GFN_trajectory_balance import GFNOracle_trajectory_balance
from generators.GFN_detailed_balance import GFNOracle_detailed_balance
from generators.GFN_local_search import GFNOracle_local_search
from generators.GFN_flow_matching import GFNOracle_flow_matching
from tqdm import tqdm


def generate_tree(oracle, depth=0, pruning=False):
    num_nodes = 0
    value = oracle.select(1)
    tree = BinarySearchTree(value)
    num_nodes += 1

    if pruning and not tree.valid():
        return tree, num_nodes, False

    if depth < MAX_DEPTH and oracle.select(2):
        tree.left, l_num_nodes, validity = generate_tree(oracle, depth + 1)
        num_nodes += l_num_nodes

        if pruning and not validity:
            return tree, num_nodes, False

    if depth < MAX_DEPTH and oracle.select(3):
        tree.right, r_num_nodes, validity = generate_tree(oracle, depth + 1)
        num_nodes += r_num_nodes

        if pruning and not validity:
            return tree, num_nodes, False

    return tree, num_nodes, tree.valid()


def fuzz(oracle, unique_valid=1, valid=1, invalid=0, local_search=False, local_search_steps=None):
    
    valids = 0
    print("Starting!", file=sys.stderr)
    valid_set = set()
    invalid_set = set()
    trials = 10000

    for i in tqdm(range(trials)):
        tqdm.write("=========")
        tqdm.write("{} trials, {} valids, {} unique valids, {} unique invalids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), len(invalid_set), len(valid_set) * 100 / valids if valids != 0 else 0), end='\r')
        tree, num_nodes, validity = generate_tree(oracle)
        if local_search:
            for i in range(local_search_steps):
                oracle.choice_sequence = oracle.choice_sequence[:len(oracle.choice_sequence)//2]
                depth = oracle.calculate_depth()
                new_tree, new_num_nodes, new_validity = generate_tree(oracle, depth)
                if validity and tree.__repr__() not in valid_set:
                    tree, num_nodes, validity = new_tree, new_num_nodes, new_validity
                    break
        tqdm.write("Tree with {} nodes".format(num_nodes))
        if validity:
            tqdm.write("\033[0;32m" + tree.__repr__() + "\033[0m")
            valids += 1
            if tree.__repr__() not in valid_set:
                valid_set.add(tree.__repr__())
                oracle.reward(unique_valid)
            else:
                oracle.reward(valid)
        else:
            tqdm.write("\033[0;31m" + tree.__repr__() + "\033[0m")
            if tree.__repr__() not in invalid_set:
                invalid_set.add(tree.__repr__())
            # oracle.reward(invalid)
        
    sizes = [valid_tree.count("(") for valid_tree in valid_set]
    tqdm.write("{} trials, {} valids, {} unique valids, {} unique invalids, {:.2f}% unique valids".format(
        trials, valids, len(valid_set), len(invalid_set), len(valid_set) * 100 / valids), end='\r')
    tqdm.write("\ndone!", file=sys.stderr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, dest="model", help="Experiment type. RL / FM / TB / DB / LS", default=5)
    parser.add_argument("--depth", type=int, dest="depth", help="Max depth of the tree", default=3)
    parser.add_argument("--value_range", type=int, dest="value_range", help="Range of values", default=3)
    parser.add_argument("--state_abstraction", type=str, dest="state_abstraction", help="State abstraction function", default="left_right_tree")

    args = parser.parse_args()

    MODEL = args.model
    STATE_ABSTRACTION = args.state_abstraction
    MAX_DEPTH = args.depth

    VALUES = range(1, args.value_range + 1)
    LEFT = [True, False]
    RIGHT = [True, False]

    domains = [(VALUES, 1), (LEFT, 2), (RIGHT, 3)]

    if MODEL == "RL":
        if STATE_ABSTRACTION == "random":
            print("====Random====")
            oracle_r = RandomOracle(domains)
            fuzz(oracle_r)
        
        elif STATE_ABSTRACTION == "sequence":
            print("====RL: Sequence====")
            oracle_s = RLOracle(sequence_ngram_fn(4), domains, epsilon=0.25)
            fuzz(oracle_s, unique_valid=20, valid=0, invalid=-1)

        elif STATE_ABSTRACTION == "tree":
            print("====RL: Tree====")
            oracle_p = RLOracle(parent_state_ngram_fn(4, MAX_DEPTH), domains, epsilon=0.25)
            fuzz(oracle_p, unique_valid=20, valid=0, invalid=-1)

        elif STATE_ABSTRACTION == "left_right_tree":
            print("====RL: Tree L/R====")
            oracle_lrt = RLOracle(left_right_parent_state_ngram_fn(4, MAX_DEPTH), domains, epsilon=0.25)
            fuzz(oracle_lrt, unique_valid=20, valid=0, invalid=-1)
        
        else:
            print("Invalid state abstraction function")
            exit(1)
    
    elif MODEL == "FM":
        oracle_g = GFNOracle_flow_matching(
            128, 128, domains)
        fuzz(oracle_g, unique_valid=1, valid=1, invalid=10e-20)

    elif MODEL == "TB":
        oracle_g = GFNOracle_trajectory_balance(
            128, 128, domains)
        fuzz(oracle_g, unique_valid=1, valid=1, invalid=10e-20)

    elif MODEL == "DB":
        oracle_g = GFNOracle_detailed_balance(
            128, 128, domains)
        fuzz(oracle_g, unique_valid=1, valid=1, invalid=10e-20)

    elif MODEL == "LS":
        oracle_g = GFNOracle_local_search(
            128, 128, domains)
        fuzz(oracle_g, unique_valid=1, valid=1, invalid=10e-20, local_search=True, local_search_steps=5)
    
    else:
        print("Invalid model")
        exit(1)
    