import sys
import random
from collections import Counter
from bst import BinarySearchTree
from BST.generators.state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn
from generators.RL import RLOracle
from generators.GFN import GFNOracle
from generators.Random import RandomOracle
import torch

MAX_DEPTH = 4

def generate_tree(oracle, depth=0):
    global logPf_traj
    value, log_pf = oracle.select(range(0, 11), 1)
    logPf_traj += log_pf
    tree = BinarySearchTree(value)
    if depth < MAX_DEPTH:
        value, log_pf = oracle.select([True, False], 2)
        logPf_traj += log_pf
        if value:
            tree.left = generate_tree(oracle, depth + 1)
    if depth < MAX_DEPTH:
        value, log_pf = oracle.select([True, False], 3)
        logPf_traj += log_pf
        if value:
            tree.right = generate_tree(oracle, depth + 1)
    return tree


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):
    global logPf_traj
    valids = 0
    print("Starting!", file=sys.stderr)
    valid_set = set()
    trials = 100000
    for i in range(trials):
        logPf_traj = 0
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
        trajectory_loss = torch.pow(logPf_traj + oracle.logZ - oracle.reward, 2)
        oracle.optimizer_logZ.zero_grad()
        ## for every learner of oracle
        for learner in oracle.learners.values():
            learner.optimizer.zero_grad()
        trajectory_loss.backward()
        oracle.optimizer_logZ.step()
        ## for every learner of oracle
        for learner in oracle.learners.values():
            learner.optimizer.step()
        

    sizes = [valid_tree.count("(") for valid_tree in valid_set]
    print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
        trials, valids, len(valid_set), len(valid_set) * 100 / valids), end='\r')
    print("\ndone!", file=sys.stderr)
    print(Counter(sizes))


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
    print ("GFN fuzzing")
    oracle = GFNOracle(parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    fuzz(oracle, unqiue_valid=20, valid=0, invalid=-1)
    

