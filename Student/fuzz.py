import random
import sys
from collections import Counter
import json
import xml.etree.ElementTree as ET
from student import Student
from generators.state_abstraction import parent_state_ngram_fn, index_parent_state_ngram_fn, sequence_ngram_fn
from generators.Random import RandomOracle
from generators.RL import RLOracle


MAX_DEPTH = 6

with open('config.json', 'r') as file:
    config = json.load(file)
    tag_options = config["tags"]
    text_options = config["texts"]
    num_attributes_options = list(range(5))
    boolean_options = [True, False]
    num_children_options = list(range(5))


def generate_student(oracle):
    def _generate_student(oracle, node, depth):
        # Tag part
        tag = None
        if node == None:
            tag = ET.Element('student')
        else:
            tag_name = oracle.select(tag_options, None)
            tag = ET.Element(tag_name)
        # Children part
        if depth < MAX_DEPTH:
            num_children = oracle.select(
                num_children_options[0 if node is not None else 1:], 2)
            if num_children > 0:
                for _ in range(num_children):
                    child = _generate_student(oracle, tag, depth + 1)
                    if child != None:
                        tag.append(child)
            else:
                # Add text
                text = random.choice(text_options)
                tag.text = text
        else:
            # Add text
            text = random.choice(text_options)
            tag.text = text
        return tag

    student = _generate_student(oracle, None, 0)
    return Student(student)


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):


MAX_DEPTH = 6

with open('config.json', 'r') as file:
    config = json.load(file)
    tag_options = config["tags"]
    text_options = config["texts"]
    num_attributes_options = list(range(5))
    boolean_options = [True, False]
    num_children_options = list(range(5))


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
    valid_set = set()
    trials = 10000
    for i in range(trials):
        print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), (len(valid_set) * 100 / valids) if valids != 0 else 0), end='\r')
        student = generate_student(oracle)

        if student.valid():
            valids += 1
            if student.__repr__() not in valid_set:
                valid_set.add(student.__repr__())
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
    print("====RL: Tree Index====")
    oracle_idxt = RLOracle(
        index_parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    fuzz(oracle_idxt, unqiue_valid=20, valid=0, invalid=-1)
