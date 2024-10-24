import random
import sys
from collections import Counter
import json
import xml.etree.ElementTree as ET

from student import Student
from generators.Random import RandomOracle

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
        num_children = oracle.select(num_children_options)
        if depth < MAX_DEPTH and num_children > 0:
            for _ in range(num_children):
                child = _generate_student(oracle, tag, depth + 1)
                if child != None:
                    tag.append(child)
        else:
            # Add text
            text = oracle.select(text_options)
            tag.text = text
        return tag
    student = _generate_student(oracle, None, 0)
    return Student(student)


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
    valid_set = set()
    trials = 100000
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
