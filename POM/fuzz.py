import random
import sys
from collections import Counter
import json
import xml.etree.ElementTree as ET

from pom import POM
from generators.Random import RandomOracle

MAX_DEPTH = 6

with open('config.json', 'r') as file:
    config = json.load(file)
    tag_options = config["tags"]
    text_options = config["texts"]
    num_children_options = list(range(5))


def generate_pom(oracle):
    def _generate_pom(oracle, node, depth):
        # Tag part
        tag = None
        if node == None:
            tag = ET.Element('project')
            tag.set('xmlns', 'http://maven.apache.org/POM/4.0.0')
            tag.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
            tag.set('xsi:schemaLocation',
                    'http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd')
        else:
            tag_name = oracle.select(tag_options, 1)
            tag = ET.Element(tag_name)
        # Children part
        if depth < MAX_DEPTH:
            num_children = oracle.select(num_children_options, 2)
            if num_children > 0:
                for _ in range(num_children):
                    child = _generate_pom(oracle, tag, depth + 1)
                    if child != None:
                        tag.append(child)
            else:
                # Add text (can be also learned by oracle.select(text_options, 3))
                text = "Leaf text"
                tag.text = text
        else:
            # Add text (can be also learned by oracle.select(text_options, 3))
            text = "Leaf text"
            tag.text = text
        return tag

    pom = _generate_pom(oracle, None, 0)
    return POM(pom)


def fuzz(oracle, unqiue_valid=0, valid=0, invalid=0):
    valids = 0
    print("Starting!", file=sys.stderr)
    valid_set = set()
    trials = 100000
    for i in range(trials):
        print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), (len(valid_set) * 100 / valids) if valids != 0 else 0), end='\r')
        pom = generate_pom(oracle)

        if i % 1000 == 0:
            print(pom)

        if pom.valid():
            valids += 1
            if pom.__repr__() not in valid_set:
                valid_set.add(pom.__repr__())
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
