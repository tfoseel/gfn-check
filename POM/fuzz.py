import random
import sys
from collections import Counter
import json
import xml.etree.ElementTree as ET
from pom import POM
from generators.state_abstraction import parent_state_ngram_fn, index_parent_state_ngram_fn, sequence_ngram_fn
from generators.Random import RandomOracle
from generators.RL import RLOracle

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
            # Base case: most simple form of valid pom.xml
            tag = ET.Element('project', {
                'xmlns': 'http://maven.apache.org/POM/4.0.0'
            })
            base_children = [("modelVersion", "4.0.0"), ("groupId", "com.example"),
                             ("artifactId", "my-app"), ("version", "1.0.0")]
            for t, s in base_children:
                child = ET.SubElement(tag, t)
                child.text = s
        else:
            tag_name = oracle.select(tag_options, 1)
            tag = ET.Element(tag_name)
        # Children part
        if depth < MAX_DEPTH:
            # No empty tree with just one leaf.
            num_children = oracle.select(
                num_children_options[0 if node is not None else 1:], 2)
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
    trials = 1000
    for i in range(trials):
        print("{} trials, {} valids, {} unique valids, {:.2f}% unique valids".format(
            i, valids, len(valid_set), (len(valid_set) * 100 / valids) if valids != 0 else 0), end='\r')
        pom = generate_pom(oracle)

        if pom.valid():
            valids += 1
            if pom.__repr__() not in valid_set:
                valid_set.add(pom.xml)
                oracle.reward(unqiue_valid)
            else:
                oracle.reward(valid)
        else:
            oracle.reward(invalid)
    # Define "sizes" other way to show the features of the tree
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
