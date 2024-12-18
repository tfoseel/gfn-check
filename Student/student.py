import xml.etree.ElementTree as ET
import xmlschema
import random
import json

MAX_DEPTH = 4

with open('config.json', 'r') as file:
    config = json.load(file)
    tag_options = config["tags"]
    text_options = config["texts"]
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

class Student:
    def __init__(self, xml):
        self.schema = xmlschema.XMLSchema("student.xsd")
        self.xml = xml

    def valid(self):
        try:
            self.schema.validate(self.xml)
            return True
        except:
            return False

    def indent(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def __repr__(self):
        self.indent(self.xml)
        return ET.tostring(self.xml, encoding='unicode')
