import xml.etree.ElementTree as ET
import xmlschema


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
