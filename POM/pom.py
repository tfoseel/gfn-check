import xml.etree.ElementTree as ET
import xmlschema


class POM:
    def __init__(self, xml):
        self.schema = xmlschema.XMLSchema("maven-4.0.0.xsd")
        self.xml = ET.tostring(xml, encoding='utf-8',
                               xml_declaration=True).decode('utf-8')

    def valid(self):
        try:
            self.schema.validate(self.xml)
            return True
        except:
            return False

    def indent(self, elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
