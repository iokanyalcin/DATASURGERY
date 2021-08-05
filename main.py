import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import helper


def main():
    data = helper.create_data_clusters()

    xml = data[1]["annot1"][0]

    tree = ET.parse(xml)
    i = tree.iter()
    
    root = tree.getroot()
    #print(ET.tostring(root))
    for child in root:
        print(child.tag, child.attrib)

        

    print(root[0][1].text)
if __name__ == "__main__":
    main()

