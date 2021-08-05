import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

def create_data_clusters():
    #Returns a dict which includes all the coresponding image and xml files clustered inside of it.
    
    img_paths = glob.glob('data/thyroid/125_[1-2-3-4-5].jpg')
    annotation_paths = glob.glob('data/thyroid/*.xml')

    data_clusters = {}
    for i in range(1,401):
        data_clusters[i] = {"img{}".format(i): glob.glob('data/thyroid/{}_[1-2-3-4-5].jpg'.format(i)),
                "annot{}".format(i):glob.glob('data/thyroid/{}.xml'.format(i)) }
    return data_clusters


def parse_xml(xml_file):
    """
    Returns tiraid type of given xml file.s
    """
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root[7].text
