import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

def create_data_clusters():
    img_paths = glob.glob('data/thyroid/125_[1-2].jpg')
    annotation_paths = glob.glob('data/thyroid/*.xml')

    data_clusters = {}
    for i in range(1,401):
        data_clusters[i] = {"img{}".format(i): glob.glob('data/thyroid/{}_[1-2-3-4-5].jpg'.format(i)),
                "annot{}".format(i):glob.glob('data/thyroid/{}.xml'.format(i)) }
    return data_clusters


def parse_xml(xml_file):
    """
    Returns tiraid type
    """
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root[7].text



#TODO: 
# Image labelling 


#####
#TODO: xml file parse tirad classifications
#tree = ET.parse(test_anot)
#root = tree.getroot()
#root.text

# 
#print(root[7].text) #root 7 triad type

#for idx, data in data.items():
#    try: 
#        xml_file_path = data["annot"+str(idx)][0]
#        print(parse_xml(xml_file_path))
#    except IndexError:
#        continue