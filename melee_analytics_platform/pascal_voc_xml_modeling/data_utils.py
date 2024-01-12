import xml.etree.ElementTree as ET
import os
from collections import Counter
import random
import pandas as pd
import json

def get_xml_dict(project_name, min_labels = 0):
    """
    Given a project name, return a dictionary of the xml file name keys and class name list values
    """
    xml_dict = {}
    for filename in os.listdir(f'projects/{project_name}/Annotations'):
        fullname = os.path.join(f'projects/{project_name}/Annotations', filename)
        tree = ET.parse(fullname).getroot()
        xml_dict[filename] = [object.find('name').text for object in tree.findall('object')]

    # get the counterooski
    class_counter = get_class_dict(xml_dict)

    # now lets find the classes that don't meet the criteria
    bads = set([label for label, count in class_counter.items() if count < min_labels])
    goods = set([label for label, count in class_counter.items() if count >= min_labels])
    print(f'Only using classes {goods} with min_labels = {min_labels}')
    # remove any images that have bads
    xml_fixed = {filename: classes for filename, classes in xml_dict.items() if len(set(classes).intersection(bads)) == 0}

    ###### TEMPORARY THING ########
    # remove anything that doesn't have clock
    xml_fixed = {filename: classes for filename, classes in xml_fixed.items() if 'Clock' in classes}
    ###############################
    
    print(f'Went from {len(xml_dict)} images to {len(xml_fixed)} after min_label filtered xml dict')

    return xml_fixed

def get_class_dict(xml_dict):
    """
    Given an XML dict, return a dictionary with keys as the class and counts as the values
    """
    return dict(Counter(x for xs in list(xml_dict.values()) for x in xs))

def get_boxes_labels(xml_path, classes):
    # read in the xml
    tree = ET.parse(xml_path).getroot()

    # set lists
    boxes = []
    labels = []

    # get all the objects
    for member in tree.findall('object'):
        labels.append(classes.index(member.find('name').text))
        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)
            
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes, labels

def train_test_split(project_name,
                     train_percent = .8,
                     overall_sample = 1,
                     min_labels = 0,
                     balance_classes = False):
    """
    Given a project name, split the total image set into train and test sets, making sure classes are balanced hopefully?
    """
    # first lets get our lil image dict
    xml_dict = get_xml_dict(project_name, min_labels = min_labels)

    # balance it somehow
    if balance_classes:
        # count it, then make an iteration thing
        class_counter = get_class_dict(xml_dict)

        # find lowest value
        low_val = min(list(class_counter.values()))

        # iterate and pray
        to_remove = []
        for label, count_of in class_counter.items():
            if count_of > (low_val*2):
                print(f'Balancing {label}: need to remove {count_of-(low_val*2)} labels')
                need_to_remove = count_of-(low_val*2)
                iterationy = [img for img, labels in xml_dict.items() if set(labels) == set([label])]
                while len(iterationy) > 0 and need_to_remove > 0:
                    rando_pick = random.choice(iterationy)
                    need_to_remove -= len(xml_dict[rando_pick])
                    to_remove.append(rando_pick)
                    iterationy.remove(rando_pick)
        for img in to_remove:
            xml_dict.pop(img)

        print('After trying to balance classes, heres the new class counter:')
        print(get_class_dict(xml_dict))

    # grabs the keys (xml file names)
    xml_names = list(xml_dict.keys())

    # sample as necessary
    if overall_sample < 1:
        overall_sample_size = round(overall_sample * len(xml_names))
        xml_names = random.sample(xml_names, overall_sample_size)

    # sample according to the splits
    sample_num = round(train_percent * len(xml_names))
    train_xmls = random.sample(xml_names, sample_num)
    test_xmls = list(set(xml_names) - set(train_xmls))

    # now make new dictionaries
    train_dict = {key[:-3] + 'jpg' : xml_dict[key] for key in train_xmls}
    test_dict = {key[:-3] + 'jpg' : xml_dict[key] for key in test_xmls}
   
    # check that they're kosher
    train_class_dict = get_class_dict(train_dict)
    test_class_dict = get_class_dict(test_dict)

    # need to mod them by a %
    classes = ['__background__'] + list(set(list(train_class_dict.keys()) + list(test_class_dict.keys())))
    train_total = sum(train_class_dict.values())
    test_total = sum(test_class_dict.values())
    train_percents = {key : round(num/train_total * 100, 3) for key,num in train_class_dict.items()} 
    test_percents = {key : round(num/test_total * 100, 3) for key,num in test_class_dict.items()} 
    dif_dict = {key: [abs(num-test_percents[key])] for key,num in train_percents.items()}
    dif_pd = pd.DataFrame(dif_dict).transpose().sort_values(0, ascending=False).reset_index().rename(columns = {'index': 'class', '0':'percent_diff'})
    print("Here's your percent difference between your train and test class distribution:")
    print(dif_pd.head(100))
    print()

    return list(train_dict.keys()), list(test_dict.keys()), classes

def write_json(path, load, file_name):
    """
    Given a file path and a load dictionary, write the dictionary out as a JSON at filepath
    """
    # first convert the load to a dumper
    dumper = json.dumps(load)

    # make a path just in case
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")

    # remove end '/' on path if it's there
    if path.endswith('/'):
        path = path[:-1]

    # and same for file_name
    if file_name.endswith('.json'):
        file_name = file_name[:-5]

    # and let it rip
    with open(f'{path}/{file_name}.json', "w") as json_file:
        json_file.write(dumper)

def read_json(path, file_name):
    # remove end '/' on path if it's there
    if path.endswith('/'):
        path = path[:-1]

    # and same for file_name
    if file_name.endswith('.json'):
        file_name = file_name[:-5]

    with open(f'{path}/{file_name}.json', 'r') as json_file:
        json_object = json.load(json_file)

    return json_object

def update_json(path, load, file_name):
    # first read in the json
    original_json = read_json(path, file_name)

    # iterate over new load and update the original
    for key,value in load.items():
        original_json[key] = value

    # and now overwrite
    write_json(path, original_json, file_name)