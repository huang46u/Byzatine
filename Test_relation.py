# %%
from ast import operator
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyparsing import srange
from torch import greater
import tools as tl
from preprocess import *
import relation_analyse as rel
import preprocess.Image_extract as ext
import imp
import os
from os import path
import json
imp.reload(ext)
imp.reload(tl)
imp.reload(rel)

relation_path = "../Zacos-genevre/Zacos-Geneve/rel/"
image_path = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0322_A.jpg.json"
labels_file = open("../Classes x annotation.txt", "r")
labels_dict = [label.rstrip("\n") for label in labels_file]
Image_dict = ext.Extract_image_all_part(image_path, labels_dict)
# %%
os.makedirs(relation_path,exist_ok=True)
# %%
ann_path = "../Zacos-genevre/Zacos-Geneve/ann/"
for file in os.listdir(ann_path):
    filename = os.path.join(ann_path, file)
    print(filename)
    Image_dict = ext.Extract_image_all_part(filename, labels_dict)
    image_dict, area_val, sorted_dict = rel.compute_area(Image_dict)
    if(len(image_dict) != 0):
        json_path = os.path.join(relation_path, file)
        data = rel.relation_analyse(Image_dict, image_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            with open(json_path, "w") as fp:
                json.dump(data, fp)
# %%
filename = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0537_B.jpg.json"
Image_dict = ext.Extract_image_all_part(filename, labels_dict)
label_dict, area_val, sorted_dict = rel.compute_area(Image_dict)
# %%
print(label_dict.keys())
if(len(label_dict) != 0):
    json_path = os.path.join(relation_path, "cdn_2004_0439_A.png.json")
    data = rel.relation_analyse(Image_dict, label_dict,sorted_dict)
    if(data != None and len(data["data"]) != 0):
        with open(json_path, "w") as fp:
            json.dump(data, fp)
# %%
for k,v in label_dict.items():
    plt.xlabel(k)
    plt.imshow(v)
    plt.show()