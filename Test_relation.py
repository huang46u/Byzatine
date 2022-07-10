# %%
from more_itertools import split_into
from sqlalchemy import asc
import tools as tl
import relation.relation_analyse as rel
from skimage.morphology import convex_hull_image
import relation.surround as su
import imp
import os
import json
import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import file_processing.Image_extract as ext
imp.reload(ext)
imp.reload(tl)
imp.reload(rel)

ann_path = "../Zacos-genevre/Zacos-Geneve/ann/"
text_path = "../Zacos-genevre/Zacos-Geneve/text/"
relation_path = "../Zacos-genevre/Zacos-Geneve/rel/"
os.makedirs(relation_path,exist_ok=True)
os.makedirs(text_path,exist_ok=True)
json_path = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0322_A.jpg.json"
labels_file = open("../Classes x annotation.txt", "r")
labels_dict = [label.rstrip("\n") for label in labels_file]

# %%
for file in os.listdir(ann_path):
    filename = os.path.join(ann_path, file)
    print(filename)
    Image_dict = ext.Extract_image_mask(filename, 
                                        category=True, 
                                        filter=True, 
                                        L_dict = labels_dict)
    image_dict, label_dict, sorted_dict = rel.compute_area(Image_dict)
    if(len(image_dict) != 0):
        data = rel.relation_analyse(image_dict, label_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            split_name = file.split("_")
            suffix = split_name[-1].split(".")
            json_new_name = "CdN_" + split_name[1]+"-"+split_name[2] +"_"+ suffix[0] + ".json"
            text_new_name = "CdN_" + split_name[1]+"-"+split_name[2] +"_"+ suffix[0] + ".txt"
            json_path = os.path.join(relation_path, json_new_name)
            text_filename = os.path.join(text_path, text_new_name)
            rel.relation_extract_to_text(data, text_filename)
            with open(json_path, "w") as fp:
                json.dump(data, fp)

# %%
filename = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0313_A.jpg.json"
file = "cdn_2004_0313_A.jpg.json"
Image_dict = ext.Extract_image_mask(filename, 
                                        category=True, 
                                        filter=True, 
                                        L_dict = labels_dict)
image_dict, label_dict, sorted_dict = rel.compute_area(Image_dict)
if(len(image_dict) != 0):
        json_path = os.path.join(relation_path, file)
        data = rel.relation_analyse(image_dict, label_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            data['data'].sort(key = lambda k : k['rel'])
            with open(json_path, "w") as fp:
                json.dump(data, fp) 

# %%
df_list1 = []
df_list2 = []
path_name = []
rel_path1 = "../Zacos-genevre/Zacos-Geneve/rel/"
rel_path2 = "../06-29 JSON Victoria"
# %%
for file in os.listdir(rel_path1):
    path_name.append(file)
    filename = os.path.join(rel_path1, file)
    print(filename)
    df1 = pd.read_json(filename)
    df1 = pd.json_normalize(df1["data"])
    df1 = df1.sort_values('rel',ignore_index=True)
    df1 = df1.reindex(columns = ["obj1", "obj2", "rel"])
    df_list1.append(df1)
    filename = os.path.join(rel_path2, file)
    df2 = pd.read_json(filename)
    df2 = pd.json_normalize(df2["data"])
    df2 = df2.reindex(columns = ["obj1", "obj2", "rel"])
    df2 = df2.sort_values('rel',ignore_index=True)
    df2.columns = ['obj1_gt', 'obj2_gt', 'rel_gt']
    df_list2.append(df2)
# %%
df =  pd.concat([df_list1[0], df_list2[0]],axis = 1,sort= True)
df.insert(0,"filename", path_name[0])
df= df.set_index('filename', append=True).swaplevel(0,1)
for i in range(1,len(df_list1)):
    df_tmp = pd.concat([df_list1[i], df_list2[i]], axis = 1,sort= False )
    df_tmp["filename"] = path_name[i]
    df_tmp = df_tmp.set_index('filename', append=True).swaplevel(0,1)
    df = pd.concat([df, df_tmp],axis = 0,sort= False) 
df.to_excel("../Zacos-genevre/Zacos-Geneve/output.xlsx")   
# %%
#Tatish
ann_path = "../Tatish/Tatis/ann/"
text_path = "../Tatish/Tatis/text/"
relation_path = "../Tatish/Tatis/rel/"
os.makedirs(relation_path,exist_ok=True)
os.makedirs(text_path,exist_ok=True)
# %%
for file in os.listdir(ann_path):
    filename = os.path.join(ann_path, file)
    print(filename)
    Image_dict = ext.Extract_image_mask(filename, 
                                        category=True, 
                                        filter=True, 
                                        L_dict = labels_dict)
    image_dict, label_dict, sorted_dict = rel.compute_area(Image_dict)
    if(len(image_dict) != 0):
        data = rel.relation_analyse(image_dict, label_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            split_name = file.split("_")
            suffix = split_name[-1].split(".")
            json_new_name =  split_name[0]+"-"+suffix[0] + ".json"
            text_new_name = split_name[0]+"-"+suffix[0] + ".txt"
            json_path = os.path.join(relation_path, json_new_name)
            text_filename = os.path.join(text_path, text_new_name)
            rel.relation_extract_to_text(data, text_filename)
            with open(json_path, "w") as fp:
                json.dump(data, fp)

# %%
filename = "../Tatish/Tatis/ann/Tatish 2301_A.jpg.json"

Image_dict = ext.Extract_image_mask(filename, 
                                        category=False, 
                                        filter=True, 
                                        L_dict = labels_dict)
Image_dict
# %%
image_dict, label_dict, sorted_dict = rel.compute_area(Image_dict)
if(len(image_dict) != 0):
        json_path = os.path.join(relation_path, file)
        data = rel.relation_analyse(image_dict, label_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            data['data'].sort(key = lambda k : k['rel'])
            with open(json_path, "w") as fp:
                json.dump(data, fp) 