# %%
from more_itertools import split_into
from sqlalchemy import asc
import tools as tl
import relation.relation_analyse as rel

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

relation_path = "../Zacos-genevre/Zacos-Geneve/rel/"
json_path = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0322_A.jpg.json"
labels_file = open("../Classes x annotation.txt", "r")
labels_dict = [label.rstrip("\n") for label in labels_file]
#%%
labels_dict
# %%
os.makedirs(relation_path,exist_ok=True)
# %%
ann_path = "../Zacos-genevre/Zacos-Geneve/ann/"
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
            #data['data'].sort(key = lambda k : k['rel'])
            split_name = file.split("_")
            surfix = split_name[-1].split(".")
            new_name = "CdN_" + split_name[1]+"-"+split_name[2] +"_"+ surfix[0] + ".json"
            json_path = os.path.join(relation_path, new_name)
            with open(json_path, "w") as fp:
                json.dump(data, fp)

# %%
filename = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0319_A.jpg.json"
file = "cdn_2004_0319_A.jpg.json"
Image_dict = ext.Extract_image_mask(filename, 
                                        category=True, 
                                        filter=True, 
                                        L_dict = labels_dict)
image_dict, label_dict, sorted_dict = rel.compute_area(Image_dict)
if(len(image_dict) != 0):
        json_path = os.path.join(relation_path, file)
        #data = rel.relation_analyse(image_dict, label_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            data['data'].sort(key = lambda k : k['rel'])
            with open(json_path, "w") as fp:
                json.dump(data, fp) 
# %%
data['data']
# %%
data['data'].sort(key = lambda k : k['rel'])
data
# %%
relation_path = "../Tatish/Tatis/rel/"
os.makedirs(relation_path,exist_ok=True)
filename = "../Tatish/Tatis/ann/Tatish 1754_A.jpg.json"
file = "Tatish 1754_A.jpg.json"
Image_dict = ext.Extract_image_mask(filename, 
                                        category=True, 
                                        filter=True, 
                                        L_dict = labels_dict)

# %%
image_dict, label_dict, sorted_dict = rel.compute_area(Image_dict)
if(len(image_dict) != 0):
        json_path = os.path.join(relation_path, file)
        #data = rel.relation_analyse(image_dict, label_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            with open(json_path, "w") as fp:
                json.dump(data, fp)
# %%
image_dict
# %%
body = image_dict['Person_Body']
st_george = image_dict['Person_St_George']
# %%
import matplotlib.pyplot as plt
plot.plot_two_image(body, st_george,"Person_Body", "Person_St_George")

# %%
data
df = pd.json_normalize(data, record_path="data", errors = "ignored")
df = df.reindex(columns = ["obj1", "obj2", "rel"])
df
# %%
a = df["rel"].value_counts()
a["Rel_Center"]
# %%
df1 = pd.read_json("../Zacos-genevre/Zacos-Geneve/rel/cdn_2004_0319_A.jpg.json")
df1 = pd.json_normalize(df1["data"])
df1 = df1.reindex(columns = ["obj1", "obj2", "rel"])
df2 = pd.read_json("../Zacos-genevre/Zacos-Geneve/rel/cdn_2004_0355_A.jpg.json")
df2 = pd.json_normalize(df2["data"])
df2 = df2.reindex(columns = ["obj1", "obj2", "rel"])
# %%
df2
# %%
df3 = pd.concat([df1,df2],axis = 1)

# %%
df3.to_excel("output.xlsx")
# %%
df2.loc[df2['rel'] == "Rel_Left"]
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
    df1 = pd.read_json(filename)
    df1 = pd.json_normalize(df1["data"])
    df1 = df1.sort_values('rel',ignore_index=True)
    df1 = df1.reindex(columns = ["obj1", "obj2", "rel"])
    
    print(df1)
    df_list1.append(df1)
    filename = os.path.join(rel_path2, file)
    df2 = pd.read_json(filename)
    df2 = pd.json_normalize(df2["data"])
    df2 = df2.reindex(columns = ["obj1", "obj2", "rel"])
    df2 = df2.sort_values('rel',ignore_index=True)
    
    
    df2.columns = ['obj1_gt', 'obj2_gt', 'rel_gt']
    print(df2)
    df_list2.append(df2)
# %%
df_list1[0]
# %%
df_list2[0]
# %%
df =  pd.concat([df_list1[0], df_list2[0]],axis = 1,sort= True)
df.insert(0,"filename", path_name[0])
df= df.set_index('filename', append=True).swaplevel(0,1)
df
# %%
for i in range(1,len(df_list1)):
    df_tmp = pd.concat([df_list1[i], df_list2[i]], axis = 1,sort= False )
    df_tmp["filename"] = path_name[i]
    df_tmp = df_tmp.set_index('filename', append=True).swaplevel(0,1)
    df = pd.concat([df, df_tmp],axis = 0,sort= False)
df    
# %%
df.to_excel("output.xlsx")
# %%
