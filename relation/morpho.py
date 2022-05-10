import numpy as np
import math
EPS = 1e-5
def membership(beta_min):
    return max(0,1-2*beta_min/math.pi)

def morpho_one_direction(object,direction ="right"):
    coord = np.argwhere(object>0)
    new_image = np.zeros(object.shape)
    if(direction == "right"): u_alpha = [1, 0]
    if(direction == "left"):  u_alpha = [-1,0]
    if(direction == "above"): u_alpha = [0, 1]
    if(direction == "below"): u_alpha = [0,-1]
    for i,j in np.argwhere(object==0):
        vectors = [i,j]-coord
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,u_alpha)/(length+EPS)
        cosine = cosine[np.where(np.logical_and(cosine>=0,cosine<=1))]
        if(len(cosine)>0):
            beta_min = np.min(np.arccos(cosine))
            new_image[i,j] = membership(beta_min)
    return new_image

def morpho_relation(object1, object2):
    morpho_ref_list = []
    morpho_res_list = []
    nece_deg_list = [] #necessity degree
    poss_deg_list = [] #possibility degree
    mean_deg_list = [] #mean membership grade
    for dir in ["right", "left", "above", "below"]:
        morpho_img = morpho_one_direction(object1, direction = dir)
        new_image = np.zeros(morpho_img.shape)
        coord = np.argwhere(object2>0)
        #exclude the intersection part
        coord = coord[np.where(object1[coord[:,0],coord[:,1]]==0)]
        new_image[coord[:,0],coord[:,1]] = morpho_img[coord[:,0], coord[:,1]]
        morpho_value = morpho_img[coord[:,0], coord[:,1]]
        nece_deg_list.append(np.max(morpho_value))
        poss_deg_list.append(np.min(morpho_value))
        mean_deg_list.append(np.mean(morpho_value))
        morpho_ref_list.append(morpho_img)
        morpho_res_list.append(new_image)
    return morpho_ref_list, morpho_res_list, nece_deg_list, poss_deg_list, mean_deg_list



