from cv2 import CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE
import numpy as np
import matplotlib.pyplot as plt
import plot
import math
from skimage.measure import block_reduce
from skimage.morphology import binary_closing,label,disk, binary_dilation
from scipy.optimize import fmin
import json
import os
import cv2
from sympy import comp

EPS = 1e-5

def unique_value(img):
    return np.unique(img.reshape(-1,img.shape[1]))[1:]

def membership(beta_min):
    return max(0,1-2*beta_min/math.pi)

def morpho_one_direction(object,direction ="right"):
    coord = np.argwhere(object>0)
    new_image = np.zeros(object.shape)
    if(direction == "right"): u_alpha = [1, 0]
    if(direction == "left"):  u_alpha = [-1,0]
    if(direction == "above"): u_alpha = [0, 1]
    if(direction == "below"): u_alpha = [0,-1]
    for i,j in np.argwhere(object>=0):
        if(object[i,j]!=0):
            new_image[i,j]=1
            continue
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

def find_freq(hist,bins, angle):
    for i in range(1,len(bins)):
        if(angle>=bins[i-1] and angle<=bins[i]):
            return hist[i-1]
    return 0

def compability_hist(hist,bins,direction= "right", step = 100):
    compability = []
    for u in np.arange(0.0,1.0,(1/float(step))):
        if(direction == "right"):
            cosine = np.sqrt(u)
            v1 = np.arccos(cosine)
            v2 = -np.arccos(cosine)
 
        elif(direction == "left"):
            cosine = np.sqrt(u)
            v1 = np.arccos(cosine)-math.pi
            v2 = -np.arccos(cosine)+math.pi
    
        elif(direction == "above"):
            cosine = np.sqrt(u)
            v1 = np.arccos(cosine)+0.5*math.pi
            v2 = -np.arccos(cosine)+0.5*math.pi
                
        elif(direction == "below"):
            cosine = np.sqrt(u)
            v1 = np.arccos(cosine)-0.5*math.pi
            v2 = -np.arccos(cosine)-0.5*math.pi
        
        freq1 = find_freq(hist,bins,v1)
        freq2 = find_freq(hist,bins,v2) 
        compability.append(max(freq1, freq2))
    
    return np.array(compability)    

def down_sample(object, scale=8):
    """
    Down sample image by scale
    Args:
        object: image to be down sample
        scale: down sample scale, default is 8
    Return:
        Down sampled image
    """
    object = block_reduce(object,block_size=(scale,scale))
    object = (object!=0)*object
    return object
  
def center_of_gravity(compability,step=100):
    if(np.sum(compability)!=0):
        return np.dot(np.arange(0.0, 1.0, 1/float(step)), compability)/np.sum(compability)
    return 0


def Histogram_Angle(image1,image2):
    coord1 = np.argwhere(image1>0)
    coord2 = np.argwhere(image2>0)
    angle = np.array([])
    #compute pairwise cosine for every pair of points in two objects
    for i in range(len(coord1)):
        vectors = coord2-coord1[i]
        ##transpose y and x 
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,[1,0])/(length+EPS)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
        #extends cosine field from [0, pi] to [-pi, pi], if the direction of vector
        # is point on negative y-axis, then the cosine is negative.
        angle = np.append(angle,np.arccos(cosine)*np.where(vectors[:,1]>0,1,-1))
    hist, bins = np.histogram(angle,bins = 180)
    return hist,bins

def compue_histogram_force(image1,image2,bin=180):
    coord1 = np.argwhere(image1>0)
    coord2 = np.argwhere(image2>0)
    hist = np.zeros(bin)
    step = 2*math.pi/bin
    bins = np.arange(-math.pi, math.pi+step, step)
    #compute pairwise cosine for every pair of points in two objects
    for i in range(len(coord1)):
        vectors = coord2-coord1[i]
        ##transpose y and x 
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,[1,0])/(length+1e-5)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
        #extends cosine field from [0, pi] to [-pi, pi], if the direction of vector
        # is point on negative y-axis, then the cosine is negative.
        weight = 1 /np.power(length+1e-5, 2) 
        weight = weight*(weight<1)
        angle = np.arccos(cosine)*np.where(vectors[:,1]>0,1,-1)
        for i in range(len(angle)):
            nb_bin = int((angle[i]+math.pi)//step)
            hist[nb_bin] += weight[i] 
    return hist, bins

def compute_compability(hist, bins, step= 100):
    cen_g_list = []
    compability_list = []
    for dir in ["left","right","above","below"]:
        compability = compability_hist(hist/np.max(hist),bins, direction = dir, step=step)
        compability_list.append(compability)
        cen_g_list.append(center_of_gravity(compability))
    return cen_g_list,compability_list

def demo_histogram_force(image1, image2, bin = 180, step = 100):
    plot.plot_two_image(image1,image2)
    hist, bins = compue_histogram_force(image1,image2,bin=bin)
    cen_g, compability = compute_compability(hist,bins,step)
    plot.plot_histogram_force(hist,bins)
    plot.plot_compability(step,cen_g,compability)
    
def bresenham_ray(image, point, theta):
    opposite = False
    steep = False
    if(theta>=math.pi): 
        theta -= math.pi
        opposite = True
    y,x = point
    w, h = image.shape
    height = 0
    coord = []
    if(theta<math.pi*0.5):
        if(theta>0.25*math.pi):
            steep = True
            theta = 0.5*math.pi - theta
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    if(not opposite): 
                        y -= 1
                    else:
                        y += 1
                    height -= 1
                if(not opposite): 
                    x += 1
                else:
                    x -= 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    if(not opposite): 
                        x += 1
                    else:
                        x -= 1
                    height -= 1
                if(not opposite): 
                    y -= 1
                else:
                    y += 1
    elif (theta == math.pi*0.5):
        if(not opposite):
            for i in range(0,y):
                coord.append((i,x))
        else:
            for i in range(y, w):
                coord.append((i,x))
    else:
        theta = theta-math.pi
        if(theta<-0.25*math.pi):
            theta = -(theta+0.5*math.pi)
            steep = True
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    if(not opposite): 
                        y -= 1
                    else:
                        y += 1
                    height -= 1
                if(not opposite): 
                    x -= 1
                else:
                    x += 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    if(not opposite): 
                        x += 1
                    else:
                        x -= 1
                    height -= 1
                if(not opposite): 
                    y += 1
                else:
                    y -= 1
    return np.array(coord)

def bresenham_line(image, point, theta):
    steep = False
    reverse = False
    if(theta>=math.pi): 
        theta -= math.pi
        reverse = True
    y,x = point
    w, h = image.shape
    height = 0
    coord = []
    if(theta<math.pi*0.5):
        if(theta>0.25*math.pi):
            steep = True
            theta = 0.5*math.pi - theta
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    y -= 1   
                    height -= 1        
                x += 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    y += 1
                    height -= 1
                x -= 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    x += 1
                    height -= 1
                y -= 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    x -= 1
                    height -= 1
                y += 1
    elif (theta == math.pi*0.5):
            for i in range(0, w):
                coord.append((i,x))
            
    else:
        theta = theta-math.pi
        if(theta<-0.25*math.pi):
            theta = -(theta+0.5*math.pi)
            steep = True
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    y -= 1
                    height -= 1
                x -= 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    y += 1
                    height -= 1
                x += 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    x += 1
                    height -= 1
                y += 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    x -= 1
                    height -= 1
                y -= 1
    coord = np.array(coord)
    if(reverse): coord = np.flip(coord,0)    
    return coord

def f(x):
    if(x>=0.5*math.pi): return (x-0.5*math.pi)/1.5*math.pi
    if(x<0.5*math.pi): return 0    
            
def mu_distance(dist, d1, d2):
    if(dist <= d1): return 1
    if(dist <= d2 and dist > d1): return (d2-dist)/(d2-d1)
    else: return 0

def get_intersect_coord(object1, coords):
    recording = False
    for coord in coords:
       if(object1[coord]==1):
           if(not recording):
                start = coord 
                recording = True
                intersect = True
       else:
           if(recording):
               end = coord
               recording = False
    return start, end, intersect

def point_surround_angle_only(object, point, bin = 120):
    """
    Compute the surround fuzziness with angular coverage criteria only
    
    Args:
        object: The region will be considered
        point: The coordinate where we perform the computation 
        bin: step size
    """
    intersect = []
    for i in np.arange(0,2,2/bin):
        coord = bresenham_ray(object,point,i*math.pi)
        if(object[coord[:,0],coord[:,1]].sum()==0):
            intersect.append(0)
        else: intersect.append(1)
    intersect = np.array(intersect)
    return f(intersect.sum()*(2/bin)*2*math.pi)

def point_surround_angle_and_distance(object, point, bin = 120):
    """
    Compute the surround fuzziness with angular coverage criteria and 
        distance criteria
    
    Args:
        object: The region will be considered
        point: The coordinate where we perform the computation 
        bin: step size
    """
    membership= 0
    for i in np.arange(0,2,2/bin):
        coord = bresenham_line(object,point,i*math.pi)
        intersect = False
        start, end, intersect= get_intersect_coord(object, coord)
        length = 0 
        if(intersect):
            x1,y1 = start
            x2,y2 = end
            length = math.sqrt((x2-x1)**2+(y2-y1)**2)
        membership += mu_distance(length)    
    return membership

def surround(object1, candidate_region, mode= "angle", bin = 120):
    coord = np.argwhere(candidate_region)
    membership_grade = np.zeros(object1.shape)
    for i in coord:
        if(mode == "angle"):
            membership_grade[i[0],i[1]]= point_surround_angle_only(
                object1,i,bin = bin)
        elif(mode == "distance"):
            membership_grade[i[0],i[1]] = point_surround_angle_and_distance(
                object1, i, bin = bin)
        else:
            raise("mode can only be 'anlge' or 'distance'")
    return membership_grade

def debug_surround(image,region,bin = 120):
    coords = np.argwhere(region)
    j=0
    for point in (coords):
        a = np.zeros(image.shape)
        intersect = []
        for i in np.arange(0,2,2/bin):
            coord = bresenham_ray(image,point,i*math.pi)
            if(image[coord[:,0],coord[:,1]].sum()==0):
                intersect.append(0)
            else: 
                a[coord[:,0],coord[:,1]]=np.max(image)
                intersect.append(1)
        intersect = np.array(intersect)
        if(j%100 ==0):
            plt.imshow(a+image)
            plt.show()
            print(intersect.sum()*(2/bin))
        j+=1
        
def debug_bresenham(image,region,bin = 120):
    coords = np.argwhere(region)
    j=0
    for point in (coords):
        a = np.zeros(image.shape)
        intersect = []
        for i in np.arange(0,1,1/bin):
            coord = bresenham_line(image,point,i*math.pi)
            if(image[coord[:,0],coord[:,1]].sum()==0):
                intersect.append(0)
            else: 
                a[coord[:,0],coord[:,1]]=np.max(image)
                intersect.append(1)
        intersect = np.array(intersect)
        if(j%100 ==0):
            plt.imshow(a+image)
            plt.show()
            print(intersect.sum()*(2/bin))
        j+=1
        
def parse_json(filename):
    """Parse json file 
    Args:
        filename: the json file name;
    Return:
        dictionnary
    """
    with open(filename, 'r') as fcc_file:
        dct = json.load(fcc_file)
        inv_dict = {int(v[0]): k for k, v in dct.items()}
        dct = {k : int(v[0]) for k,v in dct.items()}
        return dct,inv_dict
 
def read_images(dirPath):
    images = []
    for file in os.listdir(dirPath):
        if(os.path.isfile(os.path.join(dirPath, file))==True):
            base = os.path.basename(file)
            name = dirPath+'\\' +base
            img = cv2.imread(name)
            images.append(img)
    return np.array(images)  
        
def inside_body(object, body):
    """
    Test if objects are inside the body
    
    
    Args:
        object : object to test if it is insed the body
        body: image of person
    Return:
        insede: bool
        if the object is totally inside the body
    """
    coord = np.argwhere(object>0)
    return body[coord[:,0], coord[:,1]].sum()==len(coord)

def intersect_with_body(object, body):
    coord = np.argwhere(object>0)
    return body[coord[:,0], coord[:,1]].sum()>0
    
def symetry_image(image, alpha = 0.5*np.pi, d = 0):
    d_alpha = np.rad2deg(alpha)
    u = (np.cos(alpha), np.sin(alpha))
    w,h= image.shape
    image_flip = cv2.flip(image,1)
    if(d_alpha<90):
        rotation = -2*(90-d_alpha)
    else:
        rotation = 2*(d_alpha-90)
    m_rot = cv2.getRotationMatrix2D((w/2,h/2),rotation,1)
    im_rot = cv2.warpAffine(image_flip, m_rot, (w,h)) 
    m_trans = np.float32([[1,0,2*d*u[1]],[0,1,2*d*u[0]]])
    im_last = cv2.warpAffine(im_rot, m_trans,(w,h)) 
    return im_last
    
def symetry_measure(param, image):
    alpha, d = param
    im_smy = symetry_image(image,alpha,d)
    return np.linalg.norm(image-im_smy)/(2*np.linalg.norm(image))-1

def downhill_simplex(image,init):
    return fmin(symetry_measure, x0 = init, args=(image,),initial_simplex=None)
    
def symmetry_plot_data(image, param):
    alpha, d = param
    u = np.array([np.cos(alpha), np.sin(alpha)])
    w,h = image.shape
    origin = (d*u).astype(np.int32)+ np.array([h/2,w/2]).astype(np.int32)    
    line_coord = bresenham_line(image,origin,alpha)
    symm_image = symetry_image(image,alpha ,d)
    return symm_image, line_coord
    
    
    
    
    
    