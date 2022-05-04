import numpy as np
import math
import plot 
EPS = 1e-5
def compute_histogram_angle(image1,image2, n_bin = 180):
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
    hist, bins = np.histogram(angle,bins = n_bin)
    return hist,bins

def compue_histogram_force(image1,image2,n_bin=180):
    coord1 = np.argwhere(image1>0)
    coord2 = np.argwhere(image2>0)
    hist = np.zeros(n_bin)
    step = 2*math.pi/n_bin
    bins = np.arange(-math.pi, math.pi+step, step)
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
        weight = 1 /np.power(length+1e-5, 2) 
        weight = weight*(weight<1)
        angle = np.arccos(cosine)*np.where(vectors[:,1]>0,1,-1)
        for i in range(len(angle)):
            nb_bin = int((angle[i]+math.pi)//step)
            hist[nb_bin] += weight[i] 
    return hist, bins

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

def center_of_gravity(compability,step=100):
    if(np.sum(compability)!=0):
        return np.dot(np.arange(0.0, 1.0, 1/float(step)), compability)/np.sum(compability)
    return 0

def compute_compability(hist, bins, step= 100):
    cen_g_list = []
    compability_list = []
    for dir in ["left","right","above","below"]:
        compability = compability_hist(hist/np.max(hist),bins, direction = dir, step=step)
        compability_list.append(compability)
        cen_g_list.append(center_of_gravity(compability, step = step))
    return cen_g_list,compability_list

def demo_histogram_force(image1, image2, bin = 180, step = 100):
    plot.plot_two_image(image1,image2)
    hist, bins = compue_histogram_force(image1,image2,n_bin=bin)
    cen_g, compability = compute_compability(hist,bins,step)
    plot.plot_histogram(hist,bins)
    plot.plot_compability(step,cen_g,compability)

def demo_histogram_angle(image1, image2, bin = 180, step = 180):
    plot.plot_two_image(image1,image2)
    hist, bins = compute_histogram_angle(image1,image2,n_bin=bin)
    cen_g, compability = compute_compability(hist,bins,step)
    plot.plot_histogram(hist,bins)
    plot.plot_compability(step,cen_g,compability)