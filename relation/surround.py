import math 
import numpy as np
import tools as tl
def f(x):
    """
    Membership function of angle criteria
    
    Parameters:
    -----------
    x : float
        Radian covers    
    """
    if(x>=0.5*math.pi): return (x-0.5*math.pi)/(1.5*math.pi)
    if(x<0.5*math.pi): return 0    
            
def mu_distance(dist, d1, d2):
    if(dist <= d1): return 1
    if(dist <= d2 and dist > d1): return (d2-dist)/(d2-d1)
    else: return 0

def get_intersect_coord(object,line):
    if(object[line].sum()==0): return False, (-1,-1)
    else:
        cut= object[line] != 0
        cut_ends = np.where(np.diff(cut) == True)
        rr,cc = line
        return True, (rr[cut_ends[0],cc[cut_ends[0]]]) 
    
def point_surround_angle_only(object, point, n_dir = 120):
    """
    Compute the surround fuzziness with angular coverage criteria only
    
    Args:
        object: The region will be considered
        point: The coordinate where we perform the computation 
        n_dir: step size
    """
    intersect = np.zeros(n_dir)
    angles = np.arange(0,2,2/n_dir)
    for i in range(n_dir):
        coord = tl.bresenham_ray(object,point,angles[i]*math.pi)
        intersect[i]=(object[coord].sum()!=0)
    percent = intersect.sum()/n_dir
    return f(percent*2*math.pi)

def point_surround_angle_and_distance(object, point, n_dir = 120):
    """
    Compute the surround fuzziness with angular coverage criteria and 
        distance criteria
    
    Args:
        object: The region will be considered
        point: The coordinate where we perform the computation 
        bin: step size
    """
    membership= 0
    for i in np.arange(0,2,2/n_dir):
        line = tl.bresenham_ray(object,point,i*math.pi)
        intersect, inter_coord= get_intersect_coord(object, line)
        length = 0 
        if(intersect):
            x1,y1 = point
            x2,y2 = inter_coord
            length = math.sqrt((x2-x1)**2+(y2-y1)**2)
        membership += mu_distance(length)    
    return membership

def surround(object, candidate_region, mode= "angle", n_dir = 120):
    coord = np.argwhere(candidate_region)
    membership_grade = np.zeros(object.shape)
    for i in coord:
        if(mode == "angle"):
            membership_grade[i[0],i[1]]= point_surround_angle_only(
                object,i,n_dir = n_dir)
        elif(mode == "distance"):
            membership_grade[i[0],i[1]] = point_surround_angle_and_distance(
                object, i, bin = n_dir)
        else:
            raise("mode can only be 'anlge' or 'distance'")
    return membership_grade
