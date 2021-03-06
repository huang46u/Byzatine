import numpy as np
import imp
import histogram as his
from scipy.ndimage import convolve1d
import tools as tl
import math
imp.reload(his)


def fuzzy_histogram_dilation(image, hist):
    coord = np.argwhere(image > 0)
    dilation_image = np.zeros(image.shape)
    step = 2*math.pi / hist.size
    for i, j in np.argwhere(image == 0):
        vectors = [i, j] - coord
        # transpose y and x
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors, axis=1)
        cosine = np.dot(vectors, [1, 0])/(length+1e-5)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine >= -1, cosine <= 1))]
        angle = np.arccos(cosine)*np.where(vectors[:, 1] < 0, 1, -1)
        if(len(angle) > 0):
            nb_bin = (angle+math.pi)//step
            nb_bin = nb_bin.astype(np.uint8)
            degrad = hist[nb_bin]
            dilation_image[i, j] = np.max(degrad)
    return dilation_image


def visualise_structuring_element(hist1, hist2, size=500):
    step = 2*math.pi/hist1.size
    struct_element1 = np.zeros((size, size))
    struct_element2 = np.zeros((size, size))
    for i, j in np.argwhere(struct_element1 == 0):
        x = i-size/2
        y = j-size/2
        length = math.sqrt(x**2+y**2)
        # filter the cosine value
        cosine = np.dot(np.array([y, x]), [1, 0])/(length+1e-5)
        if(cosine < -1 or cosine > 1):
            continue
        angle = np.arccos(cosine)*np.where(x < 0, 1, -1)
        nb_bin = int((angle+math.pi)//step)
        struct_element1[i, j] = hist1[nb_bin-1]
        struct_element2[i, j] = hist2[nb_bin-1]
    return struct_element1, struct_element2


def between(image1, image2, kernel_size=1, bin=180):
    hist1, bin1 = his.compute_histogram_angle(image1, image2, n_bin=bin)
    hist2, bin2 = his.compute_histogram_angle(image2, image1, n_bin=bin)
    
    kernel = np.ones(kernel_size)/kernel_size
    hist1 = convolve1d(hist1, kernel, mode ="wrap")
    hist2 = convolve1d(hist2, kernel, mode ="wrap")
    hist1 = hist1/np.max(hist1)
    hist2 = hist2/np.max(hist2)
    dilation_image1 = fuzzy_histogram_dilation(image1, hist1)
    dilation_image2 = fuzzy_histogram_dilation(image2, hist2)
    inter = np.minimum(dilation_image1, dilation_image2)
    return inter, dilation_image1, dilation_image2, hist1, hist2, bin1, bin2

def eval_between(object, inter, image1, image2):
    chull = tl.convex_hull(image1, image2)
    print(inter[inter>0])
    coord = np.argwhere(object)
    x,y = coord[:,0], coord[:,1]
    convex_inter = np.minimum(chull, inter)
    object = np.minimum(convex_inter, object)
    ness = np.max(object[x,y])
    poss = np.min(object[x,y])
    mean = np.mean(object[x,y])
    return object, convex_inter, ness, poss, mean