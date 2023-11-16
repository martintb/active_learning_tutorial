from itertools import product
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def ternary_to_xy(comps,normalize=True):
    '''Ternary composition to Cartesian coordinate'''
        
    if not (comps.shape[1]==3):
        raise ValueError('Must specify exactly three components')
    
    if normalize:
        comps = comps/comps.sum(1)[:,np.newaxis]
        
    # Convert ternary data to cartesian coordinates.
    xy = np.zeros((comps.shape[0],2))
    xy[:,1] = comps[:,1]*np.sin(60.*np.pi / 180.)
    xy[:,0] = comps[:,0] + xy[:,1]*np.sin(30.*np.pi/180.)/np.sin(60*np.pi/180)
    return xy

def xy_to_ternary(xy,base=1.0):
    '''Ternary composition to Cartesian coordinate'''
        
    # Convert ternary data to cartesian coordinates.
    ternary = np.zeros((xy.shape[0],3))
    ternary[:,1] = xy[:,1]/np.sin(60.*np.pi / 180.)
    ternary[:,0] = xy[:,0] - xy[:,1]*np.sin(30.*np.pi/180.)/np.sin(60*np.pi/180)
    ternary[:,2] = 1.0 - ternary[:,1] - ternary[:,0]
    
    ternary*=base
    
    return ternary
    
def composition_grid_ternary(pts_per_row=50,basis=1.0,dim=3,eps=1e-9):
    pts = []
    for i in product(*[np.linspace(0,1.0,pts_per_row)]*(dim-1)):
        if sum(i)>(1.0+eps):
            continue
            
        j = 1.0-sum(i)
        
        if j<(0.0-eps):
            continue
        pt = [k*basis for k in [*i,j]]
        pts.append(pt)
    return np.array(pts)

def make_ordinal_labels(labels):
    encoder = OrdinalEncoder()
    labels_ordinal = encoder.fit_transform(labels.values.reshape(-1, 1)).flatten()
    labels_ordinal = labels.copy(data=labels_ordinal)
    return labels_ordinal
