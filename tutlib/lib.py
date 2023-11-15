import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import gpflow
from gpflow.utilities.traversal import leaf_components

import itertools
from itertools import product

import matplotlib.pyplot as plt
import mpltern
import numpy as np
import pandas as pd
import xarray as xr
import h5py #for Nexus file

import tqdm

import sasmodels.data
import sasmodels.core
import sasmodels.direct_model

from shapely import MultiPoint
from shapely.geometry import Point
from shapely import concave_hull

from sklearn.preprocessing import OrdinalEncoder

from scipy.signal import savgol_filter


    

def get_virtual_instrument2(noise=1e-5,boundary_dataset_path='./reference_data/pluronic.nc'):
    boundary_dataset = xr.load_dataset(boundary_dataset_path)
    boundary_dataset['a'] = boundary_dataset['a']
    boundary_dataset['b'] = boundary_dataset['b']
    boundary_dataset['c'] = boundary_dataset['c']
    boundary_dataset.attrs['labels'] = 'phase'
    boundary_dataset.attrs['components'] = ['c','a','b']
    
    inst_client = VirtualSAS(noise=noise)
    inst_client.boundary_dataset = boundary_dataset
    inst_client.data = {}
    inst_client.trace_boundaries(hull_tracing_ratio=0.25,drop_phases=['D'])
    for fname in ['low_q.ABS','med_q.ABS','high_q.ABS']:
        data = sasmodels.data.load_data('./reference_data/'+fname)
        inst_client.add_configuration(
            q =list(data.x),
            I =list(data.y),
            dI=list(data.dy),
            dq=list(data.dx),
            reset=False
        )
    inst_client.add_sasview_model(
           label='V1',
           model_name = 'teubner_strey',
           model_kw = {
               'scale':0.05,
               'background':1.0,
               'sld_a':1.0,
               'sld_b':6.0,
               'volfraction_a':0.5,
               'd':150,
               'xi':150
           }
       )
       
    inst_client.add_sasview_model(
        label='V2',
        model_name = 'teubner_strey',
        model_kw = {
            'scale':0.05,
            'background':1.0,
            'sld_a':1.0,
            'sld_b':6.0,
            'volfraction_a':0.2,
            'd':200,
            'xi':250
        }
    )
    
    inst_client.add_sasview_model(
        label='I1',
        model_name = 'sc_paracrystal',
        model_kw = {
            'scale':0.01,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'radius':100,
            'dnn':150,
        }
    )
    
    inst_client.add_sasview_model(
        label='I2',
        model_name = 'sc_paracrystal',
        model_kw = {
            'scale':0.01,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'radius':200,
            'dnn':250,
        }
    )
    
    inst_client.add_sasview_model(
        label='L1',
        model_name = 'sphere',
        model_kw = {
            'scale':0.005,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'radius':100,
        }
    )
    
    inst_client.add_sasview_model(
        label='L2',
        model_name = 'sphere',
        model_kw = {
            'scale':0.005,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'radius':200,
        }
    )
    
    inst_client.add_sasview_model(
        label='H1',
        model_name = 'cylinder',
        model_kw = {
            'scale':0.01,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'radius':100,
            'length':300,
        }
    )
    
    inst_client.add_sasview_model(
        label='H2',
        model_name = 'cylinder',
        model_kw = {
            'scale':0.001,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'radius':200,
            'length':500,
        }
    ) 
    inst_client.add_sasview_model(
        label='La',
        model_name = 'lamellar',
        model_kw = {
            'scale':0.01,
            'background':1.0,
            'sld':1.0,
            'sld_solvent':6.0,
            'thickness':200,
        }
    )
    
    inst_client.add_sasview_model(
        label='D',
        model_name = 'power_law',
        model_kw = {
            'scale':0.0,
            'background':1.0,
        }
    )
    return inst_client
    
def get_virtual_instrument1(noise=1e-5,boundary_dataset_path='./reference_data/triangleV2.nc'):
    boundary_dataset = xr.load_dataset(boundary_dataset_path)
    boundary_dataset.attrs['labels'] = 'labels'
    boundary_dataset.attrs['components'] = ['c','a','b']
    
    inst_client = VirtualSAS(noise=noise)
    inst_client.boundary_dataset = boundary_dataset
    inst_client.trace_boundaries(hull_tracing_ratio=0.95,drop_phases=['D'])
    for fname in ['low_q.ABS','med_q.ABS','high_q.ABS']:
        data = sasmodels.data.load_data('./reference_data/'+fname)
        inst_client.add_configuration(
            q =list(data.x),
            I =list(data.y),
            dI=list(data.dy),
            dq=list(data.dx),
            reset=False
        )
    inst_client.add_sasview_model(
           label='A',
           model_name = 'teubner_strey',
           model_kw = {
               'scale':0.05,
               'background':1.0,
               'sld_a':1.0,
               'sld_b':6.0,
               'volfraction_a':0.5,
               'd':150,
               'xi':150
           }
       )
       
    
    inst_client.add_sasview_model(
        label='D',
        model_name = 'power_law',
        model_kw = {
            'scale':0.0,
            'background':1.0,
        }
    )
    return inst_client

def actively_learn(input_dataset,niter,label,extrapolate,choose_next_acquisition,instrument,grid_pts_per_row=100):
    grid = composition_grid_ternary(pts_per_row=grid_pts_per_row,basis=1.0)
    
    
    results = []
    for step in tqdm.tqdm(range(niter)):
        working_dataset = input_dataset.copy()
        working_dataset['a_grid'] = ('grid',grid[:,0])
        working_dataset['b_grid'] = ('grid',grid[:,1])
        working_dataset['c_grid'] = ('grid',grid[:,2])
        working_dataset.attrs['step'] = step
        working_dataset.attrs['components'] = ['c','a','b']
        working_dataset.attrs['components_grid'] = ['c_grid','a_grid','b_grid']
        
        working_dataset = label(working_dataset)
        
        working_dataset = extrapolate(working_dataset)
        
        working_dataset = choose_next_acquisition(working_dataset)
            
        next_sample_dict = working_dataset.attrs['next_sample']
        next_data = instrument.measure(**next_sample_dict)
        
        input_dataset = xr.concat([input_dataset,next_data],dim='sample')
        
        results.append(working_dataset)
    return results


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
    
    
    
def make_ordinal_labels(labels):
    encoder = OrdinalEncoder()
    labels_ordinal = encoder.fit_transform(labels.values.reshape(-1, 1)).flatten()
    labels_ordinal = labels.copy(data=labels_ordinal)
    return labels_ordinal

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
    
    
def plot_ternary_surface(data, components, labels=None, set_axes_labels=True, ternary=True, **mpl_kw):

    if len(components) == 3 and ternary:
        try:
            import mpltern
        except ImportError as e:
            raise ImportError('Could not import mpltern. Please install via conda or pip') from e
        projection = 'ternary'
    elif len(components) == 2:
        projection = None
    else:
        raise ValueError(f'plot_surface only compatible with 2 or 3 components. You passed: {components}')

    coords = data[components].to_array('component').transpose(..., 'component').values

    if (labels is None):
        if ('labels' in data.coords):
            labels = data.coords['labels'].values
        elif ('labels' in data):
            labels = data['labels'].values
        else:
            labels = np.zeros(coords.shape[0])
    elif isinstance(labels, str) and (labels in data):
        labels = data[labels].values

    if ('cmap' not in mpl_kw) and ('color' not in mpl_kw):
        mpl_kw['cmap'] = 'viridis'
    if 'edgecolor' not in mpl_kw:
        mpl_kw['edgecolor'] = 'face'

    if ('ax' in mpl_kw):
        ax = mpl_kw.pop('ax')

    else:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=projection))

    artists = ax.tripcolor(*coords.T, labels, **mpl_kw)

    if set_axes_labels:
        if projection == 'ternary':
            labels = {k: v for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
        else:
            labels = {k: v for k, v in zip(['xlabel', 'ylabel'], components)}
        ax.set(**labels)
        ax.grid('on', color='black')
    return artists

def plot_ternary_scatter(data, components, labels=None, set_axes_labels=True, ternary=True, discrete_labels=True,
                     **mpl_kw):

        if len(components) == 3 and ternary:
            try:
                import mpltern
            except ImportError as e:
                raise ImportError('Could not import mpltern. Please install via conda or pip') from e

            projection = 'ternary'
        elif len(components) == 3:
            projection = None
        elif len(components) == 2:
            projection = None
        else:
            raise ValueError(f'plot_surface only compatible with 2 or 3 components. You passed: {components}')

        coords = np.vstack(list(data[c].values for c in components)).T

        if (labels is None):
            if ('labels' in data.coords):
                labels = data.coords['labels'].values
            elif ('labels' in data):
                labels = data['labels'].values
            else:
                labels = np.zeros(coords.shape[0])
        elif isinstance(labels, str) and (labels in data):
            labels = data[labels].values

        if ('ax' in mpl_kw):
            ax = mpl_kw.pop('ax')
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=projection))

        if discrete_labels:
            if 'markers' not in mpl_kw:
                markers = itertools.cycle(['^', 'v', '<', '>', 'o', 'd', 'p', 'x'])
            artists = []
            for label in np.unique(labels):
                mask = (labels == label)
                mpl_kw['marker'] = next(markers)
                artists.append(ax.scatter(*coords[mask].T, **mpl_kw))
        else:
            artists = ax.scatter(*coords.T, c=labels, **mpl_kw)

        if set_axes_labels:
            if projection == 'ternary':
                labels = {k: v for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
            else:
                labels = {k: v for k, v in zip(['xlabel', 'ylabel'], components)}
            ax.set(**labels)
            ax.grid('on', color='black')
        return artists