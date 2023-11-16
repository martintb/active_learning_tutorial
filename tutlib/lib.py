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

import pathlib

import tqdm

import sasmodels.data
import sasmodels.core
import sasmodels.direct_model

from shapely import MultiPoint
from shapely.geometry import Point
from shapely import concave_hull

from sklearn.preprocessing import OrdinalEncoder

from scipy.signal import savgol_filter

class VirtualSAS:
    def __init__(self,noise=1e-9):
        '''
        Generates smoothly interpolated scattering data via a noiseless GPR from an experiments netcdf file
        '''
        
        
        self.hulls = {}
        self.reference_data = []
        self.sasmodels = {}
        self.boundary_dataset = None
        self.data = {}
        self.noise = noise
        
    def trace_boundaries(self,hull_tracing_ratio=0.1,drop_phases=None,reset=True):
        if self.boundary_dataset is None:
            raise ValueError('Must set boundary_dataset before calling trace_boundaries! Use client.set_driver_object.')
            
        if drop_phases is None:
            drop_phases = []
        
        if reset:
            self.hulls = {}
        
        label_variable = self.boundary_dataset.attrs['labels']
        for label,sds in self.boundary_dataset.groupby(label_variable):
            if label in drop_phases:
                continue
            comps = sds[sds.attrs['components']].to_array('component').transpose(...,'component')
            #xy = ternary_to_xy(comps.values[:,[2,0,1]]) #shapely uses a different coordinate system than we do
            xy = ternary_to_xy(comps.values) #shapely uses a different coordinate system than we do
            mp = MultiPoint(xy)
            hull = concave_hull(mp,ratio=hull_tracing_ratio)
            self.hulls[label] = hull
    
    def locate(self,composition,fast_locate=True):
        composition = np.array(composition)
        
        if self.hulls is None:
            raise ValueError('Must call trace_boundaries before locate')
            
        point = Point(*ternary_to_xy(composition))
        locations = {}
        for phase,hull in self.hulls.items():
            if hull.contains(point):
                locations[phase] = True
                if fast_locate:
                    break
            else:
                locations[phase] = False
                
        if sum(locations.values())>1:
            warnings.warn('Location in multiple phases. Phases likely overlapping')
            
        phases = [key for key,value in locations.items() if value]
        self.data['locate_locations'] = locations
        self.data['locate_phases'] = phases
        return phases

    def add_configuration(self,q,I,dI,dq,reset=True):
        '''Read in reference data for an instrument configuration'''
        if reset:
            self.reference_data = []
        data = sasmodels.data.Data1D(
            x=np.array(q),
            y=np.array(I),
            dy=np.array(dI),
            dx=np.array(dq),
        )
        self.reference_data.append(data)
        
    def add_sasview_model(self,label,model_name,model_kw):
        calculators = []
        sasdatas = []
        for sasdata in self.reference_data:
            model_info    = sasmodels.core.load_model_info(model_name)
            kernel        = sasmodels.core.build_model(model_info)
            calculator    = sasmodels.direct_model.DirectModel(sasdata,kernel)
            calculators.append(calculator)
            sasdatas.append(sasdata)
            
        self.sasmodels[label] = {
            'name':model_name,
            'kw':model_kw,
            'calculators':calculators,
            'sasdata':sasdatas,
        }
        
    def generate(self,label):
        kw          = self.sasmodels[label]['kw']
        calculators = self.sasmodels[label]['calculators']
        sasdatas    = self.sasmodels[label]['sasdata']
        
        I_noiseless_list = []
        I_list = []
        dI_list = []
        for sasdata,calc in zip(sasdatas,calculators):
            I_noiseless = calc(**kw)
            
            dI_model = sasdata.dy*np.sqrt(I_noiseless/sasdata.y)
            mean_var= np.mean(dI_model*dI_model/I_noiseless)
            # dI = sasdata.dy*np.sqrt(noise*noise/mean_var)
            dI = sasdata.dy*self.noise/mean_var
            
            I = np.random.normal(loc=I_noiseless,scale=dI)
            
            I_noiseless = pd.Series(data=I_noiseless,index=sasdata.x)
            I = pd.Series(data=I,index=sasdata.x)
            dI = pd.Series(data=dI,index=sasdata.x)
            
            I_list.append(I)
            I_noiseless_list.append(I_noiseless)
            dI_list.append(dI)
            
        I           = pd.concat(I_list).sort_index()
        I_noiseless = pd.concat(I_noiseless_list).sort_index()
        dI          = pd.concat(dI_list).sort_index()
        return I,I_noiseless,dI
    
    def _expose(self,*args,**kwargs):
        '''Mimic the expose command from other instrument servers'''
        
        components = self.boundary_dataset.attrs['components']
        composition = [[self.data['sample_composition'][component]['value'] for component in components]] #from tiled  
        
        phases = self.locate(composition)
        if len(phases)==0:
            label = 'D'
        elif len(phases)==1:
            label = phases[0]
        else:
            label = phases[0]
        
        I,I_noiseless,dI = self.generate(label)
        
        self.data['q'] = I.index.values
        self.data['I'] = I.values
        self.data['I_noiseless'] = I_noiseless.values
        self.data['dI'] = dI.values
        self.data['components'] = components
        
        return I.values
    
            
    def measure(self,a,b,c):
        self.data['sample_composition'] = {
            'a':{'value':a,'units':''},
            'b':{'value':b,'units':''},
            'c':{'value':c,'units':''},
        }
        self._expose();
        
        sas = xr.DataArray(self.data['I'],coords={'q':self.data['q']})
        q_geom = np.geomspace(sas.q.min(),sas.q.max(), 250)
        sas = sas.interp(q=q_geom)
        log_sas = xr.DataArray(np.log10(sas.values),coords={'logq':np.log10(sas.q.values)})
        delta_logq = (log_sas.logq[1] - log_sas.logq[0]).values[()]
        dlog_sas_np = savgol_filter(log_sas, window_length=31, polyorder=2, delta=delta_logq,deriv=1)
        dlog_sas = log_sas.copy(data=dlog_sas_np)
    
        dataset = xr.Dataset()
        dataset['sas'] = sas 
        dataset['log_sas'] = log_sas
        dataset['dlog_sas'] = dlog_sas
        dataset['a'] = a
        dataset['b'] = b
        dataset['c'] = c
        return dataset
    
    def _plot_ground_truth_data(self,**mpl_kw):
        if self.hulls is None:
            raise ValueError('No hulls calculated. Run .trace_boundaries')
            
        fig,ax = plt.subplots(subplot_kw={'projection':'ternary'})
        labels = self.boundary_dataset[self.boundary_dataset.attrs['labels']]
        
        components = self.boundary_dataset.attrs['components']
        coords = np.vstack(list(self.boundary_dataset[c].values for c in components)).T
        
        artists = []
        markers = itertools.cycle(['^', 'v', '<', '>', 'o', 'd', 'p', 'x'])
        for label in np.unique(labels):
            mask = (labels == label)
            mpl_kw['marker'] = next(markers)
            artists.append(ax.scatter(*coords[mask].T, **mpl_kw))
            
        labels = {k: v for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
        ax.set(**labels)
        ax.grid('on', color='black')
        return artists
    
    def _plot_ground_truth(self,**mpl_kw):
        if self.hulls is None:
            raise ValueError('No hulls calculated. Run .trace_boundaries')
            
        fig,ax = plt.subplots(subplot_kw={'projection':'ternary'})
        
        components = self.boundary_dataset.attrs['components']
        coords = np.vstack(list(self.boundary_dataset[c].values for c in components)).T
        
        artists = []
        markers = itertools.cycle(['^', 'v', '<', '>', 'o', 'd', 'p', 'x'])
        for label,hull in self.hulls.items():
            mpl_kw['marker'] = next(markers)
            
            xy = hull.boundary.coords.xy
            xy = np.array(xy).T
            xy = xy_to_ternary(xy)
            artists.append(ax.scatter(*xy.T, **mpl_kw))
            
        labels = {k: v for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
        ax.set(**labels)
        ax.grid('on', color='black')
        return artists


    

def get_virtual_instrument2(noise=1e-5,boundary_dataset_path='./reference_data/pluronic.nc',reference_data_path="./reference_data/"):
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
        data = sasmodels.data.load_data(str(pathlib.Path(reference_data_path)/fname))
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
    
def get_virtual_instrument1(noise=1e-5,boundary_dataset_path='./reference_data/triangleV2.nc',reference_data_path="./reference_data/"):
    boundary_dataset = xr.load_dataset(boundary_dataset_path)
    boundary_dataset.attrs['labels'] = 'labels'
    boundary_dataset.attrs['components'] = ['c','a','b']
    
    inst_client = VirtualSAS(noise=noise)
    inst_client.boundary_dataset = boundary_dataset
    inst_client.trace_boundaries(hull_tracing_ratio=0.95,drop_phases=['D'])
    for fname in ['low_q.ABS','med_q.ABS','high_q.ABS']:
        data = sasmodels.data.load_data(str(pathlib.Path(reference_data_path)/fname))
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

import tensorflow as tf
import gpflow

class GaussianProcess:
    def __init__(self,dataset,kernel=None):
        self.reset_GP(dataset,kernel)
        self.iter_monitor = lambda x: None
        self.final_monitor = lambda x: None
        
    def construct_data(self):

        if 'labels_ordinal' not in self.dataset:
            raise ValueError('Must have labels_ordinal variable in Dataset before making GP!')

            
        labels = self.dataset['labels_ordinal'].values
        if len(labels.shape)==1:
            labels = labels[:,np.newaxis]
            
        domain = self.transform_domain()
        
        data = (domain,labels)
        return data
        
    def transform_domain(self,components=None):
      if components is None:
        components = self.dataset.attrs['components']
      if not (len(self.dataset.attrs['components'])==3):   
          raise ValueError("Ternary domain transform specified but len(components)!=3") 
      comp = self.dataset[components].to_array('component').transpose(...,'component')
      domain = ternary_to_xy(comp.values)
      return domain
            
    def reset_GP(self,dataset,kernel=None):
        self.dataset = dataset
        self.n_classes = dataset.attrs['n_phases']

        data = self.construct_data()
            
        if kernel is None:
            kernel = gpflow.kernels.Matern32(variance=0.1,lengthscales=0.1) 
            
        invlink = gpflow.likelihoods.RobustMax(self.n_classes)  
        likelihood = gpflow.likelihoods.MultiClass(self.n_classes, invlink=invlink)  
        self.model = gpflow.models.VGP(
            data=data, 
            kernel=kernel, 
            likelihood=likelihood, 
            num_latent_gps=self.n_classes
        ) 
        self.loss = self.model.training_loss_closure(compile=True)
        self.trainable_variables = self.model.trainable_variables
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        
    def reset_monitoring(self,log_dir='test/',iter_period=1):
        model_task = ModelToTensorBoard(log_dir, self.model,keywords_to_monitor=['*'])
        lml_task   = ScalarToTensorBoard(log_dir, lambda: self.loss(), "Training Loss")
        
        fast_tasks = MonitorTaskGroup([model_task,lml_task],period=iter_period)
        self.iter_monitor = Monitor(fast_tasks)
        
        image_task = ImageToTensorBoard(
            log_dir, 
            self.plot, 
            "Mean/Variance",
            fig_kw=dict(figsize=(18,6)),
            subplots_kw=dict(nrows=1,ncols=3)
        )
        slow_tasks = MonitorTaskGroup(image_task) 
        self.final_monitor = Monitor(slow_tasks)

    def optimize(self,N,final_monitor_step=None,progress_bar=False):
        if progress_bar:
            for i in tqdm.tqdm(tf.range(N),total=N):
                self._step(i)
        else:
            for i in tf.range(N):
                self._step(i)
            
        if final_monitor_step is None:
            final_monitor_step = i
        self.final_monitor(final_monitor_step)
            
    @tf.function
    def _step(self,i):
        self.optimizer.minimize(self.loss,self.trainable_variables) 
        self.iter_monitor(i)
    
    def predict(self,components):
        domain = self.transform_domain(components=components)
        self.y = self.model.predict_y(domain)
        self.y_mean = self.y[0].numpy() 
        self.y_var = self.y[1].numpy() 
        return {'mean':self.y_mean,'var':self.y_var}

