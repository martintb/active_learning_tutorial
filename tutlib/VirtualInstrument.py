import matplotlib.pyplot as plt
import mpltern
import numpy as np
import pandas as pd
import xarray as xr

import pathlib
import itertools

import tqdm

import sasmodels.data
import sasmodels.core
import sasmodels.direct_model

from shapely import MultiPoint
from shapely.geometry import Point
from shapely import concave_hull

from scipy.signal import savgol_filter

from tutlib.util import ternary_to_xy,xy_to_ternary


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
    
            
    def measure_multiple(self,composition_list):
      data_list = []
      for comp in composition_list:
          dataset = self.measure(**comp)
          data_list.append(dataset)
      dataset = xr.concat(data_list,dim='sample')
      return dataset


    def measure(self,a,b,c):
        self.data['sample_composition'] = {
            'a':{'value':a,'units':''},
            'b':{'value':b,'units':''},
            'c':{'value':c,'units':''},
        }
        self._expose();
        
        sas = xr.DataArray(self.data['I'],coords={'q':self.data['q']})
        q_geom = np.geomspace(sas.q.min(),sas.q.max(), 250)
        sas = sas.groupby('q').mean().interp(q=q_geom)
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
