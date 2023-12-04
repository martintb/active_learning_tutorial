import numpy as np
import xarray as xr
import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython import display

from tutlib.util import composition_grid_ternary,calculate_perimeter_score
from tutlib.plot import plot_ternary


def actively_learn(
  input_dataset,
  niter,
  label,
  extrapolate,
  choose_next_acquisition,
  instrument,
  grid_pts_per_row=100,
  plot='both',
  plot_every=5):
    grid = composition_grid_ternary(pts_per_row=grid_pts_per_row,basis=1.0)

    gt_hulls = trace_boundaries(instrument.boundary_dataset,hull_tracing_ratio=0.2)
    gt_xy =  np.vstack(gt_hulls['A'].boundary.xy).T# needs to be generalized for multi-phase

    fig = None
    
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

        # #calculate perimeter score
        # means = []
        # stds = []
        # for result in tqdm.tqdm(results):
        #   result.attrs['labels'] = 'labels'
        #   mean,std = calculate_perimeter_score(input_dataset,gt_xy)
        #   means.append(mean)
        #   stds.append(std)
        # score_x = np.arange(len(means))
        # score_ = np.array(means)
# 
        # y1 = y+np.array(stds)
        # y2 = y-np.array(stds)


        if plot and (step%plot_every)==0:
          if fig is not None:
            display.clear_output(wait=True)

          if plot=='ternary':
            fig = plot_ternary(working_dataset,['c','a','b'],next_point=next_sample_dict,show=True)
          elif plot=='score':
            
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=x,y=y,showlegend=False),row=1,col=1)
          elif plot=='both':
            fig = go.FigureWidget(make_subplots(1,2,specs=[[{'type':'xy'},{'type':'ternary'}]]))
            ternary_fig = plot_ternary(working_dataset,['c','a','b'],next_point=next_sample_dict,show=False)
            for data in ternary_fig.data:
              fig.add_trace(data.update(showlegend=False),row=1,col=2)
            fig.add_trace(go.Scatter(x=x,y=y,showlegend=False),row=1,col=1)
          else:
            raise ValueError("Plot must be 'ternary', 'score', 'both', or None")
        
        results.append(working_dataset)
    return results