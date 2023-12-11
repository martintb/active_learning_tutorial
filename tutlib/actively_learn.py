from collections import defaultdict
import numpy as np
import xarray as xr
import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython import display

from tutlib.util import (
  composition_grid_ternary,
  calculate_perimeter_score_v1,
  calculate_perimeter_score_v2,
  trace_boundaries
)
from tutlib.plot import plot_ternary


def actively_learn_v1(
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
    
    results = defaultdict(list)
    for step in tqdm.tqdm(range(niter)):
        #prepare working dataset
        working_dataset = input_dataset.copy()
        working_dataset['a_grid'] = ('grid',grid[:,0])
        working_dataset['b_grid'] = ('grid',grid[:,1])
        working_dataset['c_grid'] = ('grid',grid[:,2])
        working_dataset.attrs['step'] = step
        working_dataset.attrs['labels'] = 'labels'
        working_dataset.attrs['labels_grid'] = 'labels_grid'
        working_dataset.attrs['components'] = ['c','a','b']
        working_dataset.attrs['components_grid'] = ['c_grid','a_grid','b_grid']
        
        # label, extrap, choose..
        working_dataset = label(working_dataset)
        working_dataset = extrapolate(working_dataset)
        working_dataset = choose_next_acquisition(working_dataset)
            
        # "measure" next sample"
        next_sample_dict = working_dataset.attrs['next_sample']
        next_data = instrument.measure(**next_sample_dict)
        
        input_dataset = xr.concat([input_dataset,next_data],dim='sample')

        #calculate perimeter score
        mean,std = calculate_perimeter_score_v1(
          working_dataset,
          gt_xy,
          component_attr='components_grid',
          label_attr='labels_grid',
          )
        
        #update results dictionary
        results['step'].append(step)
        results['score_mean'].append(mean)
        results['score_std'].append(std)
        results['dataset'].append(working_dataset)

        if plot and (step%plot_every)==0:
          if fig is not None:
            display.clear_output(wait=True)

          if plot in ('both','score'):
            score_x = np.array(results['step'])
            score_y = np.array(results['score_mean'])
            score_ylo = score_y-np.array(results['score_std'])
            score_yhi = score_y+np.array(results['score_std'])

          if plot=='ternary':
            fig = plot_ternary(working_dataset,['c','a','b'],next_point=next_sample_dict,show=False)
          elif plot=='score':
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=score_x,y=score_y,showlegend=False),row=1,col=1)
            fig['layout']['xaxis']['title'] = 'Step'
            fig['layout']['yaxis']['title'] = 'Perimeter Score'
          elif plot=='both':
            fig = go.FigureWidget(make_subplots(1,2,specs=[[{'type':'xy'},{'type':'ternary'}]]))
            ternary_fig = plot_ternary(working_dataset,['c','a','b'],next_point=next_sample_dict,show=False,surface_colorbar=False)
            for data in ternary_fig.data:
              fig.add_trace(data.update(showlegend=False),row=1,col=2)
            fig.add_trace(go.Scatter(x=score_x,y=score_ylo,showlegend=False,line={'color':'blue'},opacity=0.5),row=1,col=1)
            fig.add_trace(go.Scatter(x=score_x,y=score_yhi,fill='tonexty',line={'color':'blue'},opacity=0.5,fillcolor='rgba(0.0,0.0,1.0,0.3)',showlegend=False),row=1,col=1)
            fig.add_trace(go.Scatter(x=score_x,y=score_y,line={'color':'red'},showlegend=False),row=1,col=1)
            fig['layout']['xaxis']['title'] = 'Step'
            fig['layout']['yaxis']['title'] = 'Perimeter Score'
          else:
            raise ValueError("Plot must be 'ternary', 'score', 'both', or None")
          fig.show()
        
    return results

def actively_learn_v2(
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
    
    results = defaultdict(list)
    for step in tqdm.tqdm(range(niter)):
        #prepare working dataset
        working_dataset = input_dataset.copy()
        working_dataset['a_grid'] = ('grid',grid[:,0])
        working_dataset['b_grid'] = ('grid',grid[:,1])
        working_dataset['c_grid'] = ('grid',grid[:,2])
        working_dataset.attrs['step'] = step
        working_dataset.attrs['labels'] = 'labels'
        working_dataset.attrs['labels_grid'] = 'labels_grid'
        working_dataset.attrs['components'] = ['c','a','b']
        working_dataset.attrs['components_grid'] = ['c_grid','a_grid','b_grid']
        
        # label, extrap, choose..
        working_dataset = label(working_dataset)
        working_dataset = extrapolate(working_dataset)
        working_dataset = choose_next_acquisition(working_dataset)
            
        # "measure" next sample"
        next_sample_dict = working_dataset.attrs['next_sample']
        next_data = instrument.measure(**next_sample_dict)
        
        input_dataset = xr.concat([input_dataset,next_data],dim='sample')


        #calculate all possible perimeter scores
        al_hulls = trace_boundaries(working_dataset,component_attr='components_grid',label_attr='labels_grid',hull_tracing_ratio=0.2)
        all_scores = []
        for gt_name,gt_hull in gt_hulls.items():
          for al_name,al_hull in al_hulls.items():
            score = perimeter_score(gt_hull,al_hull)
            score['GT'] = gt_name
            score['AL'] = al_name
            all_scores.append(score)

        # find best matches for each ground truth phase
        all_scores = sorted(all_scores,key = lambda x: x['score'])
        best_scores = {}
        for score in all_score:
          check1 = [(score['AL'] in key) for key in best_scores.keys()]
          check2 = [(score['GT'] in key) for key in best_scores.keys()]
          if not (any(check1) or any(check2)):
            best_scores[score['GT'],score['AL']] = score

        #update results dictionary
        results['step'].append(step)
        # results['score_mean'].append([score['mean'] for score in best_scores.values()])
        # results['score_std'].append([score['std'] for score in best_scores.values()])
        results['dataset'].append(working_dataset)

        if plot and (step%plot_every)==0:
          if fig is not None:
            display.clear_output(wait=True)

          if plot in ('all','both','score'):
            score_x = np.array(results['step'])
            score_y = np.array(results['score_mean'])
            score_ylo = score_y-np.array(results['score_std'])
            score_yhi = score_y+np.array(results['score_std'])

          if plot=='ternary':
            fig = plot_ternary(working_dataset,['c','a','b'],next_point=next_sample_dict,show=False)
          elif plot=='score':
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=score_x,y=score_y,showlegend=False),row=1,col=1)
            fig['layout']['xaxis']['title'] = 'Step'
            fig['layout']['yaxis']['title'] = 'Perimeter Score'
          elif plot=='both':
            fig = go.FigureWidget(make_subplots(1,2,specs=[[{'type':'xy'},{'type':'ternary'}]]))
            ternary_fig = plot_ternary(working_dataset,['c','a','b'],next_point=next_sample_dict,show=False,surface_colorbar=False)
            for data in ternary_fig.data:
              fig.add_trace(data.update(showlegend=False),row=1,col=2)
            fig.add_trace(go.Scatter(x=score_x,y=score_ylo,showlegend=False,line={'color':'blue'},opacity=0.5),row=1,col=1)
            fig.add_trace(go.Scatter(x=score_x,y=score_yhi,fill='tonexty',line={'color':'blue'},opacity=0.5,fillcolor='rgba(0.0,0.0,1.0,0.3)',showlegend=False),row=1,col=1)
            fig.add_trace(go.Scatter(x=score_x,y=score_y,line={'color':'red'},showlegend=False),row=1,col=1)
            fig['layout']['xaxis']['title'] = 'Step'
            fig['layout']['yaxis']['title'] = 'Perimeter Score'
          else:
            raise ValueError("Plot must be 'ternary', 'score', 'both', or None")
          fig.show()
        
    return results