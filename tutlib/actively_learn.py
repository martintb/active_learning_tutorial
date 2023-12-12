from collections import defaultdict
import numpy as np
import xarray as xr
import pandas as pd
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
from tutlib.plot import (
  plot_ternary,
  make_boundary_plots,
  make_score_plots,
)


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
  plot_every=5,
  plot_skip_phases=None,
  ):
    grid = composition_grid_ternary(pts_per_row=grid_pts_per_row,basis=1.0)

    boundary_dataset = instrument.boundary_dataset.copy()
    boundary_dataset.attrs['components'] = ['b','c','a']
    gt_hulls = trace_boundaries(boundary_dataset,hull_tracing_ratio=0.2)

    if plot_skip_phases is None:
      plot_skip_phases = []

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
        wd = working_dataset.copy()
        wd.attrs['components_grid'] = ['b_grid','c_grid','a_grid']
        al_hulls = trace_boundaries(wd,component_attr='components_grid',label_attr='labels_grid',hull_tracing_ratio=0.2)
        all_scores = []
        for gt_name,gt_hull in gt_hulls.items():
          for al_name,al_hull in al_hulls.items():
            score = calculate_perimeter_score_v2(gt_hull,al_hull)
            score['GT'] = gt_name
            score['AL'] = al_name
            all_scores.append(score)

        # find best matches for each ground truth phase
        all_scores = sorted(all_scores,key = lambda x: x['mean'])
        best_scores = {}
        for score in all_scores:
          check1 = [(score['AL'] == value['AL']) for key,value in best_scores.items()]
          check2 = [(score['GT'] == value['GT']) for key,value in best_scores.items()]
          if not (any(check1) or any(check2)):
            best_scores[score['GT']] = score

        #update results dictionary
        results['step'].append(step)
        results['score_mean'].append({key:value['mean'] for key,value in best_scores.items()})
        results['score_std'].append({key:value['std'] for key,value in best_scores.items()})
        results['dataset'].append(working_dataset)
        results['scores'].append(best_scores)

        if plot and (step%plot_every)==0:
          if fig is not None:
            display.clear_output(wait=True)

          n_rows = len(best_scores) - len(plot_skip_phases) + 1
          #print('n_rows',n_rows)
          #print('best_scores',best_scores.keys())
          #print('results[score_mean]',results['score_mean'])
          #specs = [[{'type':'ternary',"colspan":2},None]] + [[{'type':'xy'},{'type':'xy'}]]*(n_rows-1)
          specs = [[{'type':'ternary'},{'type':'ternary'}]] + [[{'type':'xy'},{'type':'xy'}]]*(n_rows-1)
          subplots = make_subplots(n_rows,2,specs=specs )
          fig = go.FigureWidget(subplots,layout={'width':600,'height':1800})

          ternary_fig = plot_ternary(
            working_dataset,
            ['c','a','b'],
            next_point=next_sample_dict,
            show=False,
            surface_colorbar=False,
            )
          for data in ternary_fig.data:
            fig.add_trace(data.update(showlegend=False),row=1,col=1)

          ternary_fig = plot_ternary(
            working_dataset,
            ['c','a','b'],
            next_point=next_sample_dict,
            show=False,
            surface_colorbar=False,
            surface_data='labels_grid'
            )
          for data in ternary_fig.data:
            fig.add_trace(data.update(showlegend=False),row=1,col=2)

          if len(best_scores)>1:
            score_plots = make_score_plots(results)
            #print(len(best_scores),len(score_plots))
            row = 2
            for key in best_scores.keys():
              if key in plot_skip_phases:
                continue

              #print('row=',row)
              fig.add_trace(score_plots[key]['lo'],row=row,col=1)
              fig.add_trace(score_plots[key]['hi'],row=row,col=1)
              fig.add_trace(score_plots[key]['mean'],row=row,col=1)

              try:
                best_score = best_scores[key]
              except KeyError:
                continue
              else:
                boundary_plots = make_boundary_plots(best_score)
                fig.add_trace(boundary_plots['GT'],row=row,col=2)
                fig.add_trace(boundary_plots['AL'],row=row,col=2)
                fig.add_trace(boundary_plots['pairs'],row=row,col=2)
              row+=1


          fig.update_layout(width=800,height=400*n_rows)
          fig.show()

        
    return results


