from collections import defaultdict
import numpy as np
import xarray as xr
import pandas as pd
import tqdm
import warnings
import shapely

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

def actively_learn(
  input_dataset,
  niter,
  num_phases,
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

    gt_hulls = trace_boundaries(boundary_dataset,hull_tracing_ratio=0.25,drop_phases=['D'])
    all_gt = shapely.unary_union(list(gt_hulls.values()))

    # reconstruct "D" phase to be difference between all other phases and full ternary
    xy = np.vstack([[0,1,0.5,0],[0,0,np.sqrt(3)/2,0]]).T
    triangle = shapely.Polygon(xy)
    gt_hulls['D'] = shapely.difference(triangle,all_gt)

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
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          working_dataset = label(working_dataset,num_phases=num_phases)
          working_dataset = extrapolate(working_dataset)
          working_dataset = choose_next_acquisition(working_dataset)
            
        # "measure" next sample"
        next_sample_dict = working_dataset.attrs['next_sample']
        next_data = instrument.measure(**next_sample_dict)
        
        input_dataset = xr.concat([input_dataset,next_data],dim='sample')

        #calculate all possible perimeter scores
        wd = working_dataset.copy()
        wd.attrs['components_grid'] = ['b_grid','c_grid','a_grid']
        al_hulls = trace_boundaries(wd,component_attr='components_grid',label_attr='labels_grid',hull_tracing_ratio=0.25)
        # al_hulls = trace_boundaries(wd,component_attr='components',label_attr='labels',hull_tracing_ratio=0.25)
        all_scores = []
        for gt_name,gt_hull in gt_hulls.items():
          for al_name,al_hull in al_hulls.items():
            if al_hull.geom_type=='Point':
                 continue
            elif al_hull.geom_type == 'LineString':
                 continue

            try:
                score = calculate_perimeter_score_v2(gt_hull,al_hull)
            except NotImplementedError:
                continue

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
        # results['dataset'].append(working_dataset)
        results['scores'].append(best_scores)

        if plot and (step%plot_every)==0:
          if fig is not None:
            display.clear_output(wait=True)

          n_rows = len(best_scores) + 1
          if 'D' in best_scores.keys():
            n_rows-=1
          # print('n_rows',n_rows)
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
            # print(len(best_scores),len(score_plots))
            row = 2
            for key in best_scores.keys():
              # print(key)
              if key in plot_skip_phases:
                continue

              # print('row=',row)
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

        
    ds_output = working_dataset.copy()
    ds_output['score_mean'] = (
        pd
        .DataFrame(results['score_mean'])
        .to_xarray()
        .rename(index='AL_step')
        .to_array('phase')
        .transpose('AL_step',...)
    )
    ds_output['score_std'] = (
        pd
        .DataFrame(results['score_std'])
        .to_xarray()
        .rename(index='AL_step')
        .to_array('phase')
        .transpose('AL_step',...)
    )
    return ds_output


