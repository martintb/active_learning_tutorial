import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpltern
import itertools

import plotly.graph_objects as go

from tutlib.util import xy_to_ternary,ternary_to_xy

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
      

def plot_ternary(dataset,components,labels='labels',include_surface=True,surface_data='acquisition',show=True,next_point=None,surface_colorbar=True):
  layout = dict(
          width=750,
          ternary=dict(
              sum=1,
              aaxis=dict(
                  title=dict(text='c'), min=0.01, linewidth=2, ticks="outside"
              ),
              baxis=dict(
                  title=dict(text='a'), min=0.01, linewidth=2, ticks="outside"
              ),
              caxis=dict(
                  title=dict(text='b'), min=0.01, linewidth=2, ticks="outside"
              ),
          ),
          showlegend=False,
      )

  fig = go.FigureWidget(layout=layout)
  if include_surface:
    if surface_colorbar:
      marker={'symbol':'circle','color':dataset[surface_data],'coloraxis':'coloraxis'}
    else:
      marker = {'symbol':'circle','color':dataset[surface_data]}
    trace = go.Scatterternary(
      a = dataset[components[0]+'_grid'],
      b = dataset[components[1]+'_grid'],
      c = dataset[components[2]+'_grid'],
      mode="markers",
      marker=marker,
      showlegend=False,
    )
    fig.add_trace(trace)

  markers = itertools.cycle(['triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'circle', 'diamond'])
  if labels in dataset:
    for label in np.unique(dataset[labels]):
      mask = dataset['labels']==label
      trace = go.Scatterternary(
        a = dataset[components[0]][mask],
        b = dataset[components[1]][mask],
        c = dataset[components[2]][mask],
        mode="markers",
        marker={'symbol':next(markers),'size':12}
      )
      fig.add_trace(trace)
  else:
    trace = go.Scatterternary(
      a = dataset[components[0]],
      b = dataset[components[1]],
      c = dataset[components[2]],
      mode="markers",
      marker={'symbol':next(markers),'size':12}
    )
    fig.add_trace(trace)

  if next_point is not None:
    trace = go.Scatterternary(
      a = [next_point[components[0]]],
      b = [next_point[components[1]]],
      c = [next_point[components[2]]],
      mode="markers",
      marker={'symbol':'x','size':12,'color':'cyan'}
    )
    fig.add_trace(trace)


  if show:
    fig.show()

  return fig

def make_score_plots(results_dict):
  score_x = results_dict['step']
  df_mean = pd.DataFrame(results_dict['score_mean'])
  df_std = pd.DataFrame(results_dict['score_std'])
  mask = df_mean.isna().any(axis=1)
  df_mean.loc[mask] = pd.NA
  df_std.loc[mask] = pd.NA

  plots = {}
  for i,(GT,score_y) in enumerate(df_mean.items(),start=1):
    score_ylo = score_y - df_std[GT]
    score_yhi = score_y + df_std[GT]
    plots[GT] = {}
    plots[GT]['lo'] = go.Scatter(
        x=score_x,
        y=score_ylo,
        showlegend=False,
        line={'color':'blue'},
        opacity=0.5
        )

    plots[GT]['hi'] = go.Scatter(
        x=score_x,
        y=score_yhi,
        fill='tonexty',
        line={'color':'blue'},
        opacity=0.5,
        fillcolor='rgba(0.0,0.0,1.0,0.3)',
        showlegend=False)

    plots[GT]['mean'] = go.Scatter(
        x=score_x,
        y=score_y,
        line={'color':'red'},
        showlegend=False)
  return plots

def make_boundary_plots(score_dict):
  gt_xy = score_dict['hull1_xy']
  xy = score_dict['hull2_xy']
  idx = score_dict['pair_idx']
  pair_xy = score_dict['pair_coord']

  out = {}
  out['GT'] = go.Scatter(x = gt_xy.T[0],y=gt_xy.T[1],mode='lines+markers',line={'color':'black'},marker={'symbol':'circle-open'},name='GT')
  out['AL'] =  go.Scatter(x = xy.T[0],y=xy.T[1],mode='lines+markers',line={'color':'green'},marker={'symbol':'triangle-up-open'},name='AL')
  out['pairs'] = go.Scatter(x=pair_xy[:,0],y=pair_xy[:,1],line={'color':'red'},opacity=0.5,name=None)
  return out
