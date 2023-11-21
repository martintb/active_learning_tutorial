import matplotlib.pyplot as plt
import numpy as np
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
      

def plot_ternary(dataset,components,labels='labels',include_surface=True,surface_data='acquisition',show=True,next_point=None):
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
    trace = go.Scatterternary(
      a = dataset[components[0]+'_grid'],
      b = dataset[components[1]+'_grid'],
      c = dataset[components[2]+'_grid'],
      mode="markers",
      marker={'symbol':'circle','color':dataset[surface_data],'coloraxis':'coloraxis'}
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
