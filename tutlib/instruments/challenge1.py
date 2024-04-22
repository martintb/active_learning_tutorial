import pathlib

import pandas as pd
import xarray as xr

from tutlib.VirtualInstrument import VirtualSAS


def get_virtual_instrument(
        noise: float = 5e-1,
        hull_tracing_ratio: float = 0.25,
        boundary_dataset_path: str = "/content/active_learning_tutorial/challenge_datasets/challenge1.nc",
        reference_data_path: str = "./reference_sans",
) -> VirtualSAS:
    """ Generate Virtual Instrument

    Parameters
    ----------
    noise: float
        scaling value for degree of noise to include in generated data. Smaller value = less noise

    hull_tracing_ratio: float
        used in concave hull calculation i.e. when tracing phases. Larger value = smoother phases but less detail

    boundary_dataset_path: str
        Filepath of NetCDF file containing the phase boundary data for this instrument

    reference_data_path: str
        Filepath of reference SANS data used for resolution smearing and noise generation
    """
    boundary_dataset = xr.load_dataset(boundary_dataset_path)
    boundary_dataset.attrs['labels'] = 'labels'
    boundary_dataset.attrs['components'] = ['c', 'a', 'b']

    inst_client = VirtualSAS(noise=noise)
    inst_client.boundary_dataset = boundary_dataset
    inst_client.trace_boundaries(hull_tracing_ratio=hull_tracing_ratio, drop_phases=['D'])
    for fname in ['low_q.ABS', 'med_q.ABS', 'high_q.ABS']:
        data = pd.read_csv(str(pathlib.Path(reference_data_path) / fname), delim_whitespace=True)
        inst_client.add_configuration(
            q=list(data.q),
            I=list(data.I),
            dI=list(data.dI),
            dq=list(data.dq),
            reset=False
        )
    inst_client.add_sasview_model(
        label='La',
        model_name='sphere',
        model_kw={
            'scale': 0.005,
            'background': 1.0,
            'sld': 1.0,
            'sld_solvent': 6.0,
            'radius': 200,
        }
    )

    inst_client.add_sasview_model(
        label='D',
        model_name='power_law',
        model_kw={
            'scale': 1e-7,
            'background': 1.0,
            'power': 4.0,
        }
    )
    return inst_client
