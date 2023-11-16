import xarray as xr
from tutlib.VirtualInstrument import VirtualInstrument

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
        data = pd.read_csv(str(pathlib.Path(reference_data_path)/fname),delim_whitespace=True)
        inst_client.add_configuration(
            q =list(data.q.astype(np.float)),
            I =list(data.I.astype(np.float)),
            dI=list(data.dI.astype(np.float)),
            dq=list(data.dq.astype(np.flo)),
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
        data = pd.read_csv(str(pathlib.Path(reference_data_path)/fname),delim_whitespace=True)
        inst_client.add_configuration(
            q =list(data.q),
            I =list(data.I),
            dI=list(data.dI),
            dq=list(data.dq),
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
