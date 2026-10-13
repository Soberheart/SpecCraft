"""Lossless numeric/JSON transport for a recorded frame, independent of pickle.

The simulator's original logs remain unchanged. Only an explicitly derived
frame0 is reconstructed into a native pickle on the preprocessing host.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda:inp.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def _numeric(value):
    array=np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise ValueError('Only finite numeric arrays/scalars may enter initial-frame transport')
    return array


def save_frame(frame,arrays_path,metadata_path,provenance):
    arrays={}
    def encode(value):
        if isinstance(value,(np.ndarray,np.generic)):
            array=_numeric(value); name='array_%04d'%len(arrays)
            arrays[name]=array
            return {'type':'numpy_scalar' if isinstance(value,np.generic) else 'array',
                    'name':name,'dtype':array.dtype.str,'shape':list(array.shape)}
        if isinstance(value,dict):
            if not all(isinstance(k,str) for k in value): raise ValueError('Frame dictionary keys must be strings')
            return {'type':'dict','items':[[key,encode(v)] for key,v in value.items()]}
        if isinstance(value,(list,tuple)):
            return {'type':'tuple' if isinstance(value,tuple) else 'list','items':[encode(v) for v in value]}
        if value is None or type(value) in (bool,int,float,str):
            if type(value) is float and not np.isfinite(value): raise ValueError('Nonfinite frame scalar')
            return {'type':'scalar','value':value}
        raise ValueError('Unsupported initial-frame field type: '+type(value).__name__)
    tree=encode(frame)
    with Path(arrays_path).open('xb') as out: np.savez_compressed(out,**arrays)
    metadata={'kind':'recorded_initial_frame_numeric_transport_v1','frame_tree':tree,
              'arrays_sha256':digest(arrays_path),'provenance':provenance,
              'historical_simulation_log_modified':False,'new_simulation_run':False}
    with Path(metadata_path).open('x',encoding='utf-8') as out:
        json.dump(metadata,out,indent=2,ensure_ascii=False,allow_nan=False)
    return metadata


def load_frame(arrays_path,metadata_path):
    metadata=json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if (metadata.get('kind')!='recorded_initial_frame_numeric_transport_v1'
            or metadata.get('historical_simulation_log_modified') is not False
            or metadata.get('new_simulation_run') is not False
            or digest(arrays_path)!=metadata['arrays_sha256']):
        raise ValueError('Initial-frame transport version/hash/provenance changed')
    used=set()
    with np.load(arrays_path,allow_pickle=False) as packet:
        if len(packet.files)!=len(set(packet.files)): raise ValueError('Duplicate numeric frame members')
        def decode(node):
            tag=node.get('type')
            if tag in ('array','numpy_scalar'):
                if set(node)!={'type','name','dtype','shape'}: raise ValueError('Unexpected numeric frame descriptor')
                name=node['name']
                if name in used or name not in packet.files: raise ValueError('Missing/reused numeric frame member')
                used.add(name); array=_numeric(packet[name])
                if array.dtype.str!=node['dtype'] or list(array.shape)!=node['shape']:
                    raise ValueError('Initial-frame numeric dtype/shape changed')
                if tag=='numpy_scalar':
                    if array.shape!=(): raise ValueError('Numeric scalar must be zero dimensional')
                    return array[()]
                return array.copy()
            if tag=='dict':
                if set(node)!={'type','items'}: raise ValueError('Unexpected frame dictionary descriptor')
                result={}
                for key,value in node['items']:
                    if not isinstance(key,str) or key in result: raise ValueError('Invalid/duplicate frame key')
                    result[key]=decode(value)
                return result
            if tag in ('list','tuple'):
                if set(node)!={'type','items'}: raise ValueError('Unexpected frame sequence descriptor')
                result=[decode(v) for v in node['items']]
                return tuple(result) if tag=='tuple' else result
            if tag=='scalar':
                if set(node)!={'type','value'}: raise ValueError('Unexpected scalar descriptor')
                value=node['value']
                if value is not None and type(value) not in (bool,int,float,str): raise ValueError('Unexpected frame scalar')
                if type(value) is float and not np.isfinite(value): raise ValueError('Nonfinite frame scalar')
                return value
            raise ValueError('Unexpected initial-frame descriptor')
        frame=decode(metadata['frame_tree'])
        if used!=set(packet.files): raise ValueError('Unreferenced/future fields in initial-frame packet')
    if (not isinstance(frame,dict) or frame.get('step')!=0 or frame.get('frame_idx')!=0 or frame.get('t')!=0):
        raise ValueError('Only recorded frame0 at the unchanged logged time origin is supported')
    return frame


def assert_same_frame(first,second):
    """Compare every field without converting dtype, coordinates or time."""
    if isinstance(first,(np.ndarray,np.generic)):
        if (type(first)!=type(second) or first.dtype!=second.dtype
                or np.asarray(first).shape!=np.asarray(second).shape or not np.array_equal(first,second)):
            raise ValueError('Initial-frame numeric content/type changed')
    elif isinstance(first,dict):
        if type(second)!=dict or list(first)!=list(second): raise ValueError('Initial-frame keys/order changed')
        for key in first: assert_same_frame(first[key],second[key])
    elif isinstance(first,(list,tuple)):
        if type(first)!=type(second) or len(first)!=len(second): raise ValueError('Initial-frame sequence changed')
        for a,b in zip(first,second): assert_same_frame(a,b)
    elif type(first)!=type(second) or first!=second:
        raise ValueError('Initial-frame scalar changed')


def validate_material_reference(frame,logged_state):
    with np.load(logged_state,allow_pickle=False) as state:
        particles=state['mpm_pos'][state['mpm_active'].astype(bool)]
    actual=_numeric(frame['obj']['particles']['pos'])
    if actual.dtype!=particles.dtype or actual.shape!=particles.shape or not np.array_equal(actual,particles):
        raise ValueError('Transported initial particles differ from recorded frame0 state')
    if np.asarray(frame['obj']['particles']['colliding']).sum()!=0:
        raise ValueError('Recorded nominal frame0 unexpectedly contains tool contact')
    return {'particle_count':len(particles),'dtype':particles.dtype.str,
            'logged_frame0_material_matches_exactly':True,'logged_time_origin_s':0.,
            'primary_frame':60,'primary_s':1.920,'auxiliary_frame':62,'auxiliary_s':1.984}
