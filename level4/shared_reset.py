"""Guarded restoration of a saved MPM/tool state; simulator import is unnecessary."""
from pathlib import Path
import numpy as np

MPM_KEYS=('pos','vel','C','F','active')
TOOL_KEYS=('pos','quat','vel','ang')


def load_shared_state(path, state):
    """Validate the whole packet before writing a simulator State container.

    MPMSolverState.mat_id is an unused zero placeholder in this simulator. Its
    actual immutable material map is particles_info.mat_idx, rebuilt from the
    frozen config and checked separately. It must never certify material IDs.
    """
    required={'mpm_'+key:getattr(state.mpm_solver,key) for key in MPM_KEYS}
    for i,tool in enumerate(state.tool_solver.data):
        required.update({'tool_%03d_%s'%(i,key):getattr(tool,key) for key in TOOL_KEYS})
    arrays={}
    with np.load(Path(path),allow_pickle=False) as packet:
        if len(packet.files)!=len(set(packet.files)) or set(packet.files)!=set(required)|{'mpm_mat_id'}:
            raise ValueError('State fields/tool count differ; reject partial state')
        for name,target in required.items():
            value=packet[name]
            if value.dtype.kind not in 'biuf' or not np.isfinite(value).all() or tuple(value.shape)!=tuple(target.shape):
                raise ValueError('Invalid complete state field: '+name)
            dtype=str(target.dtype).replace('torch.','')
            if str(value.dtype)!=dtype:
                raise ValueError('State dtype mismatch: '+name)
            arrays[name]=value.copy()
        active=arrays['mpm_active']
        if active.dtype.kind not in 'biu' or not np.isin(active,[0,1]).all() or not active.any():
            raise ValueError('Invalid or empty active material mask')
        placeholder=packet['mpm_mat_id']
        if placeholder.shape!=active.shape or placeholder.dtype.kind not in 'iu' or np.any(placeholder!=0):
            raise ValueError('Unexpected placeholder mat_id; use actual material map validation')
        for name,value in arrays.items():
            if name.endswith('_quat') and not np.isclose(np.linalg.norm(value),1,rtol=1e-6,atol=1e-6):
                raise ValueError('Invalid tool quaternion: '+name)
    for name,target in required.items():
        if isinstance(target,np.ndarray):
            np.copyto(target,arrays[name],casting='no')
        else:
            target.copy_(target.new_tensor(arrays[name]))
    state.s_global=0
    return {'dynamic_fields_restored':list(required),'mat_id_placeholder_not_used':True,
            'static_material_map_requires_frozen_source_and_config':True}
