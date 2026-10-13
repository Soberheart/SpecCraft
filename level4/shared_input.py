"""CPU checks for a new initial observation tied to a shared physical state."""
from pathlib import Path
import numpy as np
from .artifacts import read_json, sha256


def checked_capture(evidence):
    """Recompute saved-state restoration; never certify task success or a pool."""
    evidence=Path(evidence)
    manifest=read_json(evidence/'evidence_manifest.json')['files_sha256']
    from .archive import safe_name
    actual={p.relative_to(evidence).as_posix() for p in evidence.rglob('*') if p.is_file()}
    if actual!=set(manifest)|{'evidence_manifest.json'}: raise ValueError('Captured file set differs')
    for name,digest in manifest.items():
        safe_name(name)
        if sha256(evidence/name)!=digest: raise ValueError('Captured file changed: '+name)
    audit=read_json(evidence/'shared_initialization_audit.json')
    if (audit.get('kind')!='shared_state_initialization_audit_not_pool_gate'
            or audit.get('status')!='passed' or audit.get('unchanged_limit_mm')!=.001
            or audit.get('candidate_pool_released') is not False):
        raise ValueError('Shared initialization evidence not accepted')
    if sha256(evidence/'shared_source_state.npz')!=audit['shared_source_sha256']:
        raise ValueError('Shared source differs from audit')
    for i in (1,2):
        if read_json(evidence/('attempt_%d_exit.json'%i))['exit_code']!=0:
            raise ValueError('Captured process failed')
        exact_numeric_packet(evidence/'shared_source_state.npz',evidence/('attempt_%d/reset_returned.npz'%i))
    exact_numeric_packet(evidence/'attempt_1/static_material_info.npz',evidence/'attempt_2/static_material_info.npz')
    with np.load(evidence/'attempt_1/logged_frame0.npz',allow_pickle=False) as a,np.load(evidence/'attempt_2/logged_frame0.npz',allow_pickle=False) as b:
        if a['mpm_pos'].shape!=b['mpm_pos'].shape or not np.array_equal(a['mpm_active'],b['mpm_active']):
            raise ValueError('First recorded state count/mask mismatch')
        error=float(np.max(np.abs(a['mpm_pos'].astype(float)-b['mpm_pos'].astype(float)))*200)
        if not np.isfinite(error) or error>.001 or error!=audit['frame0_pair_max_abs_mm']:
            raise ValueError('First recorded state exceeds unchanged position limit')
    return dict(audit,recomputed_from_body=True,full_model_input_alignment='not_yet_verified')


def exact_numeric_packet(first,second):
    with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
        if len(a.files)!=len(set(a.files)) or set(a.files)!=set(b.files):
            raise ValueError('Numeric packet fields differ')
        for key in a.files:
            if (a[key].dtype.kind not in 'biuf' or a[key].dtype!=b[key].dtype or a[key].shape!=b[key].shape
                    or not np.isfinite(a[key]).all() or not np.array_equal(a[key],b[key])):
                raise ValueError('Numeric packet not exactly equal: '+key)


def checked_initial_input(source, metadata):
    """Validate new frame0-derived inputs and their immutable provenance."""
    source=Path(source)
    if metadata.get('kind')!='shared_initial_observation_input_v1':
        raise ValueError('Unexpected shared initial observation version')
    required={'model_config.yaml','observations.npz','shared_source_state.npz','static_material_info.npz',
              'logged_frame0.npz','initialization_evidence.json'}
    if not required.issubset(metadata['input_sha256']):
        raise ValueError('Shared input manifest lacks required state/config bodies')
    for name,h in metadata['input_sha256'].items():
        from .archive import safe_name
        safe_name(name)
        if sha256(source/name)!=h: raise ValueError('Shared input artifact changed: '+name)
    if metadata.get('historical_initial_observation_reused') is not False:
        raise ValueError('Cannot reuse historical observation with new shared state')
    evidence=read_json(source/'initialization_evidence.json')
    if (evidence.get('status')!='passed' or evidence.get('same_saved_state_restored_exactly') is not True
            or evidence.get('actual_material_map_equal') is not True or evidence.get('recomputed_from_body') is not True
            or evidence.get('unchanged_limit_mm')!=.001 or evidence.get('frame0_within_unchanged_limit') is not True
            or evidence.get('candidate_pool_released') is not False
            or evidence.get('shared_source_sha256')!=sha256(source/'shared_source_state.npz')):
        raise ValueError('New initial input lacks checked common-state provenance')
    with np.load(source/'observations.npz',allow_pickle=False) as p:
        expected={'observed_initial','z_initial','z_goal','tools','times_s','initial_particles'}
        if set(p.files)!=expected: raise ValueError('Only whitelisted initial inputs and action schedule allowed')
        result={k:p[k].copy() for k in expected}
    for key,value in result.items():
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('Nonfinite initial input: '+key)
    particles=result['initial_particles']
    with np.load(source/'logged_frame0.npz',allow_pickle=False) as p:
        saved=p['mpm_pos'][p['mpm_active'].astype(bool)]
    if particles.ndim!=2 or particles.shape[1]!=3 or not len(particles) or not np.array_equal(particles,saved):
        raise ValueError('Initial reference does not match the saved recorded frame0')
    if result['observed_initial'].ndim!=2 or result['observed_initial'].shape[1]!=4 or not len(result['observed_initial']):
        raise ValueError('Initial observation must contain xyz and material labels')
    if result['z_initial'].ndim!=2 or not result['z_initial'].size or result['z_goal'].shape!=result['z_initial'].shape:
        raise ValueError('Invalid initial/representative latent dimensions')
    times=result['times_s']
    if (times.ndim!=1 or len(times)<2 or times[0]!=0 or (np.diff(times)<=0).any()
            or not np.isclose(times[-1],1.920,rtol=0,atol=1e-8)
            or result['tools'].ndim!=3 or result['tools'].shape[0]!=len(times) or result['tools'].shape[-1]!=3):
        raise ValueError('Invalid frozen action/time schedule')
    frames=metadata.get('rollout_frames')
    if (frames!=list(range(0,61,5)) or len(times)!=len(frames)
            or not np.allclose(times,np.asarray(frames)*.032,rtol=0,atol=1e-8)):
        raise ValueError('Rollout frames and physical times must match the frozen 5-frame schedule')
    required=metadata['execution_required_s']
    if (len(required)!=61 or not np.allclose(required,np.arange(61)*.032,rtol=0,atol=1e-8)
            or metadata['unit_to_meter']!=.2):
        raise ValueError('New input must preserve the existing scale and dense physical-time schedule')
    return result
