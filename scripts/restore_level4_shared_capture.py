"""Recover a real shared-initialization diagnostic using CPU and local bodies."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from level4.archive import unpack_archive
from level4.artifacts import sha256,read_json,write_json,utc_now
from level4.shared_input import checked_capture
from level4.geometry import Geometry,measure
from scripts.restore_level4_evidence import render_topview


def restore(archive,output,expected_sha256,review,figure_reference):
    output=Path(output); output.mkdir(exist_ok=False)
    verified=unpack_archive(archive,output/'evidence',expected_sha256,2*1024**3)
    evidence=output/'evidence'; audit=checked_capture(evidence)
    expected=read_json(review)['audit']
    keys=('same_saved_state_restored_exactly','actual_material_map_equal','frame0_pair_max_abs_mm',
          'unchanged_limit_mm','frame0_within_unchanged_limit','shared_source_sha256','status')
    if any(audit[k]!=expected[k] for k in keys): raise ValueError('Recovered diagnostic differs from saved review')
    with np.load(evidence/'attempt_1/logged_frame0.npz',allow_pickle=False) as p:
        particles=p['mpm_pos'][p['mpm_active'].astype(bool)].astype(float)
    geometry=Geometry(particles,.2,'points',True,True)
    spans=[measure(geometry,axis)['value'] for axis in np.eye(3)]
    independent_spans=(particles.max(0)-particles.min(0))*200
    if not np.allclose(spans,independent_spans,rtol=0,atol=1e-12):
        raise ValueError('Recovered spans differ from direct raw-particle bounding box')
    camera=render_topview(particles*200,output/'restored_shared_frame0.png',
                         'Real shared-state run 25770711 - recorded frame0')
    write_json(output/'camera.json',camera)
    if sha256(output/'restored_shared_frame0.png')!=sha256(figure_reference):
        raise ValueError('Recovered figure differs from original local diagnostic rendering')
    report={'created_utc':utc_now(),'status':'passed_real_shared_capture_CPU_restore',
        'run_id':'25770711.pbs101','archive_sha256':expected_sha256,
        'internal_verification':verified['internal_verification'],'recomputed_audit':audit,
        'audit_compared_fields':list(keys),'raw_particle_xyz_spans_mm':spans,
        'raw_summary_crosscheck':'passed_direct_bbox','figure_sha256':sha256(output/'restored_shared_frame0.png'),
        'figure_matches_original':True,'metric_representation':'raw_point_extrema_diagnostic_only',
        'task_constraints_evaluated':False,'NSCC_access_required':False,
        'full_environment_and_GPU_pipeline_restored':False,'new_simulation_run':False}
    write_json(output/'restore_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True); p.add_argument('--review',type=Path,required=True)
    p.add_argument('--figure-reference',type=Path,required=True)
    args=p.parse_args()
    import json
    print(json.dumps(restore(args.archive,args.output,args.expected_sha256,args.review,args.figure_reference)))
