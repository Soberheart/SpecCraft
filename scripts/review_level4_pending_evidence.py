"""Recover a verified saved candidate diagnostic using CPU only.

No SSH, simulator, model, selection or task-threshold fitting is invoked.
Pickles are read only after their enclosing evidence package is hash verified.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from level4.artifacts import read_json,write_json,sha256,utc_now
from level4.archive import unpack_archive,verify_zip
from level4.candidate_interface import audit_candidate,diagnose_tool_schedule
from level4.geometry import Geometry,measure
from level4.initial_frame import assert_same_frame
from level4.shared_input import exact_numeric_packet


def recover(collection,output,expected_sha256,plot_dependencies=None):
    collection=Path(collection).resolve();output=Path(output).resolve()
    header=read_json(collection/'transfer_header.json')
    if header['job']!='25789650.pbs101' or 'job_state = F' not in header['qstat']:
        raise ValueError('Expected the completed historical diagnostic job25789650')
    receipt=read_json(collection.parent/'submission.json')
    if receipt['job']!=header['job'] or receipt['remote_parent']!=header['remote_parent']:
        raise ValueError('Submission and collection identities differ')
    for row in header['files']:
        name=row['name']
        from level4.archive import safe_name
        safe_name(name);path=collection/name
        if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Transferred evidence differs: '+name)
    archive=collection/'diagnostic_evidence.zip';internal=verify_zip(archive)
    output.mkdir(parents=True,exist_ok=False)
    unpack=unpack_archive(archive,output/'evidence',expected_sha256,max_bytes=512*1024**2)
    evidence=output/'evidence';candidate=evidence/'execution/candidate_001';reference=evidence/'nominal_reference'
    if read_json(evidence/'stage_exit.json')['exit_code']!=1 or read_json(evidence/'candidate_001_exit.json')['exit_code']!=0:
        raise ValueError('Historical exit states changed')
    if sha256(evidence/'submitted_toolkit.zip')!=receipt['toolkit_sha256']:
        raise ValueError('Frozen source delivery differs')
    source=evidence/'source_pool_manifest.json';pool=read_json(source);request=read_json(evidence/'request.json')
    if sha256(source)!=request['source_pool_sha256']:raise ValueError('Frozen pool identity changed')
    action=next(row for row in pool['candidates'] if row['candidate_id']=='candidate_001')
    # For this specific frozen source scene, the nominal grasp center is
    # (0,0,0). Verify the first nominal ee center; do not infer another scene's
    # grasp center from this job-specific recovery entry point.
    with (reference/'legacy_log.pkl').open('rb') as stream:legacy=pickle.load(stream)
    center=np.zeros(3)
    if not np.array_equal(legacy[0]['ee']['pos'],center):raise ValueError('Frozen nominal center changed')
    exact_numeric_packet(reference/'reset_returned.npz',candidate/'reset_returned.npz')
    exact_numeric_packet(reference/'static_material_info.npz',candidate/'static_material_info.npz')
    with np.load(candidate/'after_first_action.npz',allow_pickle=False) as first,np.load(reference/'reset_returned.npz',allow_pickle=False) as initial:
        for key in ('mpm_pos','mpm_vel','mpm_C','mpm_F','mpm_active','tool_000_pos','tool_000_quat','tool_000_vel','tool_000_ang'):
            if not np.array_equal(first[key],initial[key]):raise ValueError('First action altered material or ground')
    with (candidate/'scenes/0000/log.pkl').open('rb') as stream:actual=pickle.load(stream)
    if len(actual)!=63:raise ValueError('Incomplete material log')
    for frame,row in enumerate(actual):
        with (candidate/'raw_frames'/('frame_%03d.pkl'%frame)).open('rb') as stream:raw=pickle.load(stream)
        assert_same_frame(row,raw)
    try:
        audit_candidate(candidate,reference/'legacy_log.pkl',reference/'reset_returned.npz',reference/'static_material_info.npz',
            reference/'original_planner_input.npz',center,action['parameters_mm_mm_deg'],physical_reference=reference/'actual_tools')
    except ValueError as exc:strict_rejection=str(exc)
    else:raise ValueError('Historical candidate unexpectedly passed the strict interface gate')
    diagnosed=diagnose_tool_schedule(candidate,reference/'legacy_log.pkl',reference/'actual_tools',center,action['parameters_mm_mm_deg'])
    diagnosed.update(created_utc=utc_now(),source_job=header['job'],source_job_status='failed',candidate_simulator_exit_code=0)
    write_json(output/'tool_diagnosis.json',diagnosed)
    with (output/'tool_errors.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(diagnosed['records'][0]));writer.writeheader();writer.writerows(diagnosed['records'])
    spans=[]
    axes=pool['config']['axes']
    for frame,row in enumerate(actual):
        geometry=Geometry(row['obj']['particles']['pos'],.2,'points',True,True)
        measured=[measure(geometry,axes[key]) for key in ('u','v')]
        if any(m['status']!='measured' for m in measured):raise ValueError('Raw material point geometry could not be measured')
        spans.append({'frame':frame,'nominal_label_s':float(row['t']),
            'raw_particle_center_u_mm':measured[0]['value'],'raw_particle_center_v_mm':measured[1]['value']})
    with (output/'raw_material_spans.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(spans[0]));writer.writeheader();writer.writerows(spans)
    with np.load(reference/'original_planner_input.npz',allow_pickle=False) as initial:
        initial_error=float(np.max(np.abs(actual[0]['obj']['particles']['pos'].astype(float)-initial['initial_particles'].astype(float)))*200)
    if initial_error>.001:raise ValueError('Initial reset limit exceeded')
    primitive={'kind':'diagnostic_raw_particle_center_extents_not_primary_voxel_measurement',
        'axes':axes,'unit':'mm','operator':'extrema_particle_centers','unit_to_meter':.2,
        'frame0':spans[0],'primary_frame60':spans[60],'auxiliary_frame62':spans[62],
        'physical_elapsed_time':'not_certified','task_satisfaction':'not_evaluated_uncalibrated',
        'radius_added':False,'smoothing_or_hole_filling':False,'calibration_changed':False}
    write_json(output/'raw_material_summary.json',primitive)
    if plot_dependencies is not None:sys.path.insert(0,str(Path(plot_dependencies).resolve()))
    os.environ['MPLCONFIGDIR']=str(output/'plot_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,panels=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True,layout='constrained')
    for panel,frame in zip(panels,(0,60)):
        points=actual[frame]['obj']['particles']['pos'].astype(float)*200
        panel.scatter(points[:,0],points[:,1],s=1.4,color='#007f71',rasterized=True)
        panel.set_title('candidate_001, frame%d (label %.3fs)'%(frame,actual[frame]['t']))
        panel.set_xlabel('world x (mm)');panel.set_ylabel('world y (mm)');panel.set_aspect('equal');panel.set_xlim(-50,50);panel.set_ylim(-50,50)
    figure.suptitle('Independent MPM raw particles; CPU recovery; no task success claim',fontsize=10)
    figure.savefig(output/'real_material_frame0_frame60.png',dpi=160);figure.savefig(output/'real_material_frame0_frame60.pdf');plt.close(figure)
    figure,panels=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
    frames=np.arange(63);rows=diagnosed['records']
    panels[0].plot(frames,[r['finger_1_physical_transform_error_mm'] for r in rows],label='actual left finger vs nominal action transform')
    panels[0].plot(frames,[r['summary_opening_error_mm'] for r in rows],label='logged opening vs nominal opening')
    panels[0].axhline(.001,color='black',ls='--',label='original diagnostic limit (0.001 mm)')
    panels[0].set_ylabel('Error (mm)');panels[0].legend(fontsize=8)
    panels[1].plot(frames,[r['global_solver_counter'] for r in rows],label='saved resettable solver counter')
    panels[1].plot(frames,[r['no_reset_counter_reference'] for r in rows],ls='--',label='counter if never restored')
    panels[1].set_ylabel('Solver substep counter');panels[1].legend(fontsize=8)
    panels[2].plot(frames,[r['raw_particle_center_u_mm'] for r in spans],label='raw particle centers: u')
    panels[2].plot(frames,[r['raw_particle_center_v_mm'] for r in spans],label='raw particle centers: v')
    panels[2].set_ylabel('Extent (mm)');panels[2].set_xlabel('Saved frame index (elapsed physical time not certified)');panels[2].legend(fontsize=8)
    for panel in panels:panel.axvline(60,color='gray',ls=':');panel.grid(alpha=.2)
    figure.suptitle('Saved candidate_001 diagnostic; no threshold fitting or primary-time change',fontsize=10)
    figure.savefig(output/'interface_and_clock_diagnostic.png',dpi=160);figure.savefig(output/'interface_and_clock_diagnostic.pdf');plt.close(figure)
    write_json(output/'camera.json',{'projection':'world_xy','bounds_mm':[-50,50,-50,50],'frames':[0,60],
        'unit_to_meter':.2,'display_geometry':'raw_material_particle_centers','no_smoothing':True})
    write_json(output/'restore_command.json',{'command':[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),
        '--collection',str(collection),'--expected-sha256',expected_sha256,'--output','NEW_EMPTY_RESTORE_DIRECTORY',
        '--plot-dependencies',str(plot_dependencies)],'cwd':str(REPO),'new_simulations':0})
    write_json(output/'restoration_environment.json',{'python':sys.version,'numpy':np.__version__,
        'plot_packages':{p:importlib.metadata.version(p) for p in ('matplotlib','contourpy','cycler','kiwisolver','pyparsing','fonttools','pillow','python-dateutil','packaging')},
        'server_environment_changed':False})
    result={'created_utc':utc_now(),'status':'passed_real_candidate001_CPU_data_restore_original_job_remains_failed',
        'job':header['job'],'original_job_exit_code':1,'candidate_simulator_exit_code':0,
        'candidate_simulator_elapsed_s':read_json(evidence/'candidate_001_exit.json')['elapsed_s'],
        'transfer_files_verified':len(header['files']),'internal_verification':internal,'unpack_verification':unpack,
        'complete_reset_and_static_material_exact':True,'first_action_material_and_ground_exact':True,
        'logged0_material_max_abs_error_mm':initial_error,'original_reset_limit_mm_unchanged':.001,
        'raw63frames_all_nested_fields_dtype_shape_value_identical':True,
        'strict_interface_rejected_after_clock_error_fix':strict_rejection,
        'tool_transform_status':diagnosed['tool_transform_status'],'elapsed_physical_time':'unknown',
        'legacy_model_tool_input_alignment':'not_certified','task_satisfaction':'not_evaluated_uncalibrated',
        'candidate_pool_released':False,'original_NSCC_paths_used':False,'new_simulations':0,'model_prediction_run':False,
        'full_GPU_environment_restore':'pending','archive_sha256':expected_sha256,
        'output_hashes':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'restore_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--plot-dependencies',type=Path)
    args=parser.parse_args()
    report=recover(args.collection,args.output,args.expected_sha256,args.plot_dependencies)
    print(json.dumps({'status':report['status'],'internal_files':report['internal_verification']['files'],
        'tool_transform_status':report['tool_transform_status'],'clock':'unknown','new_simulations':0}))
