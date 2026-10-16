import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np
from level4.artifacts import read_json,write_json,sha256
from level4.shared_input import checked_capture,checked_initial_input
from level4.pipeline import prepare,execute
from tests.test_level4_shared_reset import make_packet


def initial_fixture(root):
    source=root/'input'; source.mkdir()
    packet=make_packet(); packet['mpm_pos']=np.array([[-.02,0,.01],[.02,0,.01]],dtype=np.float32)
    packet['mpm_active']=np.ones(2,dtype=np.int32)
    np.savez(source/'shared_source_state.npz',**packet)
    np.savez(source/'logged_frame0.npz',**packet)
    np.savez(source/'static_material_info.npz',mat_idx=np.zeros(2,dtype=np.int32),mu=np.ones(2,dtype=np.float32))
    np.savez(source/'observations.npz',observed_initial=np.array([[0,0,.01,1],[.01,0,.01,1]],dtype=np.float32),
             z_initial=np.ones((2,3),dtype=np.float32),z_goal=np.zeros((2,3),dtype=np.float32),
             tools=np.zeros((13,4,3),dtype=np.float32),times_s=np.arange(13)*.16,initial_particles=packet['mpm_pos'])
    write_json(source/'initialization_evidence.json',{'status':'passed','same_saved_state_restored_exactly':True,
        'actual_material_map_equal':True,'recomputed_from_body':True,'unchanged_limit_mm':.001,
        'frame0_within_unchanged_limit':True,'candidate_pool_released':False,
        'shared_source_sha256':sha256(source/'shared_source_state.npz'),'test_data_kind':'synthetic_only'})
    (source/'model_config.yaml').write_text('{}', encoding='utf-8')
    scene=root/'scene'; scene.mkdir()
    write_json(scene/'config.yaml',{'ee':{'type':'gripper'},'actions':{'grasp':{'type':'grasp','from_offset':[0,0,0],
        'to_pos':[0,0,0],'to_quat':[1,0,0,0]}},'log':{'n_iter':16},'sim':{'step_dt':.002,'seed':0}})
    checkpoint=root/'weights';checkpoint.write_bytes(b'synthetic_frozen_weights')
    metadata={'kind':'shared_initial_observation_input_v1','scene_dir':str(scene),'scene_config_sha256':sha256(scene/'config.yaml'),
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'seed':0,'unit_to_meter':.2,
        'rollout_frames':list(range(0,61,5)),'execution_required_s':(np.arange(61)*.032).tolist(),
        'historical_initial_observation_reused':False,
        'input_sha256':{p.name:sha256(p) for p in source.iterdir()}}
    write_json(source/'metadata.json',metadata)
    return source,metadata


class SharedInputTests(unittest.TestCase):
    def test_whitelisted_initial_input_contains_no_future_material(self):
        with tempfile.TemporaryDirectory() as d:
            source,meta=initial_fixture(Path(d)); result=checked_initial_input(source,meta)
            self.assertEqual(set(result),{'observed_initial','z_initial','z_goal','tools','times_s','initial_particles'})
            self.assertEqual(result['times_s'][-1],1.92)
            self.assertNotIn('log.pkl',{p.name for p in source.iterdir()})

    def change_packet(self,source,meta,change):
        with np.load(source/'observations.npz',allow_pickle=False) as p: values={k:p[k] for k in p.files}
        change(values); np.savez(source/'observations.npz',**values)
        meta['input_sha256']['observations.npz']=sha256(source/'observations.npz')

    def test_future_object_payload_and_old_observation_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source,meta=initial_fixture(Path(d)); meta['historical_initial_observation_reused']=True
            with self.assertRaises(ValueError): checked_initial_input(source,meta)
            meta['historical_initial_observation_reused']=False
            self.change_packet(source,meta,lambda v:v.update(future_material=np.zeros((10,3))))
            with self.assertRaises(ValueError): checked_initial_input(source,meta)

    def test_reference_mismatch_rejected_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as d:
            source,meta=initial_fixture(Path(d))
            self.change_packet(source,meta,lambda v:v['initial_particles'].__setitem__((0,0),99))
            with self.assertRaisesRegex(ValueError,'Initial reference'): checked_initial_input(source,meta)

    def test_changed_input_and_missing_complete_state_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source,meta=initial_fixture(Path(d)); (source/'shared_source_state.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'): checked_initial_input(source,meta)

    def test_main_time_or_dense_schedule_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            source,meta=initial_fixture(Path(d)); meta['execution_required_s'][-1]=1.984
            with self.assertRaisesRegex(ValueError,'dense physical-time'): checked_initial_input(source,meta)
            meta['execution_required_s'][-1]=1.92
            self.change_packet(source,meta,lambda v:v['times_s'].__setitem__(-1,1.984))
            with self.assertRaisesRegex(ValueError,'action/time'): checked_initial_input(source,meta)

    def test_latent_shape_or_nonfinite_observation_rejected(self):
        for change in (lambda v:v.update(z_goal=np.ones((4,3))),
                       lambda v:v['observed_initial'].__setitem__((0,0),np.nan)):
            with tempfile.TemporaryDirectory() as d:
                source,meta=initial_fixture(Path(d)); self.change_packet(source,meta,change)
                with self.assertRaises(ValueError): checked_initial_input(source,meta)

    def test_empty_inputs_and_wrong_intermediate_time_rejected(self):
        for change in (lambda v:v.update(times_s=np.array([])),
                       lambda v:v.update(observed_initial=np.empty((0,4))),
                       lambda v:v.update(z_initial=np.empty((0,3)),z_goal=np.empty((0,3))),
                       lambda v:v['times_s'].__setitem__(1,.2)):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                source,meta=initial_fixture(Path(d)); self.change_packet(source,meta,change)
                with self.assertRaises(ValueError): checked_initial_input(source,meta)

    def test_capture_acceptance_recomputed_from_state_not_only_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); packet=make_packet(); np.savez(root/'shared_source_state.npz',**packet)
            for i in (1,2):
                sub=root/('attempt_%d'%i); sub.mkdir(); np.savez(sub/'reset_returned.npz',**packet)
                np.savez(sub/'logged_frame0.npz',**packet)
                np.savez(sub/'static_material_info.npz',mat_idx=np.zeros(2,dtype=np.int32),mu=np.ones(2,dtype=np.float32))
                write_json(root/('attempt_%d_exit.json'%i),{'exit_code':0})
            write_json(root/'shared_initialization_audit.json',{'kind':'shared_state_initialization_audit_not_pool_gate',
                'status':'passed','unchanged_limit_mm':.001,'candidate_pool_released':False,
                'shared_source_sha256':sha256(root/'shared_source_state.npz'),'frame0_pair_max_abs_mm':0})
            def manifest():
                p=root/'evidence_manifest.json'
                if p.exists(): p.write_text(json.dumps({'files_sha256':{x.relative_to(root).as_posix():sha256(x) for x in root.rglob('*') if x.is_file() and x!=p}}), encoding='utf-8')
                else: write_json(p,{'files_sha256':{x.relative_to(root).as_posix():sha256(x) for x in root.rglob('*') if x.is_file()}})
            manifest(); self.assertTrue(checked_capture(root)['recomputed_from_body'])
            packet['mpm_C'][0,0,0]+=1; np.savez(root/'attempt_2/reset_returned.npz',**packet); manifest()
            with self.assertRaisesRegex(ValueError,'exactly equal'): checked_capture(root)

    def test_prepare_new_input_without_opening_future_raw_log(self):
        # OmegaConf is absent on this CPU host. Only config parsing/snapshot
        # are substituted; real geometry/input/candidate code executes.
        class Config(dict):
            def __getattr__(self,key):
                try: return self[key]
                except KeyError: raise AttributeError(key)
            __setattr__=dict.__setitem__
        def convert(v):
            return Config({k:convert(x) for k,x in v.items()}) if isinstance(v,dict) else v
        class Omega:
            load=staticmethod(lambda p:convert(read_json(p)))
            resolve=staticmethod(lambda c:None)
            create=staticmethod(lambda c:convert(copy.deepcopy(c)))
            to_container=staticmethod(lambda c,resolve:copy.deepcopy(c))
            save=staticmethod(lambda c,p:write_json(p,c))
        fake=ModuleType('omegaconf'); fake.OmegaConf=Omega
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source,meta=initial_fixture(root)
            # Any accidental old-source read fails immediately.
            (root/'scene/log.pkl').write_bytes(b'future data must not be loaded')
            cfg=read_json(Path(__file__).resolve().parents[1]/'configs/level4/development_pool.json')
            cfg['candidate_count']=2; cfg['grid']['initial_padding_mm']=[3,3,3]
            config=root/'config.json'; write_json(config,cfg)
            def snapshot(runtime,out):
                write_json(out/'manifest.json',{'files_sha256':{}}); return sha256(out/'manifest.json')
            with patch.dict('sys.modules',{'omegaconf':fake}),patch('level4.pipeline.snapshot_runtime',side_effect=snapshot),patch('level4.pipeline.runtime_commit',return_value='a'*40),patch('level4.pipeline.ROOT',root):
                result=prepare(source,root/'pool',config)
            self.assertEqual(result['candidate_count'],2)
            pool=read_json(root/'pool/pool.json')
            self.assertEqual(pool['initialization_mode'],'shared_saved_state')
            self.assertIn('shared_source_state.npz',pool['input_sha256'])
            with np.load(root/'pool/planner_input.npz',allow_pickle=False) as p:
                np.testing.assert_array_equal(p['initial_particles'],np.array([[-.02,0,.01],[.02,0,.01]],dtype=np.float32))
            # Preparation never constitutes candidate restore acceptance.
            with self.assertRaisesRegex(ValueError,'not certified'): execute(root/'pool',candidate_ids=['candidate_000'])

    def test_initial_preparation_failure_bundle_preserves_exit_and_source(self):
        from scripts.prepare_level4_shared_input import collect
        from level4.archive import verify_zip
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); request=root/'request.json'; write_json(request,{'kind':'synthetic_failure_test_only'})
            with patch('scripts.prepare_level4_shared_input.run',side_effect=ValueError('fixture failure')),patch('sys.stdout',io.StringIO()):
                self.assertEqual(collect(request,root/'data'),1)
            self.assertEqual(read_json(root/'data/stage_exit.json')['exit_code'],1)
            self.assertFalse(read_json(root/'data/stage_exit.json')['new_simulation_run'])
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def runtime_fixture(self,root):
        original=root/'old_runtime'; original.mkdir()
        (original/'level4').mkdir(); (original/'level4/pipeline.py').write_bytes(b'old source\n')
        (original/'assets').mkdir(); (original/'assets/board.obj').write_bytes(b'frozen asset\n')
        from level4.artifacts import runtime_environment
        manifest=root/'old_manifest.json'
        write_json(manifest,{'kind':'sealed_submission_runtime','code_commit':'a'*40,
            'environment':runtime_environment(),
            'files_sha256':{p.relative_to(original).as_posix():sha256(p) for p in original.rglob('*') if p.is_file()}})
        tools=root/'tools'; tools.mkdir(); (tools/'overlay.py').write_bytes(b'# new Level4 source\n')
        parent=root/'new_run'; parent.mkdir()
        request={'runtime':str(original),'runtime_manifest':str(manifest),
                 'runtime_overlays':{'level4/pipeline.py':'overlay.py'}}
        return request,tools,parent

    def test_runtime_copy_preserves_original_source_and_assets(self):
        from scripts.prepare_level4_shared_input import assemble_runtime
        from level4.artifacts import runtime_commit
        import sys
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); request,tools,parent=self.runtime_fixture(root)
            originals=read_json(request['runtime_manifest'])['files_sha256']
            old_cwd=Path.cwd(); old_path=sys.path[:]
            try:
                with patch.dict(os.environ):
                    files=assemble_runtime(request,tools,parent)
                    self.assertEqual(runtime_commit(files),'a'*40)
                    self.assertEqual((files/'level4/pipeline.py').read_bytes(),b'# new Level4 source\n')
                    self.assertEqual((files.parent/'original_pipeline.py').read_bytes(),b'old source\n')
                    self.assertEqual(sha256(files/'assets/board.obj'),originals['assets/board.obj'])
            finally:
                os.chdir(old_cwd); sys.path[:]=old_path
            for name,h in originals.items(): self.assertEqual(sha256(Path(request['runtime'])/name),h)

    def test_runtime_changed_source_cannot_be_used(self):
        from scripts.prepare_level4_shared_input import assemble_runtime
        with tempfile.TemporaryDirectory() as d:
            request,tools,parent=self.runtime_fixture(Path(d))
            source=Path(request['runtime'])/'assets/board.obj'; source.write_bytes(b'changed asset')
            with self.assertRaisesRegex(ValueError,'Runtime source changed'):
                assemble_runtime(request,tools,parent)
            self.assertEqual(source.read_bytes(),b'changed asset')
            self.assertTrue((parent/'runtime').exists())  # failed attempt is retained

    def test_runtime_unsafe_source_or_overlay_cannot_escape(self):
        from scripts.prepare_level4_shared_input import assemble_runtime
        for mode in ('source','overlay'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d); request,tools,parent=self.runtime_fixture(root)
                if mode=='source':
                    p=Path(request['runtime_manifest']); value=read_json(p)
                    value['files_sha256']['../escape.py']='0'*64
                    p.write_text(json.dumps(value), encoding='utf-8')
                else: request['runtime_overlays']={'level4/../../escape.py':'overlay.py'}
                with self.assertRaisesRegex(ValueError,'Unexpected'):
                    assemble_runtime(request,tools,parent)
                self.assertFalse((root/'escape.py').exists())


if __name__=='__main__': unittest.main()
