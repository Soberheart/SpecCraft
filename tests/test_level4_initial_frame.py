import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from level4.initial_frame import save_frame,load_frame,assert_same_frame,validate_material_reference,digest
from scripts.prepare_level4_shared_input import validate_preprocess_input


def fixture():
    return {'step':0,'t':0.,'frame_idx':0,
        'obj':{'particles':{'pos':np.array([[.01,.02,.03],[.04,.05,.06]],dtype=np.float32),
                           'idx':np.array([0,0],dtype=np.int64),'colliding':np.zeros(2,dtype=np.float32)}},
        'ee':{'pos':np.array([0.,0.,1.],dtype=np.float32),'open':0.,'is_gripper':True},
        'topology':{'components':1,'genus':[1]},
        'test_misc':(None,np.float32(.5),np.int32(7),False,'literal $() text',np.array(2.))}


class InitialFrameTests(unittest.TestCase):
    def packet(self,root,frame=None):
        frame=fixture() if frame is None else frame
        arrays=root/'frame.npz'; meta=root/'frame.json'
        save_frame(frame,arrays,meta,{'data_kind':'synthetic_test_only'})
        return frame,arrays,meta

    def test_every_field_dtype_and_type_survives_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            frame,a,m=self.packet(Path(d)); decoded=load_frame(a,m)
            assert_same_frame(frame,decoded)
            with np.load(a,allow_pickle=False) as p:
                self.assertTrue(all(p[k].dtype.kind in 'biuf' for k in p.files))
            self.assertNotIn('numpy._core',m.read_text(encoding='utf-8'))

    def test_exact_reference_check_and_native_preprocessing_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); frame,a,m=self.packet(root)
            state=root/'state.npz'; np.savez(state,mpm_pos=frame['obj']['particles']['pos'],mpm_active=np.ones(2))
            self.assertTrue(validate_material_reference(load_frame(a,m),state)['logged_frame0_material_matches_exactly'])
            assert_same_frame(frame,pickle.loads(pickle.dumps([load_frame(a,m)],protocol=4))[0])
            wrong=fixture(); wrong['obj']['particles']['pos'][0,0]+=.001
            with self.assertRaisesRegex(ValueError,'differ'): validate_material_reference(wrong,state)

    def test_nonfinite_object_or_arbitrary_class_rejected(self):
        for value in (np.array([np.nan]),np.array([object()],dtype=object),object(),float('inf')):
            with self.subTest(value=type(value)),tempfile.TemporaryDirectory() as d:
                frame=fixture();frame['bad']=value
                with self.assertRaises(ValueError): self.packet(Path(d),frame)

    def test_changed_npz_or_descriptor_rejected(self):
        for mode in ('hash','dtype','shape','future','duplicate','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                _,a,m=self.packet(Path(d));meta=json.loads(m.read_text(encoding='utf-8'))
                def first_array(node):
                    if node['type']=='array': return node
                    items=node.get('items',[])
                    for item in items:
                        got=first_array(item[1] if node['type']=='dict' else item)
                        if got is not None: return got
                node=first_array(meta['frame_tree'])
                if mode=='hash': a.write_bytes(b'changed')
                elif mode=='dtype': node['dtype']='<f8'
                elif mode=='shape': node['shape']=[3,2]
                elif mode=='missing': node['name']='missing_array'
                elif mode=='duplicate': meta['frame_tree']['items'].append(['reused',node.copy()])
                else:
                    with np.load(a,allow_pickle=False) as p: values={k:p[k] for k in p.files}
                    values['future_material']=np.ones((5,3));np.savez(a,**values);meta['arrays_sha256']=digest(a)
                m.write_text(json.dumps(meta), encoding='utf-8')
                with self.assertRaises(ValueError): load_frame(a,m)

    def test_noninitial_frame_rejected(self):
        for key,value in (('step',16),('t',.032),('frame_idx',1)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                frame=fixture();frame[key]=value;_,a,m=self.packet(Path(d),frame)
                with self.assertRaisesRegex(ValueError,'frame0'):load_frame(a,m)

    def test_CPU_preflight_uses_no_GPU_or_model_and_native_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frame=fixture()
            save_frame(frame,root/'nominal_first_frame.npz',root/'nominal_first_frame.json',{'data_kind':'synthetic_test_only'})
            source=Path(__file__).resolve().parents[1]/'level4/initial_frame.py'
            (root/'initial_frame_source.py').write_bytes(source.read_bytes())
            np.savez(root/'logged_frame0.npz',mpm_pos=frame['obj']['particles']['pos'],mpm_active=np.ones(2))
            names=('initial_frame_source.py','nominal_first_frame.npz','nominal_first_frame.json','logged_frame0.npz')
            (root/'tool_manifest.json').write_text(json.dumps({'files_sha256':{n:digest(root/n) for n in names}}), encoding='utf-8')
            checked=validate_preprocess_input(root,root/'report.json');assert_same_frame(frame,checked)
            report=json.loads((root/'report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['all_frame_fields_roundtrip_exactly'])
            self.assertFalse(report['new_simulation_run']);self.assertFalse(report['model_prediction_run'])


if __name__=='__main__': unittest.main()
