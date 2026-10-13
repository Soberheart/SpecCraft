"""Analytic geometry expectations; no model, simulator, or CUDA imports."""
import unittest
import numpy as np
from level4.geometry import Geometry,measure,from_mesh,grid_coordinates,from_occupancy,particle_occupancy


def box(span=(100.,60.,40.),shift=(0.,0.,0.),scale=.001):
    # Eight corner cells: exact axis extrema of their union are the given spans.
    points=np.array(np.meshgrid(*[[-(s-1)/2,(s-1)/2] for s in span],indexing="ij")).reshape(3,-1).T
    points=(points+shift)/ (1000*scale)
    return Geometry(points,scale,"voxel_union",True,True,np.full(3,1/(1000*scale)))


class GeometryTests(unittest.TestCase):
    def test_sphere_extrema_axis_sign_and_single_voxel(self):
        g=Geometry(np.vstack([np.eye(3),-np.eye(3)])*7,.001,"points",True,True)
        for axis in (np.array([1,0,0]),np.array([-1,0,0])):
            self.assertEqual(measure(g,axis)["value"],14)
        q,n=grid_coordinates([[-1,1]]*3,[1,1,1])
        g=from_occupancy(q,(q==0).all(axis=1),n,[1,1,1],.001)
        self.assertEqual(measure(g,[1,0,0])["value"],1)
    def test_raw_particle_ball_occupancy_keeps_contact_and_has_known_radius(self):
        query=np.array([[0,0,0],[1,0,0],[1.01,0,0],[10,0,0]])
        # No tool mask is accepted: the material at a contact location is retained.
        np.testing.assert_array_equal(particle_occupancy(query,[[0,0,0]],1),[1,1,0,0])
    def test_known_cube_and_translation(self):
        for shift in ((0,0,0),(123,-55,8)):
            g=box(shift=shift)
            for axis,expected in zip(np.eye(3),(100,60,40)):
                self.assertAlmostEqual(measure(g,axis)["value"],expected)

    def test_units_from_model_config(self):
        g=box(scale=.2)
        self.assertAlmostEqual(measure(g,[1,0,0])["value"],100.)
        g.unit_to_meter=1.
        self.assertAlmostEqual(measure(g,[1,0,0])["value"],500.)

    def test_fixed_axis_rotation_changes_projection(self):
        g=Geometry(np.array([[-50,-10,0],[-50,10,0],[50,-10,0],[50,10,0]]),.001,"points",True,True)
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(measure(g,[1,0,0])["value"],100)
        g.points=g.points@rotation.T
        # Pure rotation can mimic world-x narrowing. No deformation claim allowed.
        self.assertAlmostEqual(measure(g,[1,0,0])["value"],20)
        self.assertAlmostEqual(measure(g,[0,1,0])["value"],100)

    def test_axis_basis_change_and_cell_support(self):
        g=box(span=(10,10,10))
        u=np.array([1,1,0])/np.sqrt(2)
        self.assertAlmostEqual(measure(g,u)["value"],10*np.sqrt(2))
        self.assertEqual(measure(g,[2,0,0])["status"],"unknown")

    def test_empty_nan_and_zero_span(self):
        for p in (np.empty((0,3)),np.array([[np.nan,0,0]]),np.zeros((3,2))):
            self.assertEqual(measure(Geometry(p,.001,"points",True,True),[1,0,0])["status"],"unknown")
        g=Geometry(np.zeros((3,3)),.001,"points",True,True)
        self.assertEqual(measure(g,[1,0,0])["value"],0)

    def test_unit_errors_contamination_and_occlusion(self):
        for scale in (0,-1,np.inf,np.nan):
            g=box();g.unit_to_meter=scale
            self.assertEqual(measure(g,[1,0,0])["status"],"unknown")
        for field in ("material_only","complete"):
            g=box();setattr(g,field,False)
            self.assertEqual(measure(g,[1,0,0])["status"],"unknown")

    def test_mesh_padding_does_not_change_extrema(self):
        vertices=np.array([[0,0,0],[10,0,0],[0,5,0],[999,999,999],[888,888,888]])
        g=from_mesh(vertices,[[0,1,2],[3,3,3]],.001,material_only=True,complete=True)
        self.assertEqual(measure(g,[1,0,0])["value"],10)
        invalid=from_mesh(vertices,[[0,1,99]],.001,material_only=True,complete=True)
        self.assertEqual(measure(invalid,[1,0,0])["status"],"unknown")

    def test_filled_grid_and_boundary_truncation(self):
        query,counts=grid_coordinates([[-2,2],[-2,2],[-2,2]],[1,1,1])
        occupied=(np.abs(query)<=1).all(axis=1)
        g=from_occupancy(query,occupied.astype(int),counts,[1,1,1],.001)
        self.assertEqual(measure(g,[1,0,0])["value"],3)
        c=occupied.astype(int);c[0]=1
        self.assertEqual(measure(from_occupancy(query,c,counts,[1,1,1],.001),[1,0,0])["status"],"unknown")
        mask=np.zeros(len(query),bool);mask[0]=True
        masked=from_occupancy(query,occupied,counts,[1,1,1],.001,ignored_mask=mask)
        self.assertEqual(measure(masked,[1,0,0])["status"],"unknown")

    def test_invalid_grid_memory_and_nonuniform_coordinates(self):
        with self.assertRaises(ValueError):
            grid_coordinates([[0,100]]*3,[.1]*3,max_cells=100)
        q,n=grid_coordinates([[-2,2]]*3,[1]*3)
        q[2,0]+=.01
        g=from_occupancy(q,np.ones(len(q)),n,[1]*3,.001)
        self.assertEqual(measure(g,[1,0,0])["status"],"unknown")

    def test_weighted_quantile_sensitivity(self):
        g=Geometry(np.array([[0,0,0],[5,0,0],[10,0,0],[999,0,0]]),.001,"points",True,True,
                   weights=np.array([1,2,1,0.001]))
        self.assertEqual(measure(g,[1,0,0])["value"],999)
        self.assertEqual(measure(g,[1,0,0],quantiles=(.1,.9))["value"],10)
        g.weights=None
        self.assertEqual(measure(g,[1,0,0],quantiles=(.1,.9))["status"],"unknown")


if __name__=="__main__":
    unittest.main()
