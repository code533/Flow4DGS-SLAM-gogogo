import unittest
import numpy as np
from utils.camera_motion_v32b import (exact_backward_gt_flow, gt_camera_twist,
    jacobian_backward_flow, projected_depth_mask, flow_scores,
    remap_depth_to_rectified)


class TestV32b(unittest.TestCase):
    def setUp(self):
        self.h,self.w=20,30
        self.depth=np.full((self.h,self.w),2.,dtype=float)
        self.K=np.array([[100.,0,15.],[0,100.,10.],[0,0,1.]])
        self.prev=np.eye(4)
        self.cur=np.eye(4)
        self.cur[0,3]=.01

    def test_backward_sign(self):
        flow,_,_,valid=exact_backward_gt_flow(self.depth,self.K,self.cur,self.prev)
        self.assertAlmostEqual(float(np.nanmedian(flow[...,0])),-.5,places=8)
        self.assertTrue(valid[10,15])

    def test_inverse_sign(self):
        flow,*_=exact_backward_gt_flow(self.depth,self.K,self.prev,self.cur)
        self.assertAlmostEqual(float(np.nanmedian(flow[...,0])),.5,places=8)

    def test_identity(self):
        f,*_=exact_backward_gt_flow(self.depth,self.K,self.prev,self.prev)
        self.assertAlmostEqual(float(np.nanmax(np.abs(f))),0.,places=10)

    def test_invalid_depth(self):
        d=self.depth.copy();d[10,15]=0.
        f,_,_,valid=exact_backward_gt_flow(d,self.K,self.cur,self.prev)
        self.assertFalse(valid[10,15])
        self.assertTrue(np.isnan(f[10,15]).all())

    def test_outside(self):
        T=self.cur.copy();T[0,3]=2.
        _,_,_,valid=exact_backward_gt_flow(self.depth,self.K,T,self.prev)
        self.assertEqual(int(valid.sum()),0)

    def test_gt_twist(self):
        x=gt_camera_twist(self.cur,self.prev)
        self.assertAlmostEqual(x[0],.01,places=10)
        self.assertAlmostEqual(np.linalg.norm(x[3:]),0.,places=10)

    def test_jacobian_sign(self):
        x=gt_camera_twist(self.cur,self.prev)
        f=jacobian_backward_flow(self.depth,self.K,x)
        self.assertAlmostEqual(float(f[10,15,0]),-.5,places=8)
        self.assertAlmostEqual(float(jacobian_backward_flow(self.depth,self.K,-x)[10,15,0]),.5,places=8)

    def test_projected_depth_consistency(self):
        _,zp,uv,v=exact_backward_gt_flow(self.depth,self.K,self.cur,self.prev)
        ok=projected_depth_mask(self.depth,uv,zp,v,abs_tol=.01,rel_tol=0.)
        self.assertTrue(ok[10,15])
        self.assertGreater(ok.sum(),0)

    def test_depth_occlusion_rejected(self):
        _,zp,uv,v=exact_backward_gt_flow(self.depth,self.K,self.cur,self.prev)
        prev_depth=np.ones_like(self.depth)
        ok=projected_depth_mask(prev_depth,uv,zp,v,abs_tol=.01,rel_tol=0.)
        self.assertEqual(int(ok.sum()),0)

    def test_depth_rectification_nearest_neighbor(self):
        depth=np.arange(12,dtype=np.float32).reshape(3,4)
        mx,my=np.meshgrid(np.arange(4,dtype=np.float32),
                          np.arange(3,dtype=np.float32))
        mx-=1
        out=remap_depth_to_rectified(depth,mx,my)
        self.assertEqual(float(out[1,2]),float(depth[1,1]))
        self.assertEqual(float(out[1,0]),0.0)

    def test_score_perfect(self):
        a=np.ones((4,5,2));m=np.ones((4,5),dtype=bool)
        r=flow_scores(a,a,m)
        self.assertEqual(r['count'],20)
        self.assertAlmostEqual(r['rmse_epe_px'],0.)
        self.assertAlmostEqual(r['mean_direction_deg'],0.,delta=1e-7)

    def test_score_opposite(self):
        a=np.ones((4,5,2));m=np.ones((4,5),dtype=bool)
        r=flow_scores(-a,a,m)
        self.assertAlmostEqual(r['negative_cosine_fraction'],1.)
        self.assertAlmostEqual(r['mean_direction_deg'],180.,delta=1e-7)

    def test_score_empty(self):
        a=np.ones((4,5,2));m=np.zeros((4,5),dtype=bool)
        r=flow_scores(a,a,m)
        self.assertEqual(r['count'],0)
        self.assertTrue(np.isnan(r['mean_epe_px']))

    def test_score_nan_guard(self):
        a=np.ones((4,5,2));a[0,0]=np.nan
        r=flow_scores(a,np.ones_like(a),np.ones((4,5),bool))
        self.assertEqual(r['count'],19)

    def test_negative_tolerance(self):
        _,zp,uv,v=exact_backward_gt_flow(self.depth,self.K,self.cur,self.prev)
        with self.assertRaises(ValueError):
            projected_depth_mask(self.depth,uv,zp,v,abs_tol=-1)

    def test_invalid_intrinsics(self):
        K=self.K.copy();K[0,0]=0
        with self.assertRaises(ValueError):
            exact_backward_gt_flow(self.depth,K,self.cur,self.prev)

    def test_gt_rotation_twist_finite(self):
        from scipy.spatial.transform import Rotation
        T=self.cur.copy();T[:3,:3]=Rotation.from_euler('xyz',[.01,-.02,.03]).as_matrix()
        x=gt_camera_twist(T,self.prev)
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(np.linalg.norm(x[3:]),np.linalg.norm(Rotation.from_matrix(T[:3,:3]).as_rotvec()),places=8)


if __name__=="__main__":
    unittest.main()
