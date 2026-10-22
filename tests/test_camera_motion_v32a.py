import csv
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_camera_motion_v32a import (se3_exp, rigid, candidates, angle_deg,
    relative, score, geometry_check, analytic_backward_flow, reproject_backward_flow,
    analyze_frame, paired_stats, main)


class TestV32a(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prev = se3_exp(np.array([.1,.07,-.04,.1,-.15,.2]))
        self.xi = np.array([.002,-.001,.0008,.0006,-.0004,.0007])
        self.gt_prev = self.prev.copy()
        self.gt_cur = se3_exp(self.xi) @ self.prev
        self.file = self.root / 'tracking' / 'camera_motion_v31a_00017.npz'
        self.file.parent.mkdir()
        self.write()

    def write(self, status='ok', bad=False):
        L = candidates(self.prev,self.xi)
        fields=dict(status=status,frame_id=17,T_previous_est=self.prev,
          T_previous_gt=self.gt_prev,T_current_gt=self.gt_cur,
          xi_legacy_common=self.xi,xi_geo=self.xi,
          xi_legacy_actual=self.xi,
          T_candidate_legacy_common=L['right_pos'],T_candidate_geo=L['right_pos'],
          T_tracking_initial_actual=L['right_pos'])
        if bad:del fields['xi_geo']
        np.savez_compressed(self.file,**fields)

    def test_se3_small_translation(self):
        T=se3_exp([.001,-.002,.003,0,0,0])
        self.assertTrue(np.allclose(T[:3,3],[.001,-.002,.003]))

    def test_se3_inverse(self):
        self.assertTrue(np.allclose(se3_exp(self.xi)@se3_exp(-self.xi),np.eye(4),atol=1e-10))

    def test_rotation_positive(self):
        a=se3_exp([0,0,0,0,0,.4])
        self.assertAlmostEqual(a[1,0],np.sin(.4),places=9)

    def test_four_hypotheses(self):
        c=candidates(self.prev,self.xi)
        self.assertEqual(set(c),{'right_pos','left_pos','right_neg','left_neg'})
        self.assertFalse(np.allclose(c['left_pos'],c['right_pos'],atol=1e-8))

    def test_left_wins_synthetic_gt(self):
        c=candidates(self.prev,self.xi)
        errs={k:score(self.prev,v,self.gt_prev,self.gt_cur)['translation_m'] for k,v in c.items()}
        self.assertLess(errs['left_pos'],1e-8)
        self.assertLess(errs['left_pos'],errs['right_pos'])
        self.assertLess(errs['left_pos'],errs['left_neg'])

    def test_zero_previous_commutes(self):
        c=candidates(np.eye(4),self.xi)
        self.assertTrue(np.allclose(c['left_pos'],c['right_pos']))

    def test_geometry_projection_matches_left(self):
        vals=geometry_check()
        self.assertLess(vals['left_pos'], vals['right_pos']*.1)
        self.assertLess(vals['left_pos'], vals['left_neg']*.1)

    def test_backward_flow_sanity_pure_translation(self):
        xy=np.array([[320.,240.],[360.,270.]])
        z=np.array([2.,2.])
        K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
        xi=np.array([.001,0,0,0,0,0])
        f=analytic_backward_flow(xy,z,K,xi)
        self.assertTrue(np.allclose(f[:,0],-.25))
        exact=reproject_backward_flow(xy,z,K,np.eye(4),se3_exp(xi))
        self.assertTrue(np.allclose(f,exact,atol=1e-10))

    def test_pose_invalid(self):
        a=self.prev.copy();a[3,3]=2
        with self.assertRaises(ValueError):rigid(a)

    def test_relative_conventions(self):
        a=relative(self.prev,self.gt_cur,'camera')
        b=relative(self.prev,self.gt_cur,'right')
        self.assertTrue(np.allclose(a,se3_exp(self.xi)))
        self.assertFalse(np.allclose(a,b))

    def test_zero_motion_direction_nan(self):
        q=score(np.eye(4),np.eye(4),np.eye(4),np.eye(4))
        self.assertTrue(np.isnan(q['direction_deg']))

    def test_file_full(self):
        r=analyze_frame(self.file)
        self.assertEqual(r['frame_id'],17)
        self.assertLess(r['legacy_left_pos_camera_translation_m'],1e-8)
        self.assertLess(r['legacy_left_pos_camera_rotation_deg'],1e-5)
        self.assertIn('actual_clipped_camera_translation_m',r)

    def test_invalid_status(self):
        self.write(status='shadow_exception')
        with self.assertRaisesRegex(ValueError,'not ok'):analyze_frame(self.file)

    def test_missing_xi(self):
        self.write(bad=True)
        with self.assertRaisesRegex(ValueError,'Missing fields'):analyze_frame(self.file)

    def test_csv_output(self):
        output=self.root/'summary.csv'
        result=main([str(self.root),'--csv',str(output)])
        self.assertEqual(result,0)
        with output.open() as f:
            self.assertEqual(len(list(csv.DictReader(f))),1)

    def test_no_npzs(self):
        self.assertEqual(main([str(self.root/'missing')]),2)

    def test_only_geometry(self):
        self.assertEqual(main([str(self.root/'missing'),'--geometry-check-only']),0)

    def test_pair_stats(self):
        r=analyze_frame(self.file)
        stat=paired_stats([r],'legacy','camera','translation_m')
        self.assertEqual(stat['n'],1)
        self.assertEqual(stat['best_counts']['left_pos'],1)

if __name__=='__main__':unittest.main()
