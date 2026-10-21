import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_camera_motion_v31b import (se3_exp, infer_limit,
    capped_candidate, read_frame, relative_metrics, paired_summary, main)


class TestV31b(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prev = np.eye(4)
        self.gt_prev = np.eye(4)
        self.gt_now = np.eye(4)
        self.gt_now[0, 3] = .002
        self.xleg = np.array([.001, 0., 0., 0., 0., 0.])
        self.xgeo = np.array([.002, 0., 0., 0., 0., 0.])
        self.limit = .003
        self.path = self.root / 'camera_motion_v31a_00011.npz'
        self.save()

    def save(self, status='ok'):
        A = self.prev @ se3_exp(self.xleg)
        B = self.prev @ se3_exp(self.xgeo)
        L = A.copy(); G = B.copy()
        for M, raw in ((L,A),(G,B)):
            v = raw[:3,3] - self.prev[:3,3]
            M[:3,3] = self.prev[:3,3] + self.limit * v / max(np.linalg.norm(v), 1e-10)
        np.savez_compressed(self.path, status=status, frame_id=11,
            xi_legacy_common=self.xleg, xi_geo=self.xgeo,
            T_previous_est=self.prev, T_candidate_legacy_common=L,
            T_candidate_geo=G, T_previous_gt=self.gt_prev,
            T_current_gt=self.gt_now)

    def test_raw_se3(self):
        T = se3_exp(np.array([.1,-.2,.3,.0,.0,.0]))
        self.assertTrue(np.allclose(T[:3,3],[.1,-.2,.3]))
        self.assertTrue(np.allclose(T[:3,:3],np.eye(3)))

    def test_raw_advantage_masked_by_legacy(self):
        r = read_frame(self.path)
        self.assertAlmostEqual(r['translation_limit_m'], .003)
        self.assertAlmostEqual(r['legacy_raw_right_translation_error_m'], .001)
        self.assertAlmostEqual(r['geo_raw_right_translation_error_m'], 0.)
        self.assertAlmostEqual(r['legacy_legacy_clip_right_translation_error_m'], .001)
        self.assertAlmostEqual(r['geo_legacy_clip_right_translation_error_m'], .001)
        self.assertAlmostEqual(r['legacy_capped_clip_right_translation_error_m'], .001)
        self.assertAlmostEqual(r['geo_capped_clip_right_translation_error_m'], 0.)
        self.assertEqual(r['legacy_legacy_amplifies'],1)
        self.assertEqual(r['geo_legacy_amplifies'],1)

    def test_clip_above_limit(self):
        x=np.array([.02,0,0,0,0,0])
        raw=se3_exp(x)
        save=raw.copy();save[0,3]=self.limit
        cand=capped_candidate(self.prev,raw,save,self.limit)
        self.assertAlmostEqual(cand[0,3],self.limit)

    def test_rotation_exp(self):
        x=np.array([0,0,0,0,0,.2])
        R=se3_exp(x)[:3,:3]
        self.assertAlmostEqual(R[0,0], np.cos(.2))
        self.assertAlmostEqual(R[1,0], np.sin(.2))

    def test_world_to_camera_camera_relative(self):
        prev=np.eye(4);next_=np.eye(4)
        prev[:3,3]=[2,0,0]; next_[:3,3]=[2.001,0,0]
        a=relative_metrics(prev,next_,prev,next_,'camera',.001)
        self.assertAlmostEqual(a['translation_error_m'],0.)
        self.assertAlmostEqual(a['rotation_error_deg'],0.)

    def test_direction_skipped_for_static_gt(self):
        v=relative_metrics(self.prev,self.prev,self.gt_prev,self.gt_prev,'right',.001)
        self.assertTrue(np.isnan(v['direction_error_deg']))

    def test_direction_preserved_for_motion(self):
        r=read_frame(self.path)
        self.assertAlmostEqual(r['geo_raw_right_direction_error_deg'],0.,places=4)

    def test_missing_status(self):
        self.save(status='shadow_exception')
        with self.assertRaisesRegex(ValueError,'Shadow status'):
            read_frame(self.path)

    def test_bad_shapes(self):
        with self.assertRaises(ValueError):se3_exp(np.ones(5))

    def test_mismatch_limit_rejected(self):
        a=np.eye(4);b=np.eye(4)
        a[0,3]=.003;b[0,3]=.1
        with self.assertRaisesRegex(ValueError,'disagree'):
            infer_limit(np.eye(4),a,b,se3_exp(self.xleg),se3_exp(self.xgeo))

    def test_csv_batch(self):
        dst=self.root/'result.csv'
        self.assertEqual(main([str(self.root),'--csv',str(dst)]),0)
        self.assertTrue(dst.is_file())
        lines=dst.read_text().splitlines()
        self.assertEqual(len(lines),2)

    def test_reject_old_csv(self):
        file=self.root/'old.csv';file.write_text('frame_id\n1\n')
        with self.assertRaises(SystemExit):main([str(file)])

    def test_pairing(self):
        r=read_frame(self.path)
        s='\n'.join(paired_summary([r],('right',)))
        self.assertIn('legacy -> geo [right] raw translation_error_m',s)

if __name__=='__main__':unittest.main()
