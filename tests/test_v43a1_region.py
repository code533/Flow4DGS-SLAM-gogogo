import unittest
import numpy as np
from utils.v43a1_region_analysis import eligible_keyframe,analyze_paired_regions

class TestV43a1(unittest.TestCase):
 def test_event_based_sampling(self):
  self.assertEqual([i for i in range(1,12) if eligible_keyframe(i,5)],[1,6,11])
  with self.assertRaises(ValueError):eligible_keyframe(1,0)
 def make(self):
  h,w=6,8
  dyn=np.ones((h,w),bool)
  dvalid=np.ones((h,w),bool)
  rgbvalid=np.ones((h,w),bool)
  baseline=np.full((h,w),.10)
  bd=np.full((h,w),.2)
  def entry(rgb,depth):
   return dict(rgb_abs_mean=np.asarray(rgb),depth_penalized_abs=np.asarray(depth),
       prediction_covered=np.ones((h,w),bool),dynamic_region=dyn.copy(),
       gt_depth_valid=dvalid.copy(),gt_rgb_valid=rgbvalid.copy())
  return {'g0':entry(baseline,bd),
          'g3_support':entry(baseline-.01,bd-.02),
          'g4':entry(baseline+.01,bd-.02)}
 def test_both_better_and_conflict(self):
  rows,regions=analyze_paired_regions(self.make(),frame_id=22,event_index=4,
      grid_rows=2,grid_cols=2,min_dynamic_rgb_pixels=1,min_dynamic_depth_pixels=1)
  self.assertEqual(len(rows),2)
  self.assertEqual(len(regions),12)
  support=next(r for r in rows if r["hypothesis"]=="g3_support")
  g4=next(r for r in rows if r["hypothesis"]=="g4")
  self.assertAlmostEqual(support["pixel_both_better_fraction"],1.)
  self.assertAlmostEqual(g4["pixel_depth_better_rgb_worse_fraction"],1.)
  self.assertEqual(support["tiles_both_better"],4)
  self.assertEqual(g4["tiles_conflict"],4)
 def test_candidate_specific_masks_rejected(self):
  data=self.make()
  data["g4"]["gt_depth_valid"][0,0]=False
  with self.assertRaises(ValueError):analyze_paired_regions(data,frame_id=1,event_index=1)
 def test_empty_regions_and_tiles(self):
  data=self.make()
  for v in data.values():v["dynamic_region"][:]=False
  rows,tiles=analyze_paired_regions(data,frame_id=22,event_index=1)
  self.assertEqual(len(tiles),0)
  self.assertTrue(np.isnan(rows[0]["pixel_both_better_fraction"]))
 def test_tolerance_explicit_tie(self):
  data=self.make()
  data["g3_support"]["rgb_abs_mean"][:]=data["g0"]["rgb_abs_mean"]
  rows,tiles=analyze_paired_regions(data,frame_id=1,event_index=1,
       grid_rows=2,grid_cols=2,min_dynamic_rgb_pixels=1,min_dynamic_depth_pixels=1)
  self.assertTrue(all(r["rgb_winner"]=="tie" for r in tiles))
if __name__=="__main__":unittest.main()
