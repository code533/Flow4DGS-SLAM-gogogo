import tempfile
import unittest
from pathlib import Path
import torch

from utils.v42b_online import (
    validate_online_config, choose_online_motion,
    online_row, append_online_row,
)


class TestOnlineV42b(unittest.TestCase):
    def config(self, mode, **settings):
        cfg={"g34_shadow": False, "g4_ablation": False, "online_trace": False}
        cfg.update(settings)
        return validate_online_config(
            mode, cfg, use_knn=True, flow_offset=True,
            dynamic_model="offset", flow_ds=1)

    def test_supported_modes(self):
        for mode in ("off", "shadow", "g0", "g3", "g3_support", "g4"):
            self.config(mode)
        with self.assertRaises(ValueError):
            self.config("g5")
        with self.assertRaises(ValueError):
            self.config("g4", g34_shadow=True)
        with self.assertRaises(ValueError):
            validate_online_config("g4", {}, use_knn=False,
                flow_offset=True, dynamic_model="offset", flow_ds=1)
        with self.assertRaises(ValueError):
            validate_online_config("g3", {}, use_knn=True,
                flow_offset=True, dynamic_model="offset", flow_ds=2)

    def test_shadow_and_g0_are_exact_baselines(self):
        g0=torch.tensor([[.1,0,0],[.2,0,0]])
        d={"g0":g0,"g3":g0+1,"g3_support":g0+.5,"g4":g0+.25}
        for mode in ("shadow","g0"):
            self.assertTrue(torch.equal(choose_online_motion(mode,d),g0))
        self.assertTrue(torch.equal(choose_online_motion("g3_support",d),d["g3_support"]))
        self.assertTrue(torch.equal(choose_online_motion("g4",d),d["g4"]))
        with self.assertRaises(ValueError):
            choose_online_motion("off",d)
        d["g4"]=torch.full_like(g0,float("nan"))
        with self.assertRaises(FloatingPointError):
            choose_online_motion("g0",d)

    def test_online_gate_and_row_saved(self):
        g0=torch.tensor([[.2,0,0],[.4,0,0]])
        selected=torch.tensor([[.4,0,0],[.2,0,0]])
        q=torch.tensor([.8,.2])
        reason=torch.tensor([0,8])
        visible=torch.ones(2,dtype=torch.bool)
        diag={"visible":visible,"g4_accepted":torch.tensor([True,False])}
        gates={"support_only":torch.tensor([True,False])}
        row=online_row(212,"g3_support",g0,selected,q,reason,
            selected,selected,selected,diag,gates,gate_m=.3)
        self.assertEqual(row["new_gate_rejections_n"],1)
        self.assertEqual(row["new_gate_acceptances_n"],1)
        self.assertEqual(row["g0_gate_pass_n"],1)
        self.assertEqual(row["selected_gate_pass_n"],1)
        self.assertAlmostEqual(row["effective_vs_g0_mean_m"],.2,places=5)
        with tempfile.TemporaryDirectory() as folder:
            append_online_row(folder,row)
            append_online_row(folder,{**row,"keyframe_id":213})
            lines=(Path(folder)/"v42b_per_keyframe.csv").read_text().splitlines()
            self.assertEqual(len(lines),3)

    def test_off_mode_prohibits_online_trace(self):
        with self.assertRaises(ValueError):
            self.config("off",online_trace=True)


if __name__=="__main__":
    unittest.main()
