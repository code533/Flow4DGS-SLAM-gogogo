import unittest
from scripts.check_v43a4_config import assert_preset, SMOKE, FULL


class TestV43a4Config(unittest.TestCase):
    def test_smoke_preset(self):
        conf=assert_preset(SMOKE)
        exp=conf["model_params"]["v43a_observation"]
        self.assertEqual(exp["predictive_max_sources"],2)
        self.assertEqual(exp["predictive_source_every_n_event"],5)

    def test_full_preset(self):
        conf=assert_preset(FULL)
        exp=conf["model_params"]["v43a_observation"]
        self.assertEqual(exp["predictive_max_sources"],25)
        self.assertEqual(exp["predictive_source_every_n_event"],4)


if __name__=="__main__":unittest.main()
