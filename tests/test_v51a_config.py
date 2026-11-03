import unittest

from scripts.check_v51a_config import check_all


class TestV51aPresets(unittest.TestCase):
    def test_three_modes_only_differ_in_v51a(self):
        c=check_all()
        self.assertEqual(set(c),{"legacy","shadow","strict"})
        self.assertTrue(all(x["model_params"]["v42a_propagation"]["mode"]=="g0"
                            for x in c.values()))


if __name__=="__main__":unittest.main()
