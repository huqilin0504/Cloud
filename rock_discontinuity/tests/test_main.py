import unittest
from unittest.mock import patch

from rock_discontinuity.app.main import COMMANDS, main


class MainEntryPointTests(unittest.TestCase):
    def test_defaults_to_whole_cloud(self):
        whole_cloud = patch("rock_discontinuity.app.main.whole_cloud_main", return_value=11).start()
        self.addCleanup(patch.stopall)
        with patch.dict(COMMANDS, {"whole-cloud": whole_cloud}):
            self.assertEqual(main(["--plan-only"]), 11)
        whole_cloud.assert_called_once_with(["--plan-only"])

    def test_routes_roi(self):
        roi = patch("rock_discontinuity.app.main.roi_main", return_value=12).start()
        self.addCleanup(patch.stopall)
        with patch.dict(COMMANDS, {"roi": roi}):
            self.assertEqual(main(["roi", "--bbox", "0,1,0,1"]), 12)
        roi.assert_called_once_with(["--bbox", "0,1,0,1"])

    def test_routes_validation(self):
        validate = patch("rock_discontinuity.app.main.validate_main", return_value=13).start()
        self.addCleanup(patch.stopall)
        with patch.dict(COMMANDS, {"validate": validate}):
            self.assertEqual(main(["--mode=validate", "outputs/example"]), 13)
        validate.assert_called_once_with(["outputs/example"])

    def test_routes_roi_comparison(self):
        comparison = patch("rock_discontinuity.app.main.roi_compare_main", return_value=14).start()
        self.addCleanup(patch.stopall)
        with patch.dict(COMMANDS, {"roi-compare": comparison}):
            self.assertEqual(main(["roi-compare", "--output", "outputs/compare"]), 14)
        comparison.assert_called_once_with(["--output", "outputs/compare"])


if __name__ == "__main__":
    unittest.main()
