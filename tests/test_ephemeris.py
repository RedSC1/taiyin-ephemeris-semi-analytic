import json
import math
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ephemeris


class EphemerisTests(unittest.TestCase):
    def test_frozen_mercury_regression(self):
        expected = (
            -19462192.307992253,
            -59927766.037858255,
            -29992780.805886015,
        )
        actual = ephemeris.position(2451545.0, 1)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=1.0e-6)

    def test_frozen_lunar_base_regression(self):
        expected = (3.897650093444056, 0.09024898586276324, 402448.7431750884)
        actual = ephemeris._moon_base_lbr(0.0)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=1.0e-10)

    def test_earth_moon_barycenter_identity(self):
        epoch = 2451545.0
        earth = ephemeris.position(epoch, 399)
        moon = ephemeris.position(epoch, 301)
        emb = ephemeris.position(epoch, 3)
        ratio = ephemeris.EARTH_MOON_MASS_RATIO
        reconstructed = tuple(
            (ratio * earth_axis + moon_axis) / (ratio + 1.0)
            for earth_axis, moon_axis in zip(earth, moon)
        )
        for left, right in zip(reconstructed, emb):
            self.assertAlmostEqual(left, right, delta=3.0e-8)

    def test_spherical_coordinates_match_cartesian_vector(self):
        for body_id in (1, 3, 6, 301, 399):
            xyz = ephemeris.position(2451545.0, body_id)
            longitude, latitude, radius = ephemeris.lbr(2451545.0, body_id)
            reconstructed = (
                radius * math.cos(latitude) * math.cos(longitude),
                radius * math.cos(latitude) * math.sin(longitude),
                radius * math.sin(latitude),
            )
            for left, right in zip(reconstructed, xyz):
                self.assertAlmostEqual(left, right, delta=1.0e-6)

    def test_cli_returns_self_describing_json(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "ephemeris.py"), "2451545.0", "4"],
            text=True,
            capture_output=True,
            check=True,
        )
        record = json.loads(completed.stdout)
        self.assertEqual(record["target_id"], 4)
        self.assertEqual(record["center_id"], 10)
        self.assertEqual(record["frame"], "ICRF")
        self.assertEqual(len(record["xyz_km"]), 3)
        self.assertEqual(len(record["lbr_rad_km"]), 3)

    def test_interval_is_enforced(self):
        with self.assertRaises(ValueError):
            ephemeris.position(600000.0, 1)
        with self.assertRaises(ValueError):
            ephemeris.position(600000.0, 301)


if __name__ == "__main__":
    unittest.main()
