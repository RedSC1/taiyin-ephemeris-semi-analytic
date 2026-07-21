import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import taiyin_semi_analytic.core as ephemeris


class EphemerisTests(unittest.TestCase):
    def test_frozen_mercury_regression(self):
        expected = (
            -19462192.30799237,
            -59927766.037858136,
            -29992780.80588731,
        )
        actual = ephemeris.position(2451545.0, 1)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=1.0e-6)

    def test_planets_reuse_fundamental_angle_harmonics(self):
        expected_trigonometric_calls = {
            1: 14,
            2: 14,
            3: 24,
            4: 18,
            5: 12,
            6: 12,
            7: 12,
            8: 12,
            9: 14,
        }
        original_sine = ephemeris.math.sin
        original_cosine = ephemeris.math.cos
        for body_id, expected in expected_trigonometric_calls.items():
            with patch.object(
                ephemeris.math, "sin", side_effect=original_sine
            ) as sine:
                with patch.object(
                    ephemeris.math, "cos", side_effect=original_cosine
                ) as cosine:
                    ephemeris.position(2451545.0, body_id)
            self.assertEqual(sine.call_count + cosine.call_count, expected)

    def test_frozen_lunar_base_regression(self):
        expected = (3.897650093444056, 0.09024898586276324, 402448.7431750884)
        actual = ephemeris._moon_base_lbr(0.0)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=1.0e-10)

    def test_p03_date_ecliptic_to_j2000_regression(self):
        expected = (0.955335336627986, -0.30200214694055044, 400000.0)
        actual = ephemeris._date_ecliptic_to_j2000(10.0, 1.2, -0.3, 400000.0)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=1.0e-13)

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

    def test_spherical_icrf_matches_cartesian_vector(self):
        for body_id in (1, 3, 6, 301, 399):
            xyz = ephemeris.position(2451545.0, body_id)
            longitude, latitude, radius = ephemeris.spherical_icrf(
                2451545.0, body_id
            )
            reconstructed = (
                radius * math.cos(latitude) * math.cos(longitude),
                radius * math.cos(latitude) * math.sin(longitude),
                radius * math.sin(latitude),
            )
            for left, right in zip(reconstructed, xyz):
                self.assertAlmostEqual(left, right, delta=1.0e-6)

    def test_ecliptic_lbr_j2000_matches_ecliptic_vector(self):
        for body_id in (1, 3, 6, 301, 399):
            xyz = ephemeris._heliocentric_ecliptic(body_id, 2451545.0)
            longitude, latitude, radius = ephemeris.ecliptic_lbr_j2000(
                2451545.0, body_id
            )
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
        self.assertEqual(len(record["spherical_icrf_rad_km"]), 3)
        self.assertEqual(len(record["ecliptic_lbr_j2000_rad_km"]), 3)

    def test_result_evaluates_ephemeris_once(self):
        evaluator = ephemeris._heliocentric_ecliptic
        with patch.object(
            ephemeris, "_heliocentric_ecliptic", wraps=evaluator
        ) as wrapped:
            ephemeris.result(2451545.0, 4)
        self.assertEqual(wrapped.call_count, 1)

    def test_interval_is_enforced(self):
        with self.assertRaises(ValueError):
            ephemeris.position(600000.0, 1)
        with self.assertRaises(ValueError):
            ephemeris.position(600000.0, 301)


if __name__ == "__main__":
    unittest.main()
