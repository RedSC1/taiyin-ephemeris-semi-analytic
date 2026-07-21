# Taiyin Semi-Analytical Ephemeris

A frozen-coefficient semi-analytical ephemeris covering calendar years
−3000 through +3000. It has no runtime data files, uses only the Python
standard library, and is published as `taiyin-ephemeris-semi-analytic`.

- Mercury through Pluto and the Earth–Moon barycenter use compact series
  independently fitted to JPL DE441.
- Planetary harmonics are generated from one sine/cosine pair per fundamental
  angle, so periodic terms require only multiplication and addition.
- The Moon uses the truncated XL1 lunar theory table from Shouxing
  Astronomical Ephemeris (寿星天文历/寿星万年历), followed by an independently
  fitted DE441 residual correction.
- Earth and Moon heliocentric positions are reconstructed from the EMB and
  the geocentric lunar vector.

## Command line

The input epoch is Julian Date on the TDB time scale:

```bash
pip install taiyin-ephemeris-semi-analytic
taiyin-semi-analytic 2451545.0 301
```

From a source checkout, the original one-file-style demo command remains:

```bash
python3 ephemeris.py 2451545.0 301
```

The command prints JSON containing:

- `xyz_km`: heliocentric ICRF Cartesian coordinates in kilometres;
- `lbr_rad_km`: spherical longitude, latitude and radius derived from that
  same ICRF vector, in radians, radians and kilometres;
- explicit target, center and frame metadata.

## Python API

```python
from taiyin_semi_analytic import lbr, position, result

xyz = position(2451545.0, 4)  # Mars
spherical = lbr(2451545.0, 4)
record = result(2451545.0, 4)
```

Supported target IDs:

| ID | Target |
|---:|---|
| 1 | Mercury |
| 2 | Venus |
| 3 | Earth–Moon barycenter |
| 4 | Mars |
| 5 | Jupiter system barycenter |
| 6 | Saturn system barycenter |
| 7 | Uranus system barycenter |
| 8 | Neptune system barycenter |
| 9 | Pluto system barycenter |
| 10 | Sun |
| 301 | Moon |
| 399 | Earth |

All returned vectors are Sun-centered.  IDs 301 and 399 are therefore also
heliocentric, not geocentric.

## Accuracy

Held-out validation against DE441 over calendar years −3000 through +3000
gave the following heliocentric angular RMS values for the frozen planetary
series:

| Target | RMS |
|---|---:|
| Mercury | 1.66″ |
| Venus | 0.66″ |
| EMB | 0.56″ |
| Mars | 2.29″ |
| Jupiter | 3.31″ |
| Saturn | 0.29″ |
| Uranus | 3.65″ |
| Neptune | 0.21″ |
| Pluto | 1.53″ |

The corrected geocentric lunar model measured 0.704″ angular RMS and
0.263 km radial RMS on held-out 32-day-grid epochs; maximum errors on that
grid were 5.22″ and 1.52 km.

## Design lineage

Steve Moshier's PLAN404 was an important inspiration for the compact
semi-analytical representation and harmonic-recursion evaluator used here.
The planetary series were developed by fitting to JPL DE441.

The lunar ecliptic-of-date coordinates are rotated to the J2000 ecliptic with
the published [P03 precession model](https://doi.org/10.1051/0004-6361:20031539)
of Capitaine, Wallace, and Chapront.

## Files

- `taiyin_semi_analytic/core.py`: evaluator and public implementation.
- `taiyin_semi_analytic/coefficients.py`: frozen coefficients.
- `taiyin_semi_analytic/__main__.py`: `python -m taiyin_semi_analytic` entry.
- `ephemeris.py`: source-tree command-line demo wrapper.
- `pyproject.toml`: wheel/sdist and console-script configuration.
- `tests/test_ephemeris.py`: standard-library regression tests.
- `NOTICE`: attribution and third-party provenance.

The project is licensed under Apache License 2.0.  The attributed Shouxing
lunar table retains its upstream provenance notice; see `NOTICE`.
