#!/usr/bin/env python3
"""Compact frozen-coefficient ephemeris demo.

Usage:
    python3 ephemeris.py JD_TDB TARGET_ID

The returned coordinates are heliocentric.  Cartesian coordinates use ICRF;
spherical coordinates are available in both ICRF and the J2000 ecliptic frame.
Distances are kilometres and angles are radians.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Sequence

from .coefficients import LUNAR_CORRECTION, LUNAR_XL1, PLANET_MODELS


J2000 = 2451545.0
ARCSEC_PER_RADIAN = 206264.80624709636
J2000_OBLIQUITY = math.radians(84381.406 / 3600.0)
J2000_OBLIQUITY_COSINE = math.cos(J2000_OBLIQUITY)
J2000_OBLIQUITY_SINE = math.sin(J2000_OBLIQUITY)
EARTH_MOON_MASS_RATIO = 81.30056822149722

BODY_NAMES = {
    1: "mercury",
    2: "venus",
    3: "earth-moon barycenter",
    4: "mars",
    5: "jupiter system barycenter",
    6: "saturn system barycenter",
    7: "uranus system barycenter",
    8: "neptune system barycenter",
    9: "pluto system barycenter",
    10: "sun",
    301: "moon",
    399: "earth",
}


def _powers(value: float, degree: int) -> list[float]:
    result = [1.0]
    for _ in range(degree):
        result.append(result[-1] * value)
    return result


def _chebyshev(value: float, degree: int) -> list[float]:
    result = [1.0]
    if degree == 0:
        return result
    result.append(value)
    for _ in range(2, degree + 1):
        result.append(2.0 * value * result[-1] - result[-2])
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _compile_planet_runtime(model: dict[str, Any]) -> tuple[Any, ...]:
    """Flatten a frozen model into the layout used by the hot evaluator."""

    body_id = model["body_id"]
    angle_body_ids = tuple(model["angle_body_ids"])
    angle_indices = {
        argument_id: index for index, argument_id in enumerate(angle_body_ids)
    }
    max_harmonics = [0] * len(angle_body_ids)
    carrier_max_harmonic = 0
    cursor = model["secular_degree"] + 1
    arguments = []

    for argument in model["arguments"]:
        factors = tuple(
            (body, multiplier) for body, multiplier in argument["factors"]
        )
        carrier_multiplier = 0
        if (
            model["pure_target_carrier"] is not None
            and len(factors) == 1
            and factors[0][0] == body_id
        ):
            carrier_multiplier = factors[0][1]
            carrier_max_harmonic = max(
                carrier_max_harmonic, abs(carrier_multiplier)
            )
            compiled_factors = ()
        else:
            compiled_factors = tuple(
                (angle_indices[argument_id], abs(multiplier) - 1, multiplier < 0)
                for argument_id, multiplier in factors
            )
            for angle_index, harmonic, _ in compiled_factors:
                max_harmonics[angle_index] = max(
                    max_harmonics[angle_index], harmonic + 1
                )

        degree = argument["amplitude_degree"]
        width = 2 * (degree + 1)
        amplitude_rows = tuple(
            tuple(row) for row in model["coefficients"][cursor : cursor + width]
        )
        amplitudes = tuple(
            tuple(
                complex(
                    amplitude_rows[2 * power][column],
                    -amplitude_rows[2 * power + 1][column],
                )
                for power in range(degree + 1)
            )
            for column in range(3)
        )
        if carrier_multiplier:
            carrier_harmonic = (abs(carrier_multiplier) - 1, carrier_multiplier < 0)
        else:
            carrier_harmonic = None
        arguments.append((compiled_factors, carrier_harmonic, amplitudes))
        cursor += width

    if cursor != len(model["coefficients"]):
        raise RuntimeError("planetary coefficient layout mismatch")

    carrier = model["pure_target_carrier"]
    carrier_coefficients = None if carrier is None else tuple(carrier["coefficients"])
    return (
        tuple(tuple(row) for row in model["angle_coefficients"]),
        tuple(max_harmonics),
        tuple(arguments),
        carrier_coefficients,
        carrier_max_harmonic,
    )


_PLANET_RUNTIMES = {
    body_id: _compile_planet_runtime(model) for body_id, model in PLANET_MODELS.items()
}

_LUNAR_XL1_RUNTIME = tuple(
    tuple(
        tuple(tuple(table[index : index + 6]) for index in range(0, len(table), 6))
        for table in coordinate
    )
    for coordinate in LUNAR_XL1
)


def _harmonics(angle: float, maximum: int) -> list[complex]:
    """Return exp(i*n*angle) using one base sin/cos pair."""

    if maximum == 0:
        return []
    base = complex(math.cos(angle), math.sin(angle))
    harmonics = [base]
    value = base
    for _ in range(1, maximum):
        value *= base
        harmonics.append(value)
    return harmonics


def _planet_ecliptic(body_id: int, jd_tdb: float) -> tuple[float, float, float]:
    model = PLANET_MODELS[body_id]
    if not model["jd_start"] <= jd_tdb <= model["jd_end"]:
        raise ValueError(
            f"JD {jd_tdb} is outside the supported interval "
            f"[{model['jd_start']}, {model['jd_end']}]"
        )
    u = (jd_tdb - model["epoch_jd"]) / model["half_span_days"]
    coefficients = model["coefficients"]
    secular_degree = model["secular_degree"]
    channels = []
    for column in range(3):
        value = coefficients[secular_degree][column]
        for row in range(secular_degree - 1, -1, -1):
            value = value * u + coefficients[row][column]
        channels.append(value)

    (
        angle_coefficients,
        max_harmonics,
        arguments,
        carrier_coefficients,
        carrier_max_harmonic,
    ) = _PLANET_RUNTIMES[body_id]
    harmonic_tables = []
    for row, maximum in zip(angle_coefficients, max_harmonics):
        angle = row[-1]
        for coefficient in reversed(row[:-1]):
            angle = angle * u + coefficient
        harmonic_tables.append(_harmonics(angle, maximum))

    carrier_harmonics = []
    if carrier_coefficients is not None and carrier_max_harmonic:
        carrier_angle = carrier_coefficients[-1]
        for coefficient in reversed(carrier_coefficients[:-1]):
            carrier_angle = carrier_angle * u + coefficient
        carrier_harmonics = _harmonics(carrier_angle, carrier_max_harmonic)

    for factors, carrier_harmonic, amplitudes in arguments:
        if carrier_harmonic is not None:
            harmonic, negative = carrier_harmonic
            phase = carrier_harmonics[harmonic]
            if negative:
                phase = phase.conjugate()
        else:
            phase = 1.0 + 0.0j
            for angle_index, harmonic, negative in factors:
                factor = harmonic_tables[angle_index][harmonic]
                if negative:
                    factor = factor.conjugate()
                phase *= factor

        for column, polynomial in enumerate(amplitudes):
            amplitude = polynomial[-1]
            for coefficient in reversed(polynomial[:-1]):
                amplitude = amplitude * u + coefficient
            channels[column] += (amplitude * phase).real

    longitude, latitude, log_radius = channels
    radius = model["radius_scale_km"] * math.exp(log_radius)
    cos_latitude = math.cos(latitude)
    return (
        radius * cos_latitude * math.cos(longitude),
        radius * cos_latitude * math.sin(longitude),
        radius * math.sin(latitude),
    )


# P03 ecliptic-precession angles in arcseconds.  See Capitaine, Wallace &
# Chapront (2003), A&A 412, 567-586, doi:10.1051/0004-6361:20031539.
_P03_ECLIPTIC_ANGLES = {
    "phi": (0.0, 5038.481507, -1.0790069, -0.00114045, 0.000132851, -9.51e-8),
    "omega": (
        84381.406,
        -0.025754,
        0.0512623,
        -0.00772503,
        -4.67e-7,
        3.337e-7,
    ),
    "epsilon": (
        84381.406,
        -46.836769,
        -0.0001831,
        0.00200340,
        -5.76e-7,
        -4.34e-8,
    ),
    "chi": (0.0, 10.556403, -2.3814292, -0.00121197, 0.000170663, -5.60e-8),
}


def _precession(t: float, name: str) -> float:
    coefficients = _P03_ECLIPTIC_ANGLES[name]
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * t + coefficient
    return value / ARCSEC_PER_RADIAN


def _rotate_x(
    vector: tuple[float, float, float], angle: float
) -> tuple[float, float, float]:
    x, y, z = vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return x, cosine * y - sine * z, sine * y + cosine * z


def _rotate_z(
    vector: tuple[float, float, float], angle: float
) -> tuple[float, float, float]:
    x, y, z = vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return cosine * x - sine * y, sine * x + cosine * y, z


def _date_ecliptic_to_j2000(
    t: float, longitude: float, latitude: float, radius: float
) -> tuple[float, float, float]:
    """Rotate P03 mean ecliptic-of-date coordinates to J2000 ecliptic."""

    cosine_latitude = math.cos(latitude)
    vector = (
        cosine_latitude * math.cos(longitude),
        cosine_latitude * math.sin(longitude),
        math.sin(latitude),
    )
    # P03: R3(-phi) R1(-omega) R3(chi) R1(epsilon).
    vector = _rotate_x(vector, _precession(t, "epsilon"))
    vector = _rotate_z(vector, _precession(t, "chi"))
    vector = _rotate_x(vector, -_precession(t, "omega"))
    x, y, z = _rotate_z(vector, -_precession(t, "phi"))
    return (
        math.atan2(y, x) % (2.0 * math.pi),
        math.atan2(z, math.hypot(x, y)),
        radius,
    )


def _xl1_coordinate(coordinate: int, t: float) -> float:
    value = 0.0
    if coordinate == 0:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        value += (
            3.81034409
            + 8399.684730072 * t
            - 3.319e-05 * t2
            + 3.11e-08 * t3
            - 2.033e-10 * t4
        ) * ARCSEC_PER_RADIAN
        value += (
            5028.792262 * t
            + 1.1124406 * t2
            + 0.00007699 * t3
            - 0.000023479 * t4
            - 0.0000000178 * t5
        )
        if t > 10.0:
            offset = t - 10.0
            value += -0.866 + 1.43 * offset + 0.054 * offset * offset

    phase_t2 = t * t / 1.0e4
    phase_t3 = t * t * t / 1.0e8
    phase_t4 = t * t * t * t / 1.0e8
    envelope = 1.0
    for table in _LUNAR_XL1_RUNTIME[coordinate]:
        subtotal = 0.0
        for amplitude, p0, p1, p2, p3, p4 in table:
            phase = p0 + t * p1 + phase_t2 * p2 + phase_t3 * p3 + phase_t4 * p4
            subtotal += amplitude * math.cos(phase)
        value += envelope * subtotal
        envelope *= t
    if coordinate != 2:
        value /= ARCSEC_PER_RADIAN
    return value


def _moon_base_lbr(t: float) -> tuple[float, float, float]:
    return _date_ecliptic_to_j2000(
        t,
        _xl1_coordinate(0, t),
        _xl1_coordinate(1, t),
        _xl1_coordinate(2, t),
    )


def _moon_correction(channel: dict, t: float, u: float) -> float:
    chebyshev = _chebyshev(
        u, max(LUNAR_CORRECTION["secular_degree"], channel["modulation_degree"])
    )
    value = _dot(channel["secular_chebyshev_coefficients"], chebyshev)
    phase_t2 = t * t / 1.0e4
    phase_t3 = t * t * t / 1.0e8
    phase_t4 = t * t * t * t / 1.0e8
    for term in channel["terms"]:
        p0, p1, p2, p3, p4 = term["phase"]
        phase = p0 + t * p1 + phase_t2 * p2 + phase_t3 * p3 + phase_t4 * p4
        envelope = t ** term["order"]
        cosine_amplitude = _dot(term["cosine_chebyshev_coefficients"], chebyshev)
        sine_amplitude = _dot(term["sine_chebyshev_coefficients"], chebyshev)
        value += envelope * (
            cosine_amplitude * math.cos(phase)
            + sine_amplitude * math.sin(phase)
        )
    return value


def _moon_geocentric_ecliptic(jd_tdb: float) -> tuple[float, float, float]:
    epoch = LUNAR_CORRECTION["epoch_jd"]
    half_span = LUNAR_CORRECTION["half_span_days"]
    u = (jd_tdb - epoch) / half_span
    if not -1.0 <= u <= 1.0:
        raise ValueError(
            f"JD {jd_tdb} is outside the supported lunar interval "
            f"[{epoch - half_span}, {epoch + half_span}]"
        )
    t = (jd_tdb - J2000) / 36525.0
    longitude, latitude, radius = _moon_base_lbr(t)
    channels = [
        longitude,
        latitude,
        math.log(radius / LUNAR_CORRECTION["radius_scale_km"]),
    ]
    for channel in LUNAR_CORRECTION["channels"]:
        source_coordinate = channel["source_coordinate"]
        channels[source_coordinate] += _moon_correction(channel, t, u)
    longitude, latitude, log_radius = channels
    radius = LUNAR_CORRECTION["radius_scale_km"] * math.exp(log_radius)
    cos_latitude = math.cos(latitude)
    return (
        radius * cos_latitude * math.cos(longitude),
        radius * cos_latitude * math.sin(longitude),
        radius * math.sin(latitude),
    )


def _add(
    left: Sequence[float], right: Sequence[float], scale: float = 1.0
) -> tuple[float, float, float]:
    return (
        left[0] + scale * right[0],
        left[1] + scale * right[1],
        left[2] + scale * right[2],
    )


def _heliocentric_ecliptic(body_id: int, jd_tdb: float) -> tuple[float, float, float]:
    if body_id == 10:
        return 0.0, 0.0, 0.0
    if body_id in PLANET_MODELS:
        return _planet_ecliptic(body_id, jd_tdb)
    if body_id not in (301, 399):
        raise KeyError(f"unsupported target id {body_id}")
    emb = _planet_ecliptic(3, jd_tdb)
    moon_from_earth = _moon_geocentric_ecliptic(jd_tdb)
    moon_fraction = 1.0 / (EARTH_MOON_MASS_RATIO + 1.0)
    if body_id == 399:
        return _add(emb, moon_from_earth, -moon_fraction)
    return _add(emb, moon_from_earth, 1.0 - moon_fraction)


def _ecliptic_to_icrf(vector: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = vector
    return (
        x,
        J2000_OBLIQUITY_COSINE * y - J2000_OBLIQUITY_SINE * z,
        J2000_OBLIQUITY_SINE * y + J2000_OBLIQUITY_COSINE * z,
    )


def position(jd_tdb: float, body_id: int) -> tuple[float, float, float]:
    """Return a heliocentric ICRF position in kilometres."""

    return _ecliptic_to_icrf(_heliocentric_ecliptic(body_id, float(jd_tdb)))


def _vector_lbr(vector: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = vector
    radius = math.sqrt(x * x + y * y + z * z)
    if radius == 0.0:
        return 0.0, 0.0, 0.0
    return math.atan2(y, x) % (2.0 * math.pi), math.asin(z / radius), radius


def spherical_icrf(jd_tdb: float, body_id: int) -> tuple[float, float, float]:
    """Return ICRF spherical longitude, latitude (rad), and radius (km)."""

    return _vector_lbr(position(jd_tdb, body_id))


def ecliptic_lbr_j2000(
    jd_tdb: float, body_id: int
) -> tuple[float, float, float]:
    """Return J2000 ecliptic longitude, latitude (rad), and radius (km)."""

    return _vector_lbr(_heliocentric_ecliptic(body_id, float(jd_tdb)))


def result(jd_tdb: float, body_id: int) -> dict:
    ecliptic_xyz = _heliocentric_ecliptic(body_id, float(jd_tdb))
    xyz = _ecliptic_to_icrf(ecliptic_xyz)
    spherical = _vector_lbr(xyz)
    ecliptic = _vector_lbr(ecliptic_xyz)
    return {
        "jd_tdb": float(jd_tdb),
        "target_id": body_id,
        "target": BODY_NAMES.get(body_id, "unknown"),
        "center_id": 10,
        "center": "sun",
        "frame": "ICRF",
        "xyz_km": list(xyz),
        "spherical_icrf_rad_km": list(spherical),
        "ecliptic_lbr_j2000_rad_km": list(ecliptic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jd_tdb", type=float, help="Julian Date on the TDB time scale")
    parser.add_argument("target_id", type=int, choices=tuple(BODY_NAMES))
    args = parser.parse_args()
    print(json.dumps(result(args.jd_tdb, args.target_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
