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
from typing import Any, NamedTuple, Sequence

from .coefficients import LUNAR_CORRECTION, LUNAR_XL1, PLANET_MODELS
from .derivative_coefficients import (
    LUNAR_CORRECTION_PHASE_DERIVATIVES,
    LUNAR_XL1_PHASE_DERIVATIVES,
    P03_DERIVATIVE_COEFFICIENTS,
    PLANET_DERIVATIVE_COEFFICIENTS,
)


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


class CartesianPV(NamedTuple):
    """Cartesian position and velocity in kilometres and kilometres/day."""

    position_km: tuple[float, float, float]
    velocity_km_per_day: tuple[float, float, float]


class CartesianPVA(NamedTuple):
    """Cartesian position, velocity and acceleration."""

    position_km: tuple[float, float, float]
    velocity_km_per_day: tuple[float, float, float]
    acceleration_km_per_day2: tuple[float, float, float]




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


def _polynomial_plain(coefficients: Sequence[Any], x: float) -> Any:
    """Evaluate a (possibly complex) ordinary polynomial by Horner's rule."""

    if not coefficients:
        return 0.0
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient
    return value


def _polynomial_first(
    coefficients: Sequence[float], first_coefficients: Sequence[float], x: float,
    dx_djd: float,
) -> tuple[float, float]:
    value = _polynomial_plain(coefficients, x)
    first = _polynomial_plain(first_coefficients, x) * dx_djd
    return value, first


def _polynomial_second(
    coefficients: Sequence[float], first_coefficients: Sequence[float],
    second_coefficients: Sequence[float], x: float, dx_djd: float,
) -> tuple[float, float, float]:
    value = _polynomial_plain(coefficients, x)
    first = _polynomial_plain(first_coefficients, x) * dx_djd
    second = _polynomial_plain(second_coefficients, x) * dx_djd * dx_djd
    return value, first, second


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


def _compile_planet_derivative_runtime(
    model: dict[str, Any],
) -> tuple[Any, ...]:
    """Join frozen derivative tables to the ordinary runtime coefficient layout."""

    body_id = model["body_id"]
    angle_coefficients, _, arguments, carrier_coefficients, _ = _PLANET_RUNTIMES[
        body_id
    ]
    frozen = PLANET_DERIVATIVE_COEFFICIENTS[body_id]
    secular_degree = model["secular_degree"]
    raw_coefficients = model["coefficients"]
    secular = tuple(
        (
            tuple(raw_coefficients[row][column] for row in range(secular_degree + 1)),
            tuple(frozen["secular"][column][0]),
            tuple(frozen["secular"][column][1]),
        )
        for column in range(3)
    )
    angles = tuple(
        (
            tuple(coefficients),
            tuple(frozen["angles"][index][0]),
            tuple(frozen["angles"][index][1]),
        )
        for index, coefficients in enumerate(angle_coefficients)
    )

    derivative_arguments = []
    for argument_index, (factors, carrier_harmonic, amplitudes) in enumerate(arguments):
        amplitude_tables = []
        for column, polynomial in enumerate(amplitudes):
            real_derivatives, imaginary_derivatives = frozen["amplitudes"][argument_index][column]
            amplitude_tables.append(
                (
                    (
                        tuple(value.real for value in polynomial),
                        tuple(real_derivatives[0]),
                        tuple(real_derivatives[1]),
                    ),
                    (
                        tuple(value.imag for value in polynomial),
                        tuple(imaginary_derivatives[0]),
                        tuple(imaginary_derivatives[1]),
                    ),
                )
            )
        derivative_arguments.append(
            (factors, carrier_harmonic, tuple(amplitude_tables))
        )

    carrier = (
        None
        if carrier_coefficients is None
        else (
            tuple(carrier_coefficients),
            tuple(frozen["carrier"][0]),
            tuple(frozen["carrier"][1]),
        )
    )
    return (
        secular,
        angles,
        tuple(derivative_arguments),
        carrier,
    )


# These are immutable tuples joined once from the frozen tables.  No
# derivative algebra is performed in the public hot path.
_PLANET_DERIVATIVE_RUNTIMES = {
    body_id: _compile_planet_derivative_runtime(model)
    for body_id, model in PLANET_MODELS.items()
}

_LUNAR_XL1_RUNTIME = tuple(
    tuple(
        tuple(tuple(table[index : index + 6]) for index in range(0, len(table), 6))
        for table in coordinate
    )
    for coordinate in LUNAR_XL1
)


_LUNAR_XL1_DERIVATIVE_RUNTIME = tuple(
    tuple(
        tuple(
            (
                raw_term[0],
                (
                    (
                        raw_term[1],
                        raw_term[2],
                        raw_term[3] * 1.0e-4,
                        raw_term[4] * 1.0e-8,
                        raw_term[5] * 1.0e-8,
                    ),
                    tuple(derivatives[0]),
                    tuple(derivatives[1]),
                ),
            )
            for raw_term, derivatives in zip(raw_table, frozen_table)
        )
        for raw_table, frozen_table in zip(coordinate, frozen_coordinate)
    )
    for coordinate, frozen_coordinate in zip(
        _LUNAR_XL1_RUNTIME, LUNAR_XL1_PHASE_DERIVATIVES
    )
)

_LUNAR_CORRECTION_PHASE_TABLES = tuple(
    tuple(
        (
            (
                (
                    term["phase"][0],
                    term["phase"][1],
                    term["phase"][2] * 1.0e-4,
                    term["phase"][3] * 1.0e-8,
                    term["phase"][4] * 1.0e-8,
                ),
                tuple(derivatives[0]),
                tuple(derivatives[1]),
            )
        )
        for term, derivatives in zip(
            channel["terms"], frozen_channel
        )
    )
    for channel, frozen_channel in zip(
        LUNAR_CORRECTION["channels"], LUNAR_CORRECTION_PHASE_DERIVATIVES
    )
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

_P03_ECLIPTIC_ANGLE_TABLES = {
    name: (
        tuple(_P03_ECLIPTIC_ANGLES[name]),
        tuple(derivatives[0]),
        tuple(derivatives[1]),
    )
    for name, derivatives in P03_DERIVATIVE_COEFFICIENTS.items()
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


def _planet_ecliptic_velocity(
    body_id: int, jd_tdb: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    model = PLANET_MODELS[body_id]
    if not model["jd_start"] <= jd_tdb <= model["jd_end"]:
        raise ValueError(
            f"JD {jd_tdb} is outside the supported interval "
            f"[{model['jd_start']}, {model['jd_end']}]"
        )
    u = (jd_tdb - model["epoch_jd"]) / model["half_span_days"]
    du_djd = 1.0 / model["half_span_days"]
    secular, angle_tables, arguments, carrier_table = _PLANET_DERIVATIVE_RUNTIMES[body_id]
    channels = [_polynomial_first(table[0], table[1], u, du_djd) for table in secular]
    longitude, longitude_rate = channels[0]
    latitude, latitude_rate = channels[1]
    log_radius, log_radius_rate = channels[2]
    _, max_harmonics, _, _, _ = _PLANET_RUNTIMES[body_id]

    harmonic_tables = []
    for table, maximum in zip(angle_tables, max_harmonics):
        angle, angle_rate = _polynomial_first(table[0], table[1], u, du_djd)
        if maximum == 0:
            harmonic_tables.append([])
            continue
        base = complex(math.cos(angle), math.sin(angle))
        base_rate = base * (1j * angle_rate)
        harmonics = [(base, base_rate)]
        for _ in range(1, maximum):
            previous, previous_rate = harmonics[-1]
            harmonics.append(
                (previous * base, previous_rate * base + previous * base_rate)
            )
        harmonic_tables.append(harmonics)

    carrier_harmonics = []
    _, _, _, _, carrier_max_harmonic = _PLANET_RUNTIMES[body_id]
    if carrier_table is not None and carrier_max_harmonic:
        carrier, carrier_rate = _polynomial_first(
            carrier_table[0], carrier_table[1], u, du_djd
        )
        base = complex(math.cos(carrier), math.sin(carrier))
        base_rate = base * (1j * carrier_rate)
        carrier_harmonics = [(base, base_rate)]
        for _ in range(1, carrier_max_harmonic):
            previous, previous_rate = carrier_harmonics[-1]
            carrier_harmonics.append(
                (previous * base, previous_rate * base + previous * base_rate)
            )

    for factors, carrier_harmonic, amplitudes in arguments:
        if carrier_harmonic is not None:
            harmonic, negative = carrier_harmonic
            phase, phase_rate = carrier_harmonics[harmonic]
            if negative:
                phase, phase_rate = phase.conjugate(), phase_rate.conjugate()
        else:
            phase, phase_rate = 1.0 + 0.0j, 0.0 + 0.0j
            for angle_index, harmonic, negative in factors:
                factor, factor_rate = harmonic_tables[angle_index][harmonic]
                if negative:
                    factor, factor_rate = factor.conjugate(), factor_rate.conjugate()
                phase, phase_rate = (
                    phase * factor,
                    phase_rate * factor + phase * factor_rate,
                )

        for column, amplitude_table in enumerate(amplitudes):
            real_table, imaginary_table = amplitude_table
            real, real_rate = _polynomial_first(
                real_table[0], real_table[1], u, du_djd
            )
            imaginary, imaginary_rate = _polynomial_first(
                imaginary_table[0], imaginary_table[1], u, du_djd
            )
            phase_real, phase_imaginary = phase.real, phase.imag
            phase_rate_real, phase_rate_imaginary = phase_rate.real, phase_rate.imag
            term = real * phase_real - imaginary * phase_imaginary
            term_rate = (
                real_rate * phase_real + real * phase_rate_real
                - imaginary_rate * phase_imaginary
                - imaginary * phase_rate_imaginary
            )
            if column == 0:
                longitude += term
                longitude_rate += term_rate
            elif column == 1:
                latitude += term
                latitude_rate += term_rate
            else:
                log_radius += term
                log_radius_rate += term_rate

    radius = model["radius_scale_km"] * math.exp(log_radius)
    radius_rate = radius * log_radius_rate
    sine_latitude, cosine_latitude = math.sin(latitude), math.cos(latitude)
    sine_longitude, cosine_longitude = math.sin(longitude), math.cos(longitude)
    sine_latitude_rate = cosine_latitude * latitude_rate
    cosine_latitude_rate = -sine_latitude * latitude_rate
    sine_longitude_rate = cosine_longitude * longitude_rate
    cosine_longitude_rate = -sine_longitude * longitude_rate
    radius_cosine_latitude = radius * cosine_latitude
    radius_cosine_latitude_rate = (
        radius_rate * cosine_latitude + radius * cosine_latitude_rate
    )
    return (
        (
            radius_cosine_latitude * cosine_longitude,
            radius_cosine_latitude * sine_longitude,
            radius * sine_latitude,
        ),
        (
            radius_cosine_latitude_rate * cosine_longitude
            + radius_cosine_latitude * cosine_longitude_rate,
            radius_cosine_latitude_rate * sine_longitude
            + radius_cosine_latitude * sine_longitude_rate,
            radius_rate * sine_latitude + radius * sine_latitude_rate,
        ),
    )


def _planet_ecliptic_acceleration(
    body_id: int, jd_tdb: float
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    model = PLANET_MODELS[body_id]
    if not model["jd_start"] <= jd_tdb <= model["jd_end"]:
        raise ValueError(
            f"JD {jd_tdb} is outside the supported interval "
            f"[{model['jd_start']}, {model['jd_end']}]"
        )
    u = (jd_tdb - model["epoch_jd"]) / model["half_span_days"]
    du_djd = 1.0 / model["half_span_days"]
    secular, angle_tables, arguments, carrier_table = _PLANET_DERIVATIVE_RUNTIMES[body_id]
    channels = [
        _polynomial_second(table[0], table[1], table[2], u, du_djd)
        for table in secular
    ]
    longitude, longitude_rate, longitude_acceleration = channels[0]
    latitude, latitude_rate, latitude_acceleration = channels[1]
    log_radius, log_radius_rate, log_radius_acceleration = channels[2]
    _, max_harmonics, _, _, _ = _PLANET_RUNTIMES[body_id]

    harmonic_tables = []
    for table, maximum in zip(angle_tables, max_harmonics):
        angle, angle_rate, angle_acceleration = _polynomial_second(
            table[0], table[1], table[2], u, du_djd
        )
        if maximum == 0:
            harmonic_tables.append([])
            continue
        base = complex(math.cos(angle), math.sin(angle))
        base_rate = base * (1j * angle_rate)
        base_acceleration = base * (
            1j * angle_acceleration - angle_rate * angle_rate
        )
        harmonics = [(base, base_rate, base_acceleration)]
        for _ in range(1, maximum):
            previous, previous_rate, previous_acceleration = harmonics[-1]
            harmonics.append(
                (
                    previous * base,
                    previous_rate * base + previous * base_rate,
                    previous_acceleration * base
                    + 2.0 * previous_rate * base_rate
                    + previous * base_acceleration,
                )
            )
        harmonic_tables.append(harmonics)

    carrier_harmonics = []
    _, _, _, _, carrier_max_harmonic = _PLANET_RUNTIMES[body_id]
    if carrier_table is not None and carrier_max_harmonic:
        carrier, carrier_rate, carrier_acceleration = _polynomial_second(
            carrier_table[0], carrier_table[1], carrier_table[2], u, du_djd
        )
        base = complex(math.cos(carrier), math.sin(carrier))
        base_rate = base * (1j * carrier_rate)
        base_acceleration = base * (
            1j * carrier_acceleration - carrier_rate * carrier_rate
        )
        carrier_harmonics = [(base, base_rate, base_acceleration)]
        for _ in range(1, carrier_max_harmonic):
            previous, previous_rate, previous_acceleration = carrier_harmonics[-1]
            carrier_harmonics.append(
                (
                    previous * base,
                    previous_rate * base + previous * base_rate,
                    previous_acceleration * base
                    + 2.0 * previous_rate * base_rate
                    + previous * base_acceleration,
                )
            )

    for factors, carrier_harmonic, amplitudes in arguments:
        if carrier_harmonic is not None:
            harmonic, negative = carrier_harmonic
            phase, phase_rate, phase_acceleration = carrier_harmonics[harmonic]
            if negative:
                phase = phase.conjugate()
                phase_rate = phase_rate.conjugate()
                phase_acceleration = phase_acceleration.conjugate()
        else:
            phase, phase_rate, phase_acceleration = (
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            )
            for angle_index, harmonic, negative in factors:
                factor, factor_rate, factor_acceleration = harmonic_tables[angle_index][harmonic]
                if negative:
                    factor = factor.conjugate()
                    factor_rate = factor_rate.conjugate()
                    factor_acceleration = factor_acceleration.conjugate()
                phase, phase_rate, phase_acceleration = (
                    phase * factor,
                    phase_rate * factor + phase * factor_rate,
                    phase_acceleration * factor
                    + 2.0 * phase_rate * factor_rate
                    + phase * factor_acceleration,
                )

        for column, amplitude_table in enumerate(amplitudes):
            real_table, imaginary_table = amplitude_table
            real, real_rate, real_acceleration = _polynomial_second(
                real_table[0], real_table[1], real_table[2], u, du_djd
            )
            imaginary, imaginary_rate, imaginary_acceleration = _polynomial_second(
                imaginary_table[0], imaginary_table[1], imaginary_table[2], u, du_djd
            )
            phase_real, phase_imaginary = phase.real, phase.imag
            phase_rate_real, phase_rate_imaginary = phase_rate.real, phase_rate.imag
            phase_acceleration_real, phase_acceleration_imaginary = (
                phase_acceleration.real,
                phase_acceleration.imag,
            )
            term = real * phase_real - imaginary * phase_imaginary
            term_rate = (
                real_rate * phase_real + real * phase_rate_real
                - imaginary_rate * phase_imaginary
                - imaginary * phase_rate_imaginary
            )
            term_acceleration = (
                real_acceleration * phase_real
                + 2.0 * real_rate * phase_rate_real
                + real * phase_acceleration_real
                - imaginary_acceleration * phase_imaginary
                - 2.0 * imaginary_rate * phase_rate_imaginary
                - imaginary * phase_acceleration_imaginary
            )
            if column == 0:
                longitude += term
                longitude_rate += term_rate
                longitude_acceleration += term_acceleration
            elif column == 1:
                latitude += term
                latitude_rate += term_rate
                latitude_acceleration += term_acceleration
            else:
                log_radius += term
                log_radius_rate += term_rate
                log_radius_acceleration += term_acceleration

    radius = model["radius_scale_km"] * math.exp(log_radius)
    radius_rate = radius * log_radius_rate
    radius_acceleration = radius * (
        log_radius_acceleration + log_radius_rate * log_radius_rate
    )
    sine_latitude, cosine_latitude = math.sin(latitude), math.cos(latitude)
    sine_longitude, cosine_longitude = math.sin(longitude), math.cos(longitude)
    sine_latitude_rate = cosine_latitude * latitude_rate
    cosine_latitude_rate = -sine_latitude * latitude_rate
    sine_latitude_acceleration = (
        cosine_latitude * latitude_acceleration
        - sine_latitude * latitude_rate * latitude_rate
    )
    cosine_latitude_acceleration = (
        -sine_latitude * latitude_acceleration
        - cosine_latitude * latitude_rate * latitude_rate
    )
    sine_longitude_rate = cosine_longitude * longitude_rate
    cosine_longitude_rate = -sine_longitude * longitude_rate
    sine_longitude_acceleration = (
        cosine_longitude * longitude_acceleration
        - sine_longitude * longitude_rate * longitude_rate
    )
    cosine_longitude_acceleration = (
        -sine_longitude * longitude_acceleration
        - cosine_longitude * longitude_rate * longitude_rate
    )
    radius_cosine_latitude = radius * cosine_latitude
    radius_cosine_latitude_rate = (
        radius_rate * cosine_latitude + radius * cosine_latitude_rate
    )
    radius_cosine_latitude_acceleration = (
        radius_acceleration * cosine_latitude
        + 2.0 * radius_rate * cosine_latitude_rate
        + radius * cosine_latitude_acceleration
    )
    return (
        (
            radius_cosine_latitude * cosine_longitude,
            radius_cosine_latitude * sine_longitude,
            radius * sine_latitude,
        ),
        (
            radius_cosine_latitude_rate * cosine_longitude
            + radius_cosine_latitude * cosine_longitude_rate,
            radius_cosine_latitude_rate * sine_longitude
            + radius_cosine_latitude * sine_longitude_rate,
            radius_rate * sine_latitude + radius * sine_latitude_rate,
        ),
        (
            radius_cosine_latitude_acceleration * cosine_longitude
            + 2.0 * radius_cosine_latitude_rate * cosine_longitude_rate
            + radius_cosine_latitude * cosine_longitude_acceleration,
            radius_cosine_latitude_acceleration * sine_longitude
            + 2.0 * radius_cosine_latitude_rate * sine_longitude_rate
            + radius_cosine_latitude * sine_longitude_acceleration,
            radius_acceleration * sine_latitude
            + 2.0 * radius_rate * sine_latitude_rate
            + radius * sine_latitude_acceleration,
        ),
    )




def _rotate_x_velocity(
    x: float, y: float, z: float, x_rate: float, y_rate: float,
    z_rate: float, angle: float, angle_rate: float,
) -> tuple[float, float, float, float, float, float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    cosine_rate, sine_rate = -sine * angle_rate, cosine * angle_rate
    return (
        x,
        cosine * y - sine * z,
        sine * y + cosine * z,
        x_rate,
        cosine_rate * y + cosine * y_rate - sine_rate * z - sine * z_rate,
        sine_rate * y + sine * y_rate + cosine_rate * z + cosine * z_rate,
    )


def _rotate_z_velocity(
    x: float, y: float, z: float, x_rate: float, y_rate: float,
    z_rate: float, angle: float, angle_rate: float,
) -> tuple[float, float, float, float, float, float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    cosine_rate, sine_rate = -sine * angle_rate, cosine * angle_rate
    return (
        cosine * x - sine * y,
        sine * x + cosine * y,
        z,
        cosine_rate * x + cosine * x_rate - sine_rate * y - sine * y_rate,
        sine_rate * x + sine * x_rate + cosine_rate * y + cosine * y_rate,
        z_rate,
    )


def _rotate_x_acceleration(
    x: float, y: float, z: float, x_rate: float, y_rate: float,
    z_rate: float, x_acceleration: float, y_acceleration: float,
    z_acceleration: float, angle: float, angle_rate: float,
    angle_acceleration: float,
) -> tuple[float, ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    cosine_rate, sine_rate = -sine * angle_rate, cosine * angle_rate
    cosine_acceleration = -sine * angle_acceleration - cosine * angle_rate * angle_rate
    sine_acceleration = cosine * angle_acceleration - sine * angle_rate * angle_rate
    return (
        x,
        cosine * y - sine * z,
        sine * y + cosine * z,
        x_rate,
        cosine_rate * y + cosine * y_rate - sine_rate * z - sine * z_rate,
        sine_rate * y + sine * y_rate + cosine_rate * z + cosine * z_rate,
        x_acceleration,
        cosine_acceleration * y + 2.0 * cosine_rate * y_rate + cosine * y_acceleration
        - sine_acceleration * z - 2.0 * sine_rate * z_rate - sine * z_acceleration,
        sine_acceleration * y + 2.0 * sine_rate * y_rate + sine * y_acceleration
        + cosine_acceleration * z + 2.0 * cosine_rate * z_rate + cosine * z_acceleration,
    )


def _rotate_z_acceleration(
    x: float, y: float, z: float, x_rate: float, y_rate: float,
    z_rate: float, x_acceleration: float, y_acceleration: float,
    z_acceleration: float, angle: float, angle_rate: float,
    angle_acceleration: float,
) -> tuple[float, ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    cosine_rate, sine_rate = -sine * angle_rate, cosine * angle_rate
    cosine_acceleration = -sine * angle_acceleration - cosine * angle_rate * angle_rate
    sine_acceleration = cosine * angle_acceleration - sine * angle_rate * angle_rate
    return (
        cosine * x - sine * y,
        sine * x + cosine * y,
        z,
        cosine_rate * x + cosine * x_rate - sine_rate * y - sine * y_rate,
        sine_rate * x + sine * x_rate + cosine_rate * y + cosine * y_rate,
        z_rate,
        cosine_acceleration * x + 2.0 * cosine_rate * x_rate + cosine * x_acceleration
        - sine_acceleration * y - 2.0 * sine_rate * y_rate - sine * y_acceleration,
        sine_acceleration * x + 2.0 * sine_rate * x_rate + sine * x_acceleration
        + cosine_acceleration * y + 2.0 * cosine_rate * y_rate + cosine * y_acceleration,
        z_acceleration,
    )


def _date_ecliptic_to_j2000_velocity(
    t: float, longitude: tuple[float, float], latitude: tuple[float, float],
    radius: tuple[float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    longitude_value, longitude_rate = longitude
    latitude_value, latitude_rate = latitude
    radius_value, radius_rate = radius
    cosine_latitude, sine_latitude = math.cos(latitude_value), math.sin(latitude_value)
    cosine_longitude, sine_longitude = math.cos(longitude_value), math.sin(longitude_value)
    cosine_latitude_rate = -sine_latitude * latitude_rate
    sine_latitude_rate = cosine_latitude * latitude_rate
    cosine_longitude_rate = -sine_longitude * longitude_rate
    sine_longitude_rate = cosine_longitude * longitude_rate
    x = cosine_latitude * cosine_longitude
    y = cosine_latitude * sine_longitude
    z = sine_latitude
    x_rate = cosine_latitude_rate * cosine_longitude + cosine_latitude * cosine_longitude_rate
    y_rate = cosine_latitude_rate * sine_longitude + cosine_latitude * sine_longitude_rate
    z_rate = sine_latitude_rate
    dt_djd = 1.0 / 36525.0
    for name, sign in (("epsilon", 1.0), ("chi", 1.0), ("omega", -1.0), ("phi", -1.0)):
        angle, angle_rate = _polynomial_first(
            _P03_ECLIPTIC_ANGLE_TABLES[name][0],
            _P03_ECLIPTIC_ANGLE_TABLES[name][1],
            t, dt_djd,
        )
        angle /= ARCSEC_PER_RADIAN
        angle_rate /= ARCSEC_PER_RADIAN
        x, y, z, x_rate, y_rate, z_rate = (
            _rotate_x_velocity(x, y, z, x_rate, y_rate, z_rate, sign * angle, sign * angle_rate)
            if name in ("epsilon", "omega")
            else _rotate_z_velocity(x, y, z, x_rate, y_rate, z_rate, sign * angle, sign * angle_rate)
        )
    h = math.hypot(x, y)
    h_rate = (x * x_rate + y * y_rate) / h
    return (
        (math.atan2(y, x), (x * y_rate - y * x_rate) / (x * x + y * y)),
        (math.atan2(z, h), (h * z_rate - z * h_rate) / (h * h + z * z)),
        (radius_value, radius_rate),
    )


def _date_ecliptic_to_j2000_acceleration(
    t: float, longitude: tuple[float, float, float], latitude: tuple[float, float, float],
    radius: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    longitude_value, longitude_rate, longitude_acceleration = longitude
    latitude_value, latitude_rate, latitude_acceleration = latitude
    radius_value, radius_rate, radius_acceleration = radius
    cosine_latitude, sine_latitude = math.cos(latitude_value), math.sin(latitude_value)
    cosine_longitude, sine_longitude = math.cos(longitude_value), math.sin(longitude_value)
    cosine_latitude_rate = -sine_latitude * latitude_rate
    sine_latitude_rate = cosine_latitude * latitude_rate
    cosine_latitude_acceleration = -sine_latitude * latitude_acceleration - cosine_latitude * latitude_rate * latitude_rate
    sine_latitude_acceleration = cosine_latitude * latitude_acceleration - sine_latitude * latitude_rate * latitude_rate
    cosine_longitude_rate = -sine_longitude * longitude_rate
    sine_longitude_rate = cosine_longitude * longitude_rate
    cosine_longitude_acceleration = -sine_longitude * longitude_acceleration - cosine_longitude * longitude_rate * longitude_rate
    sine_longitude_acceleration = cosine_longitude * longitude_acceleration - sine_longitude * longitude_rate * longitude_rate
    x = cosine_latitude * cosine_longitude
    y = cosine_latitude * sine_longitude
    z = sine_latitude
    x_rate = cosine_latitude_rate * cosine_longitude + cosine_latitude * cosine_longitude_rate
    y_rate = cosine_latitude_rate * sine_longitude + cosine_latitude * sine_longitude_rate
    z_rate = sine_latitude_rate
    x_acceleration = cosine_latitude_acceleration * cosine_longitude + 2.0 * cosine_latitude_rate * cosine_longitude_rate + cosine_latitude * cosine_longitude_acceleration
    y_acceleration = cosine_latitude_acceleration * sine_longitude + 2.0 * cosine_latitude_rate * sine_longitude_rate + cosine_latitude * sine_longitude_acceleration
    z_acceleration = sine_latitude_acceleration
    dt_djd = 1.0 / 36525.0
    for name, sign in (("epsilon", 1.0), ("chi", 1.0), ("omega", -1.0), ("phi", -1.0)):
        angle, angle_rate, angle_acceleration = _polynomial_second(
            _P03_ECLIPTIC_ANGLE_TABLES[name][0],
            _P03_ECLIPTIC_ANGLE_TABLES[name][1],
            _P03_ECLIPTIC_ANGLE_TABLES[name][2],
            t, dt_djd,
        )
        angle /= ARCSEC_PER_RADIAN
        angle_rate /= ARCSEC_PER_RADIAN
        angle_acceleration /= ARCSEC_PER_RADIAN
        rotated = (
            _rotate_x_acceleration if name in ("epsilon", "omega") else _rotate_z_acceleration
        )(
            x, y, z, x_rate, y_rate, z_rate, x_acceleration, y_acceleration,
            z_acceleration, sign * angle, sign * angle_rate, sign * angle_acceleration,
        )
        x, y, z, x_rate, y_rate, z_rate, x_acceleration, y_acceleration, z_acceleration = rotated
    h = math.hypot(x, y)
    h_rate = (x * x_rate + y * y_rate) / h
    h_acceleration = (x * x_acceleration + y * y_acceleration + x_rate * x_rate + y_rate * y_rate) / h - h_rate * h_rate / h
    longitude_denominator = x * x + y * y
    longitude_numerator = x * y_rate - y * x_rate
    longitude_denominator_rate = 2.0 * (x * x_rate + y * y_rate)
    longitude_numerator_rate = x * y_acceleration - y * x_acceleration
    latitude_denominator = h * h + z * z
    latitude_numerator = h * z_rate - z * h_rate
    latitude_denominator_rate = 2.0 * (h * h_rate + z * z_rate)
    latitude_numerator_rate = h * z_acceleration - z * h_acceleration
    return (
        (
            math.atan2(y, x),
            longitude_numerator / longitude_denominator,
            longitude_numerator_rate / longitude_denominator
            - longitude_numerator * longitude_denominator_rate
            / (longitude_denominator * longitude_denominator),
        ),
        (
            math.atan2(z, h),
            latitude_numerator / latitude_denominator,
            latitude_numerator_rate / latitude_denominator
            - latitude_numerator * latitude_denominator_rate
            / (latitude_denominator * latitude_denominator),
        ),
        (radius_value, radius_rate, radius_acceleration),
    )




def _moon_geocentric_ecliptic_velocity(
    jd_tdb: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    epoch = LUNAR_CORRECTION["epoch_jd"]
    half_span = LUNAR_CORRECTION["half_span_days"]
    u = (jd_tdb - epoch) / half_span
    if not -1.0 <= u <= 1.0:
        raise ValueError(
            f"JD {jd_tdb} is outside the supported lunar interval "
            f"[{epoch - half_span}, {epoch + half_span}]"
        )
    t = (jd_tdb - J2000) / 36525.0
    base = _date_ecliptic_to_j2000_velocity(
        t,
        _xl1_coordinate_velocity(t, 0),
        _xl1_coordinate_velocity(t, 1),
        _xl1_coordinate_velocity(t, 2),
    )
    longitude, latitude, radius = base
    log_radius = math.log(radius[0] / LUNAR_CORRECTION["radius_scale_km"])
    log_radius_rate = radius[1] / radius[0]
    channels = [longitude, latitude, (log_radius, log_radius_rate)]
    for index, channel in enumerate(LUNAR_CORRECTION["channels"]):
        source = channel["source_coordinate"]
        correction = _moon_correction_velocity(
            channel, t, u, _LUNAR_CORRECTION_PHASE_TABLES[index]
        )
        channels[source] = (
            channels[source][0] + correction[0],
            channels[source][1] + correction[1],
        )
    longitude, latitude, log_radius = channels
    radius_value = LUNAR_CORRECTION["radius_scale_km"] * math.exp(log_radius[0])
    radius_rate = radius_value * log_radius[1]
    sine_latitude, cosine_latitude = math.sin(latitude[0]), math.cos(latitude[0])
    sine_longitude, cosine_longitude = math.sin(longitude[0]), math.cos(longitude[0])
    sine_latitude_rate = cosine_latitude * latitude[1]
    cosine_latitude_rate = -sine_latitude * latitude[1]
    sine_longitude_rate = cosine_longitude * longitude[1]
    cosine_longitude_rate = -sine_longitude * longitude[1]
    radius_cosine_latitude = radius_value * cosine_latitude
    radius_cosine_latitude_rate = radius_rate * cosine_latitude + radius_value * cosine_latitude_rate
    return (
        (
            radius_cosine_latitude * cosine_longitude,
            radius_cosine_latitude * sine_longitude,
            radius_value * sine_latitude,
        ),
        (
            radius_cosine_latitude_rate * cosine_longitude + radius_cosine_latitude * cosine_longitude_rate,
            radius_cosine_latitude_rate * sine_longitude + radius_cosine_latitude * sine_longitude_rate,
            radius_rate * sine_latitude + radius_value * sine_latitude_rate,
        ),
    )


def _moon_geocentric_ecliptic_acceleration(
    jd_tdb: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    epoch = LUNAR_CORRECTION["epoch_jd"]
    half_span = LUNAR_CORRECTION["half_span_days"]
    u = (jd_tdb - epoch) / half_span
    if not -1.0 <= u <= 1.0:
        raise ValueError(
            f"JD {jd_tdb} is outside the supported lunar interval "
            f"[{epoch - half_span}, {epoch + half_span}]"
        )
    t = (jd_tdb - J2000) / 36525.0
    longitude, latitude, radius = _date_ecliptic_to_j2000_acceleration(
        t,
        _xl1_coordinate_acceleration(t, 0),
        _xl1_coordinate_acceleration(t, 1),
        _xl1_coordinate_acceleration(t, 2),
    )
    log_radius = math.log(radius[0] / LUNAR_CORRECTION["radius_scale_km"])
    log_radius_rate = radius[1] / radius[0]
    log_radius_acceleration = radius[2] / radius[0] - log_radius_rate * log_radius_rate
    channels = [longitude, latitude, (log_radius, log_radius_rate, log_radius_acceleration)]
    for index, channel in enumerate(LUNAR_CORRECTION["channels"]):
        source = channel["source_coordinate"]
        correction = _moon_correction_acceleration(
            channel, t, u, _LUNAR_CORRECTION_PHASE_TABLES[index]
        )
        channels[source] = tuple(
            channels[source][component] + correction[component]
            for component in range(3)
        )
    longitude, latitude, log_radius = channels
    radius_value = LUNAR_CORRECTION["radius_scale_km"] * math.exp(log_radius[0])
    radius_rate = radius_value * log_radius[1]
    radius_acceleration = radius_value * (log_radius[2] + log_radius[1] * log_radius[1])
    sine_latitude, cosine_latitude = math.sin(latitude[0]), math.cos(latitude[0])
    sine_longitude, cosine_longitude = math.sin(longitude[0]), math.cos(longitude[0])
    sine_latitude_rate = cosine_latitude * latitude[1]
    cosine_latitude_rate = -sine_latitude * latitude[1]
    sine_latitude_acceleration = cosine_latitude * latitude[2] - sine_latitude * latitude[1] * latitude[1]
    cosine_latitude_acceleration = -sine_latitude * latitude[2] - cosine_latitude * latitude[1] * latitude[1]
    sine_longitude_rate = cosine_longitude * longitude[1]
    cosine_longitude_rate = -sine_longitude * longitude[1]
    sine_longitude_acceleration = cosine_longitude * longitude[2] - sine_longitude * longitude[1] * longitude[1]
    cosine_longitude_acceleration = -sine_longitude * longitude[2] - cosine_longitude * longitude[1] * longitude[1]
    radius_cosine_latitude = radius_value * cosine_latitude
    radius_cosine_latitude_rate = radius_rate * cosine_latitude + radius_value * cosine_latitude_rate
    radius_cosine_latitude_acceleration = radius_acceleration * cosine_latitude + 2.0 * radius_rate * cosine_latitude_rate + radius_value * cosine_latitude_acceleration
    return (
        (
            radius_cosine_latitude * cosine_longitude,
            radius_cosine_latitude * sine_longitude,
            radius_value * sine_latitude,
        ),
        (
            radius_cosine_latitude_rate * cosine_longitude + radius_cosine_latitude * cosine_longitude_rate,
            radius_cosine_latitude_rate * sine_longitude + radius_cosine_latitude * sine_longitude_rate,
            radius_rate * sine_latitude + radius_value * sine_latitude_rate,
        ),
        (
            radius_cosine_latitude_acceleration * cosine_longitude + 2.0 * radius_cosine_latitude_rate * cosine_longitude_rate + radius_cosine_latitude * cosine_longitude_acceleration,
            radius_cosine_latitude_acceleration * sine_longitude + 2.0 * radius_cosine_latitude_rate * sine_longitude_rate + radius_cosine_latitude * sine_longitude_acceleration,
            radius_acceleration * sine_latitude + 2.0 * radius_rate * sine_latitude_rate + radius_value * sine_latitude_acceleration,
        ),
    )




def _xl1_coordinate_velocity(t: float, coordinate: int) -> tuple[float, float]:
    value = 0.0
    rate = 0.0
    dt_djd = 1.0 / 36525.0
    if coordinate == 0:
        t2, t3, t4, t5 = t * t, t * t * t, t * t * t * t, t * t * t * t * t
        value += (
            3.81034409 + 8399.684730072 * t - 3.319e-05 * t2
            + 3.11e-08 * t3 - 2.033e-10 * t4
        ) * ARCSEC_PER_RADIAN
        rate += (
            8399.684730072 - 2.0 * 3.319e-05 * t
            + 3.0 * 3.11e-08 * t2 - 4.0 * 2.033e-10 * t3
        ) * ARCSEC_PER_RADIAN
        value += 5028.792262 * t + 1.1124406 * t2 + 0.00007699 * t3 - 0.000023479 * t4 - 0.0000000178 * t5
        rate += 5028.792262 + 2.0 * 1.1124406 * t + 3.0 * 0.00007699 * t2 - 4.0 * 0.000023479 * t3 - 5.0 * 0.0000000178 * t4
        if t > 10.0:
            offset = t - 10.0
            value += -0.866 + 1.43 * offset + 0.054 * offset * offset
            rate += 1.43 + 0.108 * offset
        rate *= dt_djd
    envelope, envelope_rate = 1.0, 0.0
    for table in _LUNAR_XL1_DERIVATIVE_RUNTIME[coordinate]:
        subtotal, subtotal_rate = 0.0, 0.0
        for amplitude, phase_table in table:
            phase, phase_rate = _polynomial_first(
                phase_table[0], phase_table[1], t, dt_djd
            )
            cosine, sine = math.cos(phase), math.sin(phase)
            subtotal += amplitude * cosine
            subtotal_rate -= amplitude * sine * phase_rate
        value += envelope * subtotal
        rate += envelope_rate * subtotal + envelope * subtotal_rate
        envelope, envelope_rate = envelope * t, envelope_rate * t + envelope * dt_djd
    if coordinate != 2:
        value /= ARCSEC_PER_RADIAN
        rate /= ARCSEC_PER_RADIAN
    return value, rate


def _xl1_coordinate_acceleration(t: float, coordinate: int) -> tuple[float, float, float]:
    value = rate = acceleration = 0.0
    dt_djd = 1.0 / 36525.0
    if coordinate == 0:
        t2, t3, t4, t5 = t * t, t * t * t, t * t * t * t, t * t * t * t * t
        value += (
            3.81034409 + 8399.684730072 * t - 3.319e-05 * t2
            + 3.11e-08 * t3 - 2.033e-10 * t4
        ) * ARCSEC_PER_RADIAN
        rate += (
            8399.684730072 - 2.0 * 3.319e-05 * t
            + 3.0 * 3.11e-08 * t2 - 4.0 * 2.033e-10 * t3
        ) * ARCSEC_PER_RADIAN
        acceleration += (
            -2.0 * 3.319e-05 + 6.0 * 3.11e-08 * t - 12.0 * 2.033e-10 * t2
        ) * ARCSEC_PER_RADIAN
        value += 5028.792262 * t + 1.1124406 * t2 + 0.00007699 * t3 - 0.000023479 * t4 - 0.0000000178 * t5
        rate += 5028.792262 + 2.0 * 1.1124406 * t + 3.0 * 0.00007699 * t2 - 4.0 * 0.000023479 * t3 - 5.0 * 0.0000000178 * t4
        acceleration += 2.0 * 1.1124406 + 6.0 * 0.00007699 * t - 12.0 * 0.000023479 * t2 - 20.0 * 0.0000000178 * t3
        if t > 10.0:
            offset = t - 10.0
            value += -0.866 + 1.43 * offset + 0.054 * offset * offset
            rate += 1.43 + 0.108 * offset
            acceleration += 0.108
        rate *= dt_djd
        acceleration *= dt_djd * dt_djd
    envelope, envelope_rate, envelope_acceleration = 1.0, 0.0, 0.0
    for table in _LUNAR_XL1_DERIVATIVE_RUNTIME[coordinate]:
        subtotal = subtotal_rate = subtotal_acceleration = 0.0
        for amplitude, phase_table in table:
            phase, phase_rate, phase_acceleration = _polynomial_second(
                phase_table[0], phase_table[1], phase_table[2], t, dt_djd
            )
            cosine, sine = math.cos(phase), math.sin(phase)
            subtotal += amplitude * cosine
            subtotal_rate -= amplitude * sine * phase_rate
            subtotal_acceleration -= amplitude * (
                cosine * phase_rate * phase_rate + sine * phase_acceleration
            )
        value += envelope * subtotal
        rate += envelope_rate * subtotal + envelope * subtotal_rate
        acceleration += (
            envelope_acceleration * subtotal
            + 2.0 * envelope_rate * subtotal_rate
            + envelope * subtotal_acceleration
        )
        envelope, envelope_rate, envelope_acceleration = (
            envelope * t,
            envelope_rate * t + envelope * dt_djd,
            envelope_acceleration * t + 2.0 * envelope_rate * dt_djd,
        )
    if coordinate != 2:
        scale = 1.0 / ARCSEC_PER_RADIAN
        value, rate, acceleration = value * scale, rate * scale, acceleration * scale
    return value, rate, acceleration




def _chebyshev_velocity(u: float, du_djd: float, degree: int) -> list[tuple[float, float]]:
    values = [(1.0, 0.0)]
    if degree == 0:
        return values
    values.append((u, du_djd))
    for _ in range(2, degree + 1):
        previous, previous_rate = values[-1]
        before, before_rate = values[-2]
        values.append(
            (
                2.0 * u * previous - before,
                2.0 * du_djd * previous + 2.0 * u * previous_rate - before_rate,
            )
        )
    return values


def _chebyshev_acceleration(
    u: float, du_djd: float, degree: int
) -> list[tuple[float, float, float]]:
    values = [(1.0, 0.0, 0.0)]
    if degree == 0:
        return values
    values.append((u, du_djd, 0.0))
    for _ in range(2, degree + 1):
        previous, previous_rate, previous_acceleration = values[-1]
        before, before_rate, before_acceleration = values[-2]
        values.append(
            (
                2.0 * u * previous - before,
                2.0 * du_djd * previous + 2.0 * u * previous_rate - before_rate,
                4.0 * du_djd * previous_rate + 2.0 * u * previous_acceleration - before_acceleration,
            )
        )
    return values


def _moon_correction_velocity(
    channel: dict, t: float, u: float, phase_tables: Sequence[Any]
) -> tuple[float, float]:
    degree = max(LUNAR_CORRECTION["secular_degree"], channel["modulation_degree"])
    basis = _chebyshev_velocity(u, 1.0 / LUNAR_CORRECTION["half_span_days"], degree)
    value = sum(coefficient * item[0] for coefficient, item in zip(channel["secular_chebyshev_coefficients"], basis))
    rate = sum(coefficient * item[1] for coefficient, item in zip(channel["secular_chebyshev_coefficients"], basis))
    dt_djd = 1.0 / 36525.0
    for term, phase_table in zip(channel["terms"], phase_tables):
        phase, phase_rate = _polynomial_first(phase_table[0], phase_table[1], t, dt_djd)
        cosine, sine = math.cos(phase), math.sin(phase)
        cosine_amplitude = sum(c * item[0] for c, item in zip(term["cosine_chebyshev_coefficients"], basis))
        sine_amplitude = sum(c * item[0] for c, item in zip(term["sine_chebyshev_coefficients"], basis))
        cosine_rate = sum(c * item[1] for c, item in zip(term["cosine_chebyshev_coefficients"], basis))
        sine_rate = sum(c * item[1] for c, item in zip(term["sine_chebyshev_coefficients"], basis))
        envelope = t ** term["order"]
        envelope_rate = term["order"] * t ** (term["order"] - 1) * dt_djd if term["order"] else 0.0
        term_value = cosine_amplitude * cosine + sine_amplitude * sine
        term_rate = cosine_rate * cosine + sine_rate * sine + (-cosine_amplitude * sine + sine_amplitude * cosine) * phase_rate
        value += envelope * term_value
        rate += envelope_rate * term_value + envelope * term_rate
    return value, rate


def _moon_correction_acceleration(
    channel: dict, t: float, u: float, phase_tables: Sequence[Any]
) -> tuple[float, float, float]:
    degree = max(LUNAR_CORRECTION["secular_degree"], channel["modulation_degree"])
    basis = _chebyshev_acceleration(u, 1.0 / LUNAR_CORRECTION["half_span_days"], degree)
    value = sum(coefficient * item[0] for coefficient, item in zip(channel["secular_chebyshev_coefficients"], basis))
    rate = sum(coefficient * item[1] for coefficient, item in zip(channel["secular_chebyshev_coefficients"], basis))
    acceleration = sum(coefficient * item[2] for coefficient, item in zip(channel["secular_chebyshev_coefficients"], basis))
    dt_djd = 1.0 / 36525.0
    for term, phase_table in zip(channel["terms"], phase_tables):
        phase, phase_rate, phase_acceleration = _polynomial_second(
            phase_table[0], phase_table[1], phase_table[2], t, dt_djd
        )
        cosine, sine = math.cos(phase), math.sin(phase)
        cosine_amplitude = sum(c * item[0] for c, item in zip(term["cosine_chebyshev_coefficients"], basis))
        sine_amplitude = sum(c * item[0] for c, item in zip(term["sine_chebyshev_coefficients"], basis))
        cosine_rate = sum(c * item[1] for c, item in zip(term["cosine_chebyshev_coefficients"], basis))
        sine_rate = sum(c * item[1] for c, item in zip(term["sine_chebyshev_coefficients"], basis))
        cosine_acceleration = sum(c * item[2] for c, item in zip(term["cosine_chebyshev_coefficients"], basis))
        sine_acceleration = sum(c * item[2] for c, item in zip(term["sine_chebyshev_coefficients"], basis))
        order = term["order"]
        envelope = t ** order
        envelope_rate = order * t ** (order - 1) * dt_djd if order else 0.0
        envelope_acceleration = order * (order - 1) * t ** (order - 2) * dt_djd * dt_djd if order > 1 else 0.0
        term_value = cosine_amplitude * cosine + sine_amplitude * sine
        term_rate = cosine_rate * cosine + sine_rate * sine + (-cosine_amplitude * sine + sine_amplitude * cosine) * phase_rate
        harmonic_rate = -cosine_amplitude * sine + sine_amplitude * cosine
        harmonic_acceleration = (
            cosine_acceleration * cosine + sine_acceleration * sine
            + 2.0 * (-cosine_rate * sine + sine_rate * cosine) * phase_rate
            + harmonic_rate * phase_acceleration
            - term_value * phase_rate * phase_rate
        )
        value += envelope * term_value
        rate += envelope_rate * term_value + envelope * term_rate
        acceleration += envelope_acceleration * term_value + 2.0 * envelope_rate * term_rate + envelope * harmonic_acceleration
    return value, rate, acceleration


def _state_velocity(
    jd_tdb: float, body_id: int, icrf: bool
) -> CartesianPV:
    if not math.isfinite(jd_tdb):
        raise ValueError("jd_tdb must be finite")
    zero = (0.0, 0.0, 0.0)
    if body_id == 10:
        ecliptic = zero, zero
    elif body_id in PLANET_MODELS:
        ecliptic = _planet_ecliptic_velocity(body_id, jd_tdb)
    elif body_id in (301, 399):
        emb = _planet_ecliptic_velocity(3, jd_tdb)
        moon = _moon_geocentric_ecliptic_velocity(jd_tdb)
        moon_fraction = 1.0 / (EARTH_MOON_MASS_RATIO + 1.0)
        moon_scale = -moon_fraction if body_id == 399 else 1.0 - moon_fraction
        ecliptic = (
            _add(emb[0], moon[0], moon_scale),
            _add(emb[1], moon[1], moon_scale),
        )
    else:
        raise KeyError(f"unsupported target id {body_id}")
    if icrf:
        ecliptic = _ecliptic_to_icrf(ecliptic[0]), _ecliptic_to_icrf(ecliptic[1])
    return CartesianPV(ecliptic[0], ecliptic[1])


def _state_acceleration(
    jd_tdb: float, body_id: int, icrf: bool
) -> CartesianPVA:
    if not math.isfinite(jd_tdb):
        raise ValueError("jd_tdb must be finite")
    zero = (0.0, 0.0, 0.0)
    if body_id == 10:
        ecliptic = zero, zero, zero
    elif body_id in PLANET_MODELS:
        ecliptic = _planet_ecliptic_acceleration(body_id, jd_tdb)
    elif body_id in (301, 399):
        emb = _planet_ecliptic_acceleration(3, jd_tdb)
        moon = _moon_geocentric_ecliptic_acceleration(jd_tdb)
        moon_fraction = 1.0 / (EARTH_MOON_MASS_RATIO + 1.0)
        moon_scale = -moon_fraction if body_id == 399 else 1.0 - moon_fraction
        ecliptic = (
            _add(emb[0], moon[0], moon_scale),
            _add(emb[1], moon[1], moon_scale),
            _add(emb[2], moon[2], moon_scale),
        )
    else:
        raise KeyError(f"unsupported target id {body_id}")
    if icrf:
        ecliptic = tuple(_ecliptic_to_icrf(vector) for vector in ecliptic)
    return CartesianPVA(ecliptic[0], ecliptic[1], ecliptic[2])


def position_velocity_ecliptic_j2000(
    jd_tdb: float, body_id: int
) -> CartesianPV:
    """Return J2000 ecliptic Cartesian position and velocity."""

    return _state_velocity(float(jd_tdb), body_id, False)


def position_velocity_icrf(jd_tdb: float, body_id: int) -> CartesianPV:
    """Return ICRF Cartesian position and velocity."""

    return _state_velocity(float(jd_tdb), body_id, True)


def velocity_ecliptic_j2000(
    jd_tdb: float, body_id: int
) -> tuple[float, float, float]:
    """Return only the J2000 ecliptic Cartesian velocity."""

    return _state_velocity(float(jd_tdb), body_id, False).velocity_km_per_day


def velocity_icrf(jd_tdb: float, body_id: int) -> tuple[float, float, float]:
    """Return only the ICRF Cartesian velocity."""

    return _state_velocity(float(jd_tdb), body_id, True).velocity_km_per_day


def position_velocity_acceleration_ecliptic_j2000(
    jd_tdb: float, body_id: int
) -> CartesianPVA:
    """Return J2000 ecliptic Cartesian position, velocity and acceleration."""

    return _state_acceleration(float(jd_tdb), body_id, False)


def position_velocity_acceleration_icrf(
    jd_tdb: float, body_id: int
) -> CartesianPVA:
    """Return ICRF Cartesian position, velocity and acceleration."""

    return _state_acceleration(float(jd_tdb), body_id, True)


def acceleration_ecliptic_j2000(
    jd_tdb: float, body_id: int
) -> tuple[float, float, float]:
    """Return only the J2000 ecliptic Cartesian acceleration."""

    return _state_acceleration(float(jd_tdb), body_id, False).acceleration_km_per_day2


def acceleration_icrf(
    jd_tdb: float, body_id: int
) -> tuple[float, float, float]:
    """Return only the ICRF Cartesian acceleration."""

    return _state_acceleration(float(jd_tdb), body_id, True).acceleration_km_per_day2


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
