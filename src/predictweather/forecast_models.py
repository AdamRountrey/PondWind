from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ModelPointForecast:
    source: str
    display_name: str
    run_at_utc: str
    valid_time_utc: str
    forecast_hour: int
    site_lat: float
    site_lon: float
    grid_lat: float
    grid_lon: float
    grid_distance_km: float
    u10_mps: float
    v10_mps: float
    wind_speed_mps: float
    wind_from_direction_deg: float
    gust_mps: float | None
    sustained_kind: str
    gust_kind: str | None
    acquisition_mode: str
    files: dict[str, str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConsensusBoundary:
    selected_source: str
    selected_reason: str
    boundary: ModelPointForecast
    consensus_u10_mps: float
    consensus_v10_mps: float
    consensus_speed_mps: float
    consensus_direction_deg: float
    available_sources: list[str]
    weight_details: list[dict]
    skill_metadata: dict | None = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["boundary"] = self.boundary.as_dict()
        return payload


def vector_to_speed_direction(u_mps: float, v_mps: float) -> tuple[float, float]:
    speed_mps = float(math.hypot(u_mps, v_mps))
    direction_from_deg = float((270.0 - math.degrees(math.atan2(v_mps, u_mps))) % 360.0)
    return speed_mps, direction_from_deg


def circular_diff_deg(a_deg: float, b_deg: float) -> float:
    diff = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
    return float(diff)


def _vector_median(samples: list[ModelPointForecast]) -> tuple[float, float, float, float]:
    u_values = sorted(sample.u10_mps for sample in samples)
    v_values = sorted(sample.v10_mps for sample in samples)
    middle = len(samples) // 2
    if len(samples) % 2 == 1:
        u_mps = float(u_values[middle])
        v_mps = float(v_values[middle])
    else:
        u_mps = 0.5 * float(u_values[middle - 1] + u_values[middle])
        v_mps = 0.5 * float(v_values[middle - 1] + v_values[middle])
    speed_mps, direction_deg = vector_to_speed_direction(u_mps, v_mps)
    return u_mps, v_mps, speed_mps, direction_deg


def _agrees_with_consensus(
    sample: ModelPointForecast,
    consensus_speed_mps: float,
    consensus_direction_deg: float,
) -> bool:
    speed_tolerance_mps = max(1.5, consensus_speed_mps * 0.30)
    direction_tolerance_deg = 30.0
    return (
        abs(sample.wind_speed_mps - consensus_speed_mps) <= speed_tolerance_mps
        and circular_diff_deg(sample.wind_from_direction_deg, consensus_direction_deg) <= direction_tolerance_deg
    )


def _freshest(samples: list[ModelPointForecast]) -> ModelPointForecast:
    return max(samples, key=lambda sample: datetime.fromisoformat(sample.run_at_utc.replace("Z", "+00:00")))


def choose_consensus_boundary(
    samples: list[ModelPointForecast],
    *,
    skill_adjustments: dict[str, float] | None = None,
    skill_metadata: dict | None = None,
) -> ConsensusBoundary:
    if not samples:
        raise ValueError("No model forecasts available for consensus.")

    consensus_u10_mps, consensus_v10_mps, consensus_speed_mps, consensus_direction_deg = _vector_median(samples)
    available_sources = [sample.source for sample in samples]
    representative = _freshest(samples)

    base_resolution_weights = {
        "hrrr": 1.30,
        "hrdps": 1.30,
        "nam": 1.10,
        "icon": 1.00,
        "ecmwf": 1.00,
        "gfs": 0.95,
    }
    weighted_members: list[tuple[ModelPointForecast, float]] = []
    for sample in samples:
        base_weight = base_resolution_weights.get(sample.source, 1.0)
        distance_penalty = 1.0 / (1.0 + max(sample.grid_distance_km, 0.0) / 20.0)
        speed_error = abs(sample.wind_speed_mps - consensus_speed_mps)
        speed_scale = max(2.0, consensus_speed_mps * 0.35)
        speed_penalty = 1.0 / (1.0 + (speed_error / speed_scale) ** 2)
        direction_error = circular_diff_deg(sample.wind_from_direction_deg, consensus_direction_deg)
        direction_penalty = 1.0 / (1.0 + (direction_error / 45.0) ** 2)
        skill_factor = 1.0 if skill_adjustments is None else float(skill_adjustments.get(sample.source, 1.0))
        weight = base_weight * distance_penalty * speed_penalty * direction_penalty * skill_factor
        weighted_members.append((sample, weight))

    total_weight = sum(weight for _, weight in weighted_members)
    if total_weight <= 0.0:
        total_weight = float(len(weighted_members))
        weighted_members = [(sample, 1.0) for sample, _ in weighted_members]

    u_mps = float(sum(sample.u10_mps * weight for sample, weight in weighted_members) / total_weight)
    v_mps = float(sum(sample.v10_mps * weight for sample, weight in weighted_members) / total_weight)
    speed_mps, direction_deg = vector_to_speed_direction(u_mps, v_mps)

    gust_weight_sum = sum(weight for sample, weight in weighted_members if sample.gust_mps is not None)
    gust_mps = None
    gust_kind = None
    if gust_weight_sum > 0.0:
        gust_mps = float(
            sum(float(sample.gust_mps) * weight for sample, weight in weighted_members if sample.gust_mps is not None) / gust_weight_sum
        )
        gust_kind = "weighted model gust consensus"

    dominant_sample = max(weighted_members, key=lambda item: item[1])[0]
    boundary = ModelPointForecast(
        source="weighted_consensus",
        display_name="consensus",
        run_at_utc=representative.run_at_utc,
        valid_time_utc=representative.valid_time_utc,
        forecast_hour=representative.forecast_hour,
        site_lat=representative.site_lat,
        site_lon=representative.site_lon,
        grid_lat=dominant_sample.grid_lat,
        grid_lon=dominant_sample.grid_lon,
        grid_distance_km=dominant_sample.grid_distance_km,
        u10_mps=u_mps,
        v10_mps=v_mps,
        wind_speed_mps=speed_mps,
        wind_from_direction_deg=direction_deg,
        gust_mps=gust_mps,
        sustained_kind="10m sustained weighted consensus",
        gust_kind=gust_kind,
        acquisition_mode="weighted_consensus",
        files={sample.source: sample.files.get("UGRD", next(iter(sample.files.values()), "")) for sample, _ in weighted_members},
    )
    top_sources = sorted(weighted_members, key=lambda item: item[1], reverse=True)[:3]
    top_labels = ", ".join(sample.display_name for sample, _ in top_sources)
    if skill_metadata and skill_metadata.get("observations_used", 0) > 0:
        station_label = skill_metadata.get("station_id") or "nearby"
        reason = (
            f"Weighted vector consensus stayed close to the model mean, leaned modestly toward higher-resolution members, "
            f"and applied recent {station_label} observation skill adjustments. Strongest weights: {top_labels}."
        )
    else:
        reason = f"Weighted vector consensus stayed close to the model mean while leaning modestly toward higher-resolution members. Strongest weights: {top_labels}."
    weight_details: list[dict] = []
    for sample, raw_weight in sorted(weighted_members, key=lambda item: item[1], reverse=True):
        base_weight = base_resolution_weights.get(sample.source, 1.0)
        distance_penalty = 1.0 / (1.0 + max(sample.grid_distance_km, 0.0) / 20.0)
        speed_error = abs(sample.wind_speed_mps - consensus_speed_mps)
        speed_scale = max(2.0, consensus_speed_mps * 0.35)
        speed_penalty = 1.0 / (1.0 + (speed_error / speed_scale) ** 2)
        direction_error = circular_diff_deg(sample.wind_from_direction_deg, consensus_direction_deg)
        direction_penalty = 1.0 / (1.0 + (direction_error / 45.0) ** 2)
        skill_factor = 1.0 if skill_adjustments is None else float(skill_adjustments.get(sample.source, 1.0))
        weight_details.append(
            {
                "source": sample.source,
                "display_name": sample.display_name,
                "base_resolution_weight": base_weight,
                "distance_penalty": distance_penalty,
                "speed_penalty": speed_penalty,
                "direction_penalty": direction_penalty,
                "skill_factor": skill_factor,
                "raw_weight": raw_weight,
                "normalized_weight": raw_weight / total_weight,
            }
        )
    return ConsensusBoundary(
        selected_source=boundary.source,
        selected_reason=reason,
        boundary=boundary,
        consensus_u10_mps=consensus_u10_mps,
        consensus_v10_mps=consensus_v10_mps,
        consensus_speed_mps=consensus_speed_mps,
        consensus_direction_deg=consensus_direction_deg,
        available_sources=available_sources,
        weight_details=weight_details,
        skill_metadata=skill_metadata,
    )


def model_table_rows(samples: list[ModelPointForecast]) -> list[dict[str, str]]:
    order = {"ecmwf": 0, "gfs": 1, "icon": 2, "nam": 3, "hrdps": 4, "hrrr": 5}
    ordered = sorted(samples, key=lambda sample: (order.get(sample.source, 99), sample.display_name))
    rows: list[dict[str, str]] = []
    for sample in ordered:
        wind_kts = sample.wind_speed_mps * 1.94384449
        gust_kts = sample.gust_mps * 1.94384449 if sample.gust_mps is not None else None
        rows.append(
            {
                "model": sample.display_name.lower(),
                "wind": f"{wind_kts:.1f}",
                "gust": "n/a" if gust_kts is None else f"{gust_kts:.1f}",
            }
        )
    return rows
