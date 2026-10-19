"""Raw data loading: train and test cases, input tables and topology.

Cases are indexed 0 … 3646 in Run No order: the 2922 train cases (2015–2022) first, then
the 725 test cases (2023-01-01 → 2024-12-25). Labels exist for the train cases only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "data" / "kernel"
EXTENDED_DIR = ROOT / "data" / "extended"
HOURS = 168
LABEL_PREFIX = "result_committed_"

# Water routing per plant, from the `connections` in the topology YAML.
# Upstream: the reservoir(s) the plant takes its water from.
UPSTREAM = {
    "Hogga": ["Bandak"],
    "Tokke": ["Vinjevatn"],
    "Lio": ["Byrtevatn"],
    "Byrte": ["Botnedalsvatn"],
    "Vinje": ["Vaamarvatn"],
    "Haukeli": ["Vatjern"],
    "Songa": ["Songavatn", "Bitdalsvatn"],
    "Kjela": ["Foersvatn"],
    "Vesle Kjela": ["Kjelavatn"],
}
# Downstream: the reservoir the plant releases into (None = leaves the modelled system).
DOWNSTREAM = {
    "Hogga": None,
    "Tokke": "Bandak",
    "Lio": None,
    "Byrte": "Byrtevatn",
    "Vinje": "Vinjevatn",
    "Haukeli": "Vinjevatn",
    "Songa": "Totak",
    "Kjela": "Hyljelihyl",
    "Vesle Kjela": "Foersvatn",
}


@dataclass
class Data:
    run_no: np.ndarray  # (case,)
    starts: pd.Series  # (case,) case start times, UTC
    n_train: int  # cases 0 … n_train-1 have labels
    units: list[str]  # e.g. "Tokke_G1", in label-column order
    label_columns: list[str]  # submission column order (unit-major)
    Y: np.ndarray  # labels (train case, unit, hour)
    price: pd.Series  # day-ahead price, EUR/MWh (hourly up to 2025-09-30)
    inflow: pd.DataFrame  # hourly inflow per reservoir, m³/s
    volume: pd.DataFrame  # daily volume per reservoir, Mm³
    water_value: pd.DataFrame  # daily water value per reservoir, EUR/MWh
    max_vol: dict[str, float]  # reservoir -> max volume, Mm³
    generators: dict[str, dict]  # unit -> YAML generator entry (p_max, startcost_const, ...)

    @property
    def n_cases(self) -> int:
        return len(self.run_no)


def plant_of(unit: str) -> str:
    """'Vesle Kjela_G1' -> 'Vesle Kjela'."""
    return unit.rsplit("_", 1)[0]


def _read_timeseries(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def load_data() -> Data:
    labels = pd.read_csv(KERNEL_DIR / "Unit_commitment_decisions.csv", parse_dates=["starttime"])
    test = pd.read_csv(ROOT / "prediction_mapping.csv", parse_dates=["starttime"])
    label_columns = [c for c in labels.columns if c.startswith(LABEL_PREFIX)]
    units = list(dict.fromkeys(c.removeprefix(LABEL_PREFIX).rsplit("_t", 1)[0] for c in label_columns))

    run_no = np.concatenate([labels["Run No"].to_numpy(), test["Run No"].to_numpy()])
    if not (np.diff(run_no) == 1).all():
        raise ValueError("Run No is not consecutive across train and test")
    topology = yaml.safe_load((EXTENDED_DIR / "Tokke_Vinje_topology.yaml").read_text())["model"]

    return Data(
        run_no=run_no,
        starts=pd.concat([labels["starttime"], test["starttime"]], ignore_index=True).dt.tz_localize("UTC"),
        n_train=len(labels),
        units=units,
        label_columns=label_columns,
        Y=labels[label_columns].to_numpy().reshape(len(labels), len(units), HOURS),
        price=_read_timeseries(KERNEL_DIR / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0],
        inflow=_read_timeseries(KERNEL_DIR / "Historical_inflow_1958_2025.csv"),
        volume=_read_timeseries(KERNEL_DIR / "Historical_volume_2015_2024.csv"),
        water_value=_read_timeseries(KERNEL_DIR / "Synthetic_water_value_2015_2024.csv"),
        max_vol={name: r["max_vol"] for name, r in topology["reservoir"].items()},
        generators=topology["generator"],
    )


# --- Lookups shared by the feature groups -------------------------------------------------

def hourly_window(series: pd.Series, starts: pd.Series, offset_hours: int = 0) -> np.ndarray:
    """The 168 hourly values from each start (+ offset), shape (case, hour).

    Assumes an hourly index without gaps, which holds for 2015–2024 (EDA §5.4).
    """
    first = series.index.get_indexer(starts + pd.Timedelta(hours=offset_hours))
    if (first < 0).any():
        raise KeyError("a case start is missing from the hourly index")
    return series.to_numpy()[first[:, None] + np.arange(HOURS)]


def daily_value(table: pd.DataFrame, days: pd.Series, columns: list[str]) -> np.ndarray:
    """Mean of `columns` on each day in `days`, shape (case,)."""
    return table.loc[days, columns].mean(axis=1).to_numpy()


def week_prices(data: Data, offset_hours: int = 0) -> np.ndarray:
    """Hourly prices of each case week (offset 168 = the week after), shape (case, hour)."""
    return hourly_window(data.price, data.starts, offset_hours)


def upstream_wv(data: Data, day_offset: int) -> np.ndarray:
    """Each unit's upstream water value on case start + `day_offset` days, shape (case, unit)."""
    days = data.starts + pd.Timedelta(days=day_offset)
    return np.stack([daily_value(data.water_value, days, UPSTREAM[plant_of(u)]) for u in data.units], axis=1)


def downstream_wv(data: Data, day_offset: int) -> np.ndarray:
    """Like upstream_wv for the downstream reservoir; 0 where the water leaves the system."""
    days = data.starts + pd.Timedelta(days=day_offset)
    columns = []
    for unit in data.units:
        down = DOWNSTREAM[plant_of(unit)]
        columns.append(daily_value(data.water_value, days, [down]) if down else np.zeros(data.n_cases))
    return np.stack(columns, axis=1)
