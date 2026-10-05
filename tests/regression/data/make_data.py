# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Builds ``force_2023.csv``, the frozen dataset of the regression checks, from the FORCE dataset (Zenodo 10.5281/zenodo.21109134, CC-BY 4.0).

    python make_data.py path/to/dataset_30min_full_imputed.csv

One year (2023) without the day of the spring clock change, hourly (the even half-hours), a dozen columns: the calendar, the temperature, the RTE national load and the Enedis load by segment.
The file is committed and never rebuilt by CI: the checks must not move because a dataset moved. Rebuild it only on purpose, and then the results
file changes with it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = {
    "date": "date", "tod": "tod", "toy": "toy", "day_type_week": "day_type_week", "day_type_jf": "day_type_jf",
    "rte_load_france": "load", "meteo_temperature_celsius_france_load": "temperature",
    "enedis_load_total_france": "enedis_total", "enedis_load_residential_total_france": "enedis_residential",
    "enedis_load_professional_total_france": "enedis_professional", "enedis_load_entreprise_total_france": "enedis_entreprise",
    "enedis_load_industrial_total_france": "enedis_industrial", "year": "year",
}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    df = pd.read_csv(sys.argv[1], sep=";", usecols=list(SOURCE_COLUMNS), encoding="utf-8-sig").rename(columns=SOURCE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["year"] == 2023) & (df["tod"] % 2 == 0)].dropna().sort_values("date").reset_index(drop=True)
    df["hour"] = (df["tod"] // 2).astype(int)
    complete = df.groupby(df["date"].dt.date)["hour"].transform("size") == 24          # the spring clock change has no 02:00: that whole day is dropped
    df = df[complete].reset_index(drop=True)
    for column in ("load", "enedis_total", "enedis_residential", "enedis_professional", "enedis_entreprise", "enedis_industrial"):
        df[column] = df[column].round(1)
    df["temperature"] = df["temperature"].round(2)
    df["toy"] = df["toy"].round(5)
    df["day_type_week"] = df["day_type_week"].astype(int)
    df["day_type_jf"] = df["day_type_jf"].astype(int)
    columns = ["date", "hour", "toy", "day_type_week", "day_type_jf", "temperature", "load", "enedis_total", "enedis_residential",
               "enedis_professional", "enedis_entreprise", "enedis_industrial"]
    out = Path(__file__).with_name("force_2023.csv")
    df[columns].to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
    Path(__file__).with_name("force_2023.sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest() + "  force_2023.csv\n", encoding="utf-8")
    print(f"{len(df)} rows, {df['date'].min()} -> {df['date'].max()} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
