"""Builds data/ for the MLflow experiment-tracking notebook.

Source: Hotel Booking Demand dataset
    Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets.
    Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126
    License: CC BY 4.0. Mirror used: TidyTuesday 2020-02-11 (hotels.csv).

Writes (byte-identical on every rebuild):
    data/bookings_2015_2017.csv.gz   bookings whose outcome is known (is_canceled)
    data/upcoming_arrivals.csv       the last 7 days of arrivals, outcome column removed

Usage:
    python _make_dataset.py                  # downloads hotels.csv
    python _make_dataset.py path/hotels.csv  # uses a local copy
"""

import sys
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
OUT = Path(__file__).parent / "data"

COLUMNS = [
    "booking_id", "arrival_date", "hotel", "lead_time",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
    "meal", "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "reserved_room_type",
    "deposit_type", "customer_type", "days_in_waiting_list", "adr",
    "required_car_parking_spaces", "total_of_special_requests", "is_canceled",
]
# Dropped on purpose: reservation_status / reservation_status_date (they ARE the outcome),
# assigned_room_type (set at check-in), country / agent / company (IDs, recorded late),
# booking_changes (accumulates after the booking is made).

raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else URL)

raw["arrival_date"] = pd.to_datetime(
    raw["arrival_date_year"].astype(str) + "-" + raw["arrival_date_month"] + "-"
    + raw["arrival_date_day_of_month"].astype(str),
    format="%Y-%B-%d",
)
df = raw.sort_values(["arrival_date", "hotel"], kind="stable").reset_index(drop=True)
df["booking_id"] = [f"BK{100001 + i}" for i in range(len(df))]
df["arrival_date"] = df["arrival_date"].dt.strftime("%Y-%m-%d")
df = df[COLUMNS]

cutoff = (pd.Timestamp(df["arrival_date"].max()) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
history = df[df["arrival_date"] < cutoff]
upcoming = df[df["arrival_date"] >= cutoff].drop(columns="is_canceled")

OUT.mkdir(exist_ok=True)
history.to_csv(OUT / "bookings_2015_2017.csv.gz", index=False,
               compression={"method": "gzip", "mtime": 0})
upcoming.to_csv(OUT / "upcoming_arrivals.csv", index=False)

print(f"history : {len(history):,} bookings, arrivals {history.arrival_date.min()} .. "
      f"{history.arrival_date.max()}, cancel rate {history.is_canceled.mean():.3f}")
print(f"upcoming: {len(upcoming):,} bookings, arrivals {cutoff} .. {upcoming.arrival_date.max()}")
