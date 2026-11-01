"""Dataset inspection: prints sheets, columns and basic statistics of every raw workbook.

Run:  python -m src.inspect_data
"""
from __future__ import annotations

import pandas as pd

from . import config


def main() -> None:
    files = sorted(config.RAW_DIR.glob("*.xlsx"))
    if not files:
        print(f"No .xlsx files in {config.RAW_DIR}. Place the BIRDS EPS workbooks there.")
        return
    for path in files:
        book = pd.ExcelFile(path)
        print("\n" + "=" * 70 + f"\nFILE: {path.name}  sheets: {book.sheet_names}")
        for sheet in book.sheet_names:
            df = book.parse(sheet)
            solar = [c for c in config.SENSOR_COLUMNS if c in df.columns]
            battery = [c for c in df.columns if str(c).startswith(config.BATTERY_COLUMN_PREFIXES)]
            ts = pd.to_numeric(df[config.TIME_COL], errors="coerce")
            print(f"\n-- sheet '{sheet}': {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"   time stamp: {ts.min()} .. {ts.max()}  (missing: {int(ts.isna().sum())}, "
                  f"typical step: {ts.diff().mode().iloc[0] if ts.notna().any() else 'n/a'} s)")
            print(f"   solar columns used ({len(solar)}): {solar}")
            print(f"   battery columns excluded: {battery}")


if __name__ == "__main__":
    main()
