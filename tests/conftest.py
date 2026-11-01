import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.knowledge_base import load_knowledge_base  # noqa: E402


@pytest.fixture(scope="session")
def kb():
    return load_knowledge_base()


def make_wide(voltages, currents=None, satellite="TESTSAT", session="1 Jan 2021", panel="py", step=5.0):
    """Tiny hand-made telemetry table for unit tests (not dataset data)."""
    n = len(voltages)
    currents = currents if currents is not None else [100.0] * n
    df = pd.DataFrame({config.TIME_COL: np.arange(n) * step,
                       config.SATELLITE_COL: satellite, config.SESSION_COL: session})
    for p in config.PANELS:
        df[config.voltage_col(p)] = 4000.0
        df[config.current_col(p)] = 100.0
        df[config.temperature_col(p)] = 20.0
    df[config.voltage_col(panel)] = voltages
    df[config.current_col(panel)] = currents
    return df
