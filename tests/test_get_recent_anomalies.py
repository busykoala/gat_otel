import json
from pathlib import Path
from typing import List

import pytest

from otel_backend.app import get_recent_anomalies
from otel_backend.trace.data_df import get_as_dataframe
from otel_backend.trace.extract import Trace, extract_data


@pytest.mark.asyncio
async def test_get_anomalies_with_no_recent_data():
    current_file_path = Path(__file__)
    project_root = current_file_path.parent.parent
    json_file_path = project_root / "tests" / "assets" / "data.json"
    with json_file_path.open("r") as f:
        data = json.load(f)

    extracted_traces: List[Trace] = []
    for trace in data:
        extracted_trace = await extract_data(trace)
        extracted_traces.extend(extracted_trace)

    traces_df = await get_as_dataframe(extracted_traces)
    traces_df = traces_df[["source_namespace_label", "source_pod_label", "timestamp"]]

    recent_anomalies = await get_recent_anomalies(traces_df)
    assert len(recent_anomalies) == 0
