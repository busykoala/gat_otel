import json
from pathlib import Path
from typing import List

import pytest

from otel_backend.trace.extract import Trace, extract_data


@pytest.mark.asyncio
async def test_unpacked_data_can_be_extracted():
    current_file_path = Path(__file__)
    project_root = current_file_path.parent.parent
    json_file_path = project_root / "tests" / "assets" / "data.json"
    with json_file_path.open("r") as f:
        data = json.load(f)

    extracted_traces: List[Trace] = []
    for trace in data:
        extracted_trace = await extract_data(trace)
        extracted_traces.extend(extracted_trace)

    assert len(extracted_traces) == 2048
