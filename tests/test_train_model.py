import json
from pathlib import Path
from typing import List

import pytest
import torch

from otel_backend.gat.gat import GAT
from otel_backend.gat.preprocessor import split_data
from otel_backend.trace.data_df import get_as_dataframe
from otel_backend.trace.extract import Trace, extract_data

model = GAT(
    optimizer=torch.optim.Adam,
    num_features=24,
    num_classes=2,
    weight_decay=1e-3,
    dropout=0.7,
    hidden_dim=16,
    lr=0.005,
    patience=3,
)


@pytest.mark.asyncio
async def test_training_model():
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
    data = split_data(traces_df)
    model.train_model(data)

    # use first few traces to test
    first_trace = extracted_traces[:20]
    first_trace_df = await get_as_dataframe(first_trace)
    first_trace_data = split_data(first_trace_df)
    prediction = model.test_model(first_trace_data)
    expected = torch.zeros(20, dtype=torch.long)

    assert torch.equal(
        prediction, expected
    ), f"Prediction: {prediction}, Expected: {expected}"
