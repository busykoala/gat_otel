import asyncio

import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, Request

from otel_backend import logger
from otel_backend.gat.gat import GAT
from otel_backend.gat.preprocessor import split_data
from otel_backend.trace.data_df import get_as_dataframe
from otel_backend.trace.deserializers import deserialize_traces
from otel_backend.trace.extract import extract_data
from otel_backend.trace.models import TraceResponse

app = FastAPI()
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

GLOBAL_ANOMALY_ROWS = pd.DataFrame()
training_lock = asyncio.Lock()


async def store_anomalies(data, traces_df):
    global GLOBAL_ANOMALY_ROWS
    predictions = model.test_model(data)
    predicted_indices = predictions == 1
    anomaly_rows = traces_df[predicted_indices.numpy()]
    unique_anomaly_rows = anomaly_rows.drop_duplicates(
        subset=["source_namespace_label", "source_pod_label"]
    )[["source_namespace_label", "source_pod_label", "timestamp"]]
    GLOBAL_ANOMALY_ROWS = pd.concat(
        [GLOBAL_ANOMALY_ROWS, unique_anomaly_rows]
    ).drop_duplicates(subset=["source_namespace_label", "source_pod_label"])


async def process_traces(raw_data: bytes):
    async with training_lock:
        traces = None
        extracted_traces = []
        try:
            traces = await deserialize_traces(raw_data)
            extracted_traces = await extract_data(traces)
            traces_df = await get_as_dataframe(extracted_traces)
            data = split_data(traces_df)
            model.train_model(data)
            await store_anomalies(data, traces_df)
        except Exception as e:
            logger.error(f"Error processing traces: {e}")


@app.post("/v1/traces", response_model=TraceResponse)
async def receive_traces(
    request: Request, background_tasks: BackgroundTasks
) -> TraceResponse:
    raw_data = await request.body()
    background_tasks.add_task(process_traces, raw_data)
    return TraceResponse(status="received")


@app.get("/anomalies")
async def get_anomalies():
    global GLOBAL_ANOMALY_ROWS
    if not GLOBAL_ANOMALY_ROWS.empty:
        return GLOBAL_ANOMALY_ROWS.to_dict(orient="records")
    else:
        return {"message": "No anomalies detected yet."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
