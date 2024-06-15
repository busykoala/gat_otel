from gzip import GzipFile
from io import BytesIO

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

from otel_backend import logger


async def deserialize_traces(data: bytes) -> dict:
    try:
        with GzipFile(fileobj=BytesIO(data), mode="rb") as f:
            decompressed_data = f.read()
    except IOError:
        decompressed_data = data
    try:
        trace = trace_service_pb2.ExportTraceServiceRequest()
        trace.ParseFromString(decompressed_data)
        return MessageToDict(trace)
    except Exception as e:
        logger.error(f"Error parsing trace data: {e}")
        raise
