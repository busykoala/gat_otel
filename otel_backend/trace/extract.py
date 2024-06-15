from dataclasses import dataclass, field
from typing import List

from otel_backend import logger


@dataclass
class TraceLabels:
    source_pod_label: str = ""
    source_namespace_label: str = ""
    source_port_label: str = ""
    destination_pod_label: str = ""
    destination_namespace_label: str = ""
    destination_port_label: str = ""
    ack_flag: bool = False
    psh_flag: bool = False


@dataclass
class Trace:
    ip_source: str = ""
    ip_destination: str = ""
    is_anomaly: bool = False
    labels: TraceLabels = field(default_factory=TraceLabels)
    timestamp: str = ""


async def extract_data(trace) -> List[Trace]:
    data = trace.get("resourceSpans", [])
    extracted_info: List[Trace] = []
    for item in data:
        trace_instance = Trace()
        try:
            for scopeSpan in item["scopeSpans"]:
                for span in scopeSpan["spans"]:
                    for attribute in span["attributes"]:
                        key, value = attribute["key"], attribute["value"]
                        if key == "cilium.flow_event.IP.source":
                            trace_instance.ip_source = value["stringValue"]
                        elif key == "cilium.flow_event.IP.destination":
                            trace_instance.ip_destination = value["stringValue"]
                        elif key == "cilium.flow_event.l4.TCP.source_port":
                            trace_instance.labels.source_port_label = value[
                                "stringValue"
                            ]
                        elif key == "cilium.flow_event.l4.TCP.destination_port":
                            trace_instance.labels.destination_port_label = value[
                                "stringValue"
                            ]
                        elif key == "cilium.flow_event.l4.TCP.flags.ACK":
                            trace_instance.labels.ack_flag = (
                                value["stringValue"] == "true"
                            )
                        elif key == "cilium.flow_event.l4.TCP.flags.PSH":
                            trace_instance.labels.psh_flag = (
                                value["stringValue"] == "true"
                            )
                        elif key == "cilium.flow_event.source.namespace":
                            trace_instance.labels.source_namespace_label = value[
                                "stringValue"
                            ]
                        elif key == "cilium.flow_event.destination.namespace":
                            trace_instance.labels.destination_namespace_label = value[
                                "stringValue"
                            ]
                        elif key == "cilium.flow_event.source.pod_name":
                            trace_instance.labels.source_pod_label = value[
                                "stringValue"
                            ]
                        elif key == "cilium.flow_event.destination.pod_name":
                            trace_instance.labels.destination_pod_label = value[
                                "stringValue"
                            ]
                        elif key == "cilium.flow_event.source.labels.is_anomaly":
                            trace_instance.is_anomaly = value["stringValue"] == "true"
                        elif key == "cilium.flow_event.time":
                            trace_instance.timestamp = value["stringValue"]
                    extracted_info.append(trace_instance)
        except KeyError as e:
            logger.error(f"Key error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
    return extracted_info
