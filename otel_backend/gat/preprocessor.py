import pandas as pd

from otel_backend.gat.converter import convert_to_graph
from otel_backend.gat.encoder import (
    boolean_string_to_int,
    ip_encoder,
    number_normalizer,
    string_encoder,
)


def split_data(df: pd.DataFrame):
    df = preprocess_df(df)
    X = preprocess_X(df)
    y = preprocess_y(df)
    data = convert_to_graph(X, y)
    return data


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)
    return df


def preprocess_X(df):
    X = df.drop(columns=["is_anomaly"])
    encoder_map = {
        "ip_source": ip_encoder,
        "ip_destination": ip_encoder,
        "source_pod_label": string_encoder,
        "destination_pod_label": string_encoder,
        "source_namespace_label": string_encoder,
        "destination_namespace_label": string_encoder,
        "source_port_label": number_normalizer,
        "destination_port_label": number_normalizer,
        "ack_flag": boolean_string_to_int,
        "psh_flag": boolean_string_to_int,
    }

    for column, encoder_function in encoder_map.items():
        X = encoder_function(X, column)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    features = [
        "ack_flag",
        "psh_flag",
        "ip_source_part1",
        "ip_source_part2",
        "ip_source_part3",
        "ip_source_part4",
        "ip_source_part5",
        "ip_source_part6",
        "ip_source_part7",
        "ip_source_part8",
        "ip_destination_part1",
        "ip_destination_part2",
        "ip_destination_part3",
        "ip_destination_part4",
        "ip_destination_part5",
        "ip_destination_part6",
        "ip_destination_part7",
        "ip_destination_part8",
        "source_pod_label_normalized",
        "destination_pod_label_normalized",
        "source_namespace_label_normalized",
        "destination_namespace_label_normalized",
        "source_port_label_normalized",
        "destination_port_label_normalized",
    ]
    X = X[features]
    return X


def preprocess_y(df):
    return df["is_anomaly"]
