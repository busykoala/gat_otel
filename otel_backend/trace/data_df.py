from dataclasses import fields, is_dataclass
from typing import Any, List, Type

import pandas as pd

from otel_backend.trace.extract import Trace


def get_all_field_names(cls: Type[Any]) -> List[str]:
    """Recursively get all field names from the dataclass and nested dataclasses."""
    if not is_dataclass(cls):
        raise TypeError("The input must be a dataclass type")

    field_names = []
    for field in fields(cls):
        if is_dataclass(field.type):
            nested_fields = get_all_field_names(field.type)
            field_names.extend(nested_fields)
        else:
            field_names.append(field.name)
    return field_names


def get_all_values(obj: Any) -> List[str]:
    """Recursively get all values from the dataclass and nested dataclasses."""
    if not is_dataclass(obj):
        raise TypeError("The input object must be a dataclass instance")

    values = []
    for field in fields(obj.__class__):
        value = getattr(obj, field.name)
        if is_dataclass(value):
            nested_values = get_all_values(value)
            values.extend(nested_values)
        else:
            values.append(value)
    return values


async def get_as_dataframe(TRACES: List[Trace]) -> pd.DataFrame:
    data = []
    headers = get_all_field_names(Trace)

    for trace in TRACES:
        values = get_all_values(trace)
        data.append(dict(zip(headers, values)))

    df = pd.DataFrame(data)
    return df
