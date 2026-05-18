DATA_SCHEMA = {
    "table_name": "measurements",
    "columns": {
        "sensor_id": "string",
        "timestamp": "datetime_utc",
        "type": "string",
        "value": "float",
        "city": "string",
        "district": "string",
    },
    "time_column": "timestamp",
    "value_column": "value",
    "id_column": "sensor_id",
    "filterable_columns": ["type", "city", "district"],
}


CANONICAL_METRICS = {
    "flow_in": {
        "label": "input flow",
        "description": "Flow entering a monitored point or asset.",
        "value_column": "value",
        "filters": {
            "type": "flow_in"
        }
    },
    "flow_out": {
        "label": "output flow",
        "description": "Flow leaving a monitored point or asset.",
        "value_column": "value",
        "filters": {
            "type": "flow_out"
        }
    },
    "pressure": {
        "label": "pressure",
        "description": "Pressure measurement value.",
        "value_column": "value",
        "filters": {
            "type": "pressure"
        }
    },
    "consumption": {
        "label": "consumption",
        "description": "Consumption measurement value.",
        "value_column": "value",
        "filters": {
            "type": "consumption"
        }
    },
    "leakage": {
        "label": "leakage",
        "description": "Leakage-related measurement value.",
        "value_column": "value",
        "filters": {
            "type": "leakage"
        }
    },
}


METRIC_ALIASES = {
    "input flow": "flow_in",
    "inlet flow": "flow_in",
    "flow in": "flow_in",
    "incoming flow": "flow_in",
    "output flow": "flow_out",
    "outlet flow": "flow_out",
    "flow out": "flow_out",
    "pressure": "pressure",
    "water pressure": "pressure",
    "consumption": "consumption",
    "water consumption": "consumption",
    "leakage": "leakage",
    "water leakage": "leakage",
}


TIME_SCHEMA_HINTS = {
    "time_column": "timestamp",
    "timezone": "UTC",
    "stored_format": "datetime",
}