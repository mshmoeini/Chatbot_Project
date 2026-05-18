METRIC_ALIASES = {
    "input flow": "flow_in",
    "inlet flow": "flow_in",
    "flow in": "flow_in",
    "incoming flow": "flow_in",
    "water inflow": "flow_in",

    "output flow": "flow_out",
    "outlet flow": "flow_out",
    "flow out": "flow_out",
    "outgoing flow": "flow_out",
    "water outflow": "flow_out",

    "pressure": "pressure",
    "water pressure": "pressure",
    "network pressure": "pressure",

    "leakage": "leakage",
    "water leakage": "leakage",
    "network leakage": "leakage",

    "consumption": "consumption",
    "water consumption": "consumption",
    "usage": "consumption",
}


DISTRICT_ALIASES = {
    "marconi": "Marconi",
    "ponte": "Ponte",
    "piemonte": "Piemonte",
}


CHART_TYPE_ALIASES = {
    "line": "line",
    "line chart": "line",
    "line graph": "line",

    "bar": "bar",
    "bar chart": "bar",
    "bar graph": "bar",

    "scatter": "scatter",
    "scatter plot": "scatter",
}


AGGREGATION_ALIASES = {
    "average": "avg",
    "avg": "avg",
    "mean": "avg",

    "sum": "sum",
    "total": "sum",

    "minimum": "min",
    "min": "min",

    "maximum": "max",
    "max": "max",

    "count": "count",
    "number of": "count",
}


ENTITY_ALIAS_MAPS = {
    "metric": METRIC_ALIASES,
    "district": DISTRICT_ALIASES,
    "chart_type": CHART_TYPE_ALIASES,
    "aggregation": AGGREGATION_ALIASES,
}