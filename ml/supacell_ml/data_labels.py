DATA_LABELS = ("real", "synthetic", "simulated")


def validate_label(label: str) -> str:
    if label not in DATA_LABELS:
        raise ValueError(f"label must be one of {DATA_LABELS}; got {label!r}")
    return label
