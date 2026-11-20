from .data_labels import validate_label


def simulate_battery(label: str):
    validate_label(label)
    raise NotImplementedError("Add a labeled battery scenario before running the battery model.")
