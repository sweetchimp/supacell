from .data_labels import validate_label


def run_simulation(label: str):
    validate_label(label)
    raise NotImplementedError("Add a labeled simulation input dataset before running simulations.")
