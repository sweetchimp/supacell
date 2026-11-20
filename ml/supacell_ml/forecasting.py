from .data_labels import validate_label


def fit_forecast(label: str):
    validate_label(label)
    raise NotImplementedError("Add a labeled forecasting dataset before fitting models.")
