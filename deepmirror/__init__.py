"""deepmirror CLI package."""

from .api import (
    authenticate,
    download_structure_prediction,
    get_predict_hlm,
    list_models,
    list_structure_tasks,
    model_info,
    model_metadata,
    predict,
    predict_hlm,
    save_token,
    structure_predict,
    train,
)

__all__ = [
    "authenticate",
    "list_models",
    "predict",
    "save_token",
    "list_structure_tasks",
    "download_structure_prediction",
    "structure_predict",
    "train",
    "model_metadata",
    "predict_hlm",
    "get_predict_hlm",
    "model_info",
]
