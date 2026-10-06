import numpy as np

from reflectorch.inference.inference_model import EasyInferenceModel


def test_restore_ambient_sld_updates_sld_columns_for_batched_and_single_predictions():
    for shape in [(7,), (2, 7)]:
        prediction = {"predicted_params_array": np.zeros(shape)}

        EasyInferenceModel._restore_slds_after_ambient_shift(
            None, prediction, slice(3, 5), 3.5
        )

        assert np.all(prediction["predicted_params_array"][..., 3:5] == 3.5)
        assert np.all(prediction["predicted_params_array"][..., :3] == 0.0)
        assert np.all(prediction["predicted_params_array"][..., 5:] == 0.0)
