import numpy as np
import pytest

from compare_sld_profiles import calculate_sld_from_model


@pytest.mark.parametrize("extra_layer", ["", "layer2 5e-6 40 2\n"])
def test_sld_comparison_preserves_model_ambient(tmp_path, extra_layer):
    model = tmp_path / "model.txt"
    model.write_text(
        "fronting 2e-6 inf 1\nlayer1 4e-6 100 2\n"
        + extra_layer
        + "backing 6e-6 inf 1\n"
    )
    _, profile, parameters = calculate_sld_from_model(model, z_range=(-100, 300))
    assert np.isclose(profile[0], 2.0)
    assert np.isclose(profile[-1], 6.0)
    assert parameters["2_layer" if extra_layer else "1_layer"]["ambient_sld"] == 2.0
