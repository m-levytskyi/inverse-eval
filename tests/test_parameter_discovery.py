import numpy as np

from parameter_discovery import generate_true_sld_profile


def test_true_sld_profile_uses_fronting_and_roughness():
    params = {
        "1_layer": {
            "params": [20.0, 2.0, 3.0, 4.0, 8.0],
            "param_names": [
                "thickness",
                "amb_rough",
                "sub_rough",
                "layer_sld",
                "sub_sld",
            ],
        }
    }
    x_axis = np.array([-20.0, 0.0, 10.0, 20.0, 50.0])

    _, profile = generate_true_sld_profile(params, ambient_sld=2.0, x_axis=x_axis)

    assert np.allclose(profile, [2.0, 3.0, 4.0, 6.0, 8.0], atol=0.01)
