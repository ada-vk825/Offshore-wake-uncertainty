import os

from wake_uncertainty.config import FEATURES, TARGET, N_MC, RANDOM_SEED, DEFAULT_MODEL_PATH


def test_target_name():
    '''Test that the configured prediction target is power_ratio
    
    Parameters:
    --------------------
    None
    
    Returns:
    --------------------
    None

    Example:
    --------------------
    Run this test from the project root folder in the cmd
    "pytest tests/test_config.py::test_target_name"
    '''
    assert TARGET == "power_ratio"


def test_default_mc_iterations():
    '''Test that the default number of Monte Carlo iterations is 100
        
        Parameters:
        --------------------
        None
        
        Returns:
        --------------------
        None
    
        Example:
        --------------------
        Run this test from the project root folder in the cmd
        "pytest tests/test_config.py::test_default_mc_iterations"
        '''
    assert N_MC == 50


def test_random_seed():
    assert RANDOM_SEED == 42


def test_model_feature_order():
    expected_features = [
        "ws_free",
        "wd",
        "ti",
        "target_rotor_dia_m",
        "target_hub_height_m",
        "target_rated_power_kw",
        "nb_distance_km",
        "nb_direction_deg",
        "nb_n_turbines",
        "nb_spacing_D",
        "nb_rotor_dia_m",
        "nb_hub_height_m",
        "nb_rated_power_kw",
        "pi_n_blocking",
        "pi_blocking_ratio",
        "pi_blocking_distance",
        "pi_streamwise_dist_1st",
        "pi_lateral_dist_1st",
        "pi_rotor_dia_1st",
    ]

    assert FEATURES == expected_features


def test_default_model_filename():
    assert os.path.basename(DEFAULT_MODEL_PATH) == "best_model.pkl"
