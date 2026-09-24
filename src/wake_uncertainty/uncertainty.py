import sys

import numpy as np
import pandas as pd

from .config import FEATURES, KW_JENSEN, RANDOM_SEED, N_MC

from .model import load_model

from .utils import place_turbines, make_neighbour_polygon, compute_geometric_features

from .exception import CustomException
from .logger import logging


def make_target_polygon(n_turbines, spacing_D, diameter):
    """
    Create a square target wind-farm polygon based on turbine layout parameters.

    The square side length is calculated from the approximate number of
    turbines per side and the required turbine spacing. The polygon is
    centred at the origin and is used as the target-farm boundary for
    turbine placement.

    Parameters
    ----------
    n_turbines : int
        Number of turbines to be placed in the target wind farm.
    spacing_D : float
        Minimum turbine spacing expressed as a multiple of rotor diameter D.
    diameter : float
        Rotor diameter of the target turbine in metres.

    Returns
    -------
    shapely.geometry.Polygon
        Square polygon representing the target wind-farm boundary.

    Raises
    ------
    CustomException
        If the target polygon cannot be created.

    Example
    -------
    >>> target_poly = make_target_polygon(
    ...     n_turbines=50,
    ...     spacing_D=7,
    ...     diameter=164
    ... )
    >>> print(target_poly.area)
    """
    try:
        from shapely.geometry import Polygon

        turbines_per_side = int(np.ceil(np.sqrt(n_turbines)))
        target_spacing_m = spacing_D * diameter
        side_length = (turbines_per_side + 1) * target_spacing_m
        half_side = side_length / 2

        target_poly = Polygon([(-half_side, -half_side),
                               (half_side, -half_side),
                               (half_side, half_side),
                               (-half_side, half_side)])
        return target_poly

    except Exception as e:
        logging.exception(f"Failed to create target polygon: {e}")
        raise CustomException(e, sys)


def run_monte_carlo(target_config, neighbour_config, wind_config,
                    target_poly=None, n_mc=N_MC, random_seed=RANDOM_SEED):
    """
    Estimate neighbouring-farm wake loss and its uncertainty using Monte Carlo simulation.

    The function loads the trained XGBoost surrogate model, generates the
    target-farm turbine layout, and calculates a baseline target-farm power
    without the neighbouring farm. It then repeatedly samples the
    neighbouring-farm turbine spacing and placement, calculates
    physics-informed geometric wake features for the combined target and
    neighbouring farms, and uses the surrogate model to predict target-farm
    power in the presence of the neighbour.

    For each valid Monte Carlo iteration, the power loss and percentage wake
    loss are calculated. The final result is the mean wake loss, standard
    deviation of wake loss, mean power loss, and standard deviation of power
    loss.

    The neighbouring farm is represented by a square polygon whose location
    is determined by the specified separation distance and direction. The
    turbine positions within the polygon are generated using the turbine
    placement function and the selected spacing.

    Parameters
    ----------
    target_config : dict
        Configuration of the target wind farm. Expected keys include:

        - ``n_turbines`` : int
            Number of target-farm turbines.
        - ``spacing_D`` : float
            Target turbine spacing in rotor diameters.
        - ``diameter`` : float
            Target turbine rotor diameter in metres.
        - ``hub_height`` : float
            Target turbine hub height in metres.
        - ``rated_power`` : float
            Target turbine rated power in kW.

    neighbour_config : dict
        Configuration of the neighbouring wind farm. Expected keys include:

        - ``distance_km`` : float
            Distance between the target and neighbouring farm centroids in km.
        - ``direction_deg`` : float
            Direction of the neighbouring farm relative to the target farm
            in degrees.
        - ``n_turbines`` : int
            Number of turbines in the neighbouring farm.
        - ``spacing_values_D`` : list or array of float
            Candidate turbine spacings in rotor diameters from which the
            Monte Carlo simulation samples.
        - ``diameter`` : float
            Neighbouring turbine rotor diameter in metres.
        - ``hub_height`` : float
            Neighbouring turbine hub height in metres.
        - ``rated_power`` : float
            Neighbouring turbine rated power in kW.

    wind_config : dict
        Wind and atmospheric conditions used by the surrogate model.
        Expected keys include:

        - ``ws`` : float
            Free-stream wind speed in m/s.
        - ``wd`` : float
            Wind direction in degrees.
        - ``ti`` : float
            Turbulence intensity.

    target_poly : shapely.geometry.Polygon or None, optional
        Existing target-farm polygon. If ``None``, a square target polygon
        is generated using ``target_config``.

    n_mc : int, optional
        Number of Monte Carlo iterations. Defaults to ``N_MC``.

    random_seed : int, optional
        Random seed used to make turbine placement and Monte Carlo sampling
        reproducible. Defaults to ``RANDOM_SEED``.

    Returns
    -------
    dict
        Dictionary containing the estimated wake-loss statistics:

        - ``wake_loss`` : float
            Mean wake loss across valid Monte Carlo simulations, expressed
            as a percentage.
        - ``wake_uncertainty`` : float
            Standard deviation of wake loss across Monte Carlo simulations,
            expressed in percentage points.
        - ``power_loss`` : float
            Mean target-farm power loss, expressed in MW.
        - ``power_uncertainty`` : float
            Standard deviation of target-farm power loss, expressed in MW.

    Raises
    ------
    CustomException
        If an unexpected error occurs during the Monte Carlo simulation.

    Example
    -------
    >>> target_config = {
    ...     "n_turbines": 50,
    ...     "spacing_D": 7,
    ...     "diameter": 164,
    ...     "hub_height": 105,
    ...     "rated_power": 8400
    ... }
    >>> neighbour_config = {
    ...     "distance_km": 20,
    ...     "direction_deg": 180,
    ...     "n_turbines": 50,
    ...     "spacing_values_D": [3, 5, 7, 9],
    ...     "diameter": 178,
    ...     "hub_height": 119,
    ...     "rated_power": 10000
    ... }
    >>> wind_config = {
    ...     "ws": 10,
    ...     "wd": 180,
    ...     "ti": 0.06
    ... }
    >>> results = run_monte_carlo(
    ...     target_config,
    ...     neighbour_config,
    ...     wind_config,
    ...     n_mc=100,
    ...     random_seed=42
    ... )
    >>> print(results["wake_loss"])
    >>> print(results["wake_uncertainty"])
    """
    try:
        model_bundle = load_model()

        if isinstance(model_bundle, dict):
            model = model_bundle["model"]
            features = model_bundle.get("features", FEATURES)

        else:
            model = model_bundle
            features = FEATURES

        rng = np.random.default_rng(random_seed)

        if target_poly is None:
            target_poly = make_target_polygon(target_config["n_turbines"],
                                              target_config["spacing_D"],
                                              target_config["diameter"])

        target_cx = float(target_poly.centroid.x)
        target_cy = float(target_poly.centroid.y)
        target_spacing_m = (target_config["spacing_D"] * target_config["diameter"])

        x_target, y_target, _ = place_turbines(target_poly, target_config["n_turbines"],
                                               target_spacing_m, seed=random_seed)

        if len(x_target) != target_config["n_turbines"]:
            raise ValueError("Could not place all target turbines.")

        x_target = np.asarray(x_target)
        y_target = np.asarray(y_target)
        n_target = len(x_target)
        target_D = np.full(n_target, target_config["diameter"])

        baseline_rows = []

        baseline_spacing_D = float(np.mean(neighbour_config["spacing_values_D"]))

        for t_idx in range(n_target):
            mask = np.ones(n_target, dtype=bool)
            mask[t_idx] = False
            geo_feats = compute_geometric_features(
                    x_target[t_idx],
                    y_target[t_idx],
                    target_D[t_idx],
                    x_target[mask],
                    y_target[mask],
                    target_D[mask],
                    wind_config["wd"],
                    kw=KW_JENSEN
                )

            baseline_rows.append({
                "ws_free": wind_config["ws"],
                "wd": wind_config["wd"],
                "ti": wind_config["ti"],
                "target_rotor_dia_m": target_config["diameter"],
                "target_hub_height_m": target_config["hub_height"],
                "target_rated_power_kw": target_config["rated_power"],
                "nb_distance_km": neighbour_config["distance_km"],
                "nb_direction_deg": neighbour_config["direction_deg"],
                "nb_n_turbines": neighbour_config["n_turbines"],
                "nb_spacing_D": baseline_spacing_D,
                "nb_rotor_dia_m": neighbour_config["diameter"],
                "nb_hub_height_m": neighbour_config["hub_height"],
                "nb_rated_power_kw": neighbour_config["rated_power"],
                "pi_n_blocking": geo_feats["pi_n_blocking"],
                "pi_blocking_ratio": geo_feats["pi_blocking_ratio"],
                "pi_blocking_distance": geo_feats["pi_blocking_distance"],
                "pi_streamwise_dist_1st": geo_feats["pi_streamwise_dist_1st"],
                "pi_lateral_dist_1st": geo_feats["pi_lateral_dist_1st"],
                "pi_rotor_dia_1st": geo_feats["pi_rotor_dia_1st"]})

        baseline_df = pd.DataFrame(baseline_rows)
        baseline_ratio = model.predict(baseline_df[features])
        baseline_ratio = np.clip(baseline_ratio, 0, 1)
        baseline_power_kw = (baseline_ratio * target_config["rated_power"])
        P_without_neighbour = float(baseline_power_kw.sum())

        if P_without_neighbour <= 0:
            raise ValueError("Baseline power is zero.")

        wake_losses = []
        power_losses = []

        for mc_iter in range(n_mc):
            spacing_D = float(rng.choice(neighbour_config["spacing_values_D"]))
            nb_poly, _, _ = make_neighbour_polygon(target_cx, target_cy, neighbour_config["distance_km"],
                                                   neighbour_config["direction_deg"], neighbour_config["n_turbines"], spacing_D,
                                                   neighbour_config["diameter"])

            nb_spacing_m = spacing_D * neighbour_config["diameter"]

            x_nb, y_nb, _ = place_turbines(nb_poly, neighbour_config["n_turbines"], 
                                           nb_spacing_m, seed=(random_seed + mc_iter))

            if len(x_nb) != neighbour_config["n_turbines"]:
                continue

            x_nb = np.asarray(x_nb)
            y_nb = np.asarray(y_nb)
            n_nb = len(x_nb)
            x_all = np.concatenate([x_target, x_nb])
            y_all = np.concatenate([y_target, y_nb])
            D_all = np.concatenate([target_D, np.full(n_nb, neighbour_config["diameter"])])

            feature_rows = []

            for t_idx in range(n_target):
                other_mask = np.ones(len(x_all), dtype=bool)
                other_mask[t_idx] = False
                geo_feats = compute_geometric_features(
                        x_all[t_idx],
                        y_all[t_idx],
                        D_all[t_idx],
                        x_all[other_mask],
                        y_all[other_mask],
                        D_all[other_mask],
                        wind_config["wd"],
                        kw=KW_JENSEN
                    )

                feature_rows.append({"ws_free": wind_config["ws"], 
                                     "wd": wind_config["wd"],
                                     "ti": wind_config["ti"],
                                     "target_rotor_dia_m": target_config[ "diameter"],
                                     "target_hub_height_m": target_config["hub_height"],
                                     "target_rated_power_kw": target_config["rated_power"],
                                     "nb_distance_km": neighbour_config["distance_km"],
                                     "nb_direction_deg":  neighbour_config["direction_deg"],
                                     "nb_n_turbines": n_nb,
                                     "nb_spacing_D": spacing_D,
                                     "nb_rotor_dia_m": neighbour_config[ "diameter"],
                                     "nb_hub_height_m": neighbour_config["hub_height"],
                                     "nb_rated_power_kw": neighbour_config["rated_power"],
                                     "pi_n_blocking": geo_feats["pi_n_blocking"],
                                     "pi_blocking_ratio": geo_feats["pi_blocking_ratio"],
                                     "pi_blocking_distance": geo_feats["pi_blocking_distance"],
                                     "pi_streamwise_dist_1st": geo_feats["pi_streamwise_dist_1st"],
                                     "pi_lateral_dist_1st": geo_feats["pi_lateral_dist_1st"],
                                     "pi_rotor_dia_1st": geo_feats["pi_rotor_dia_1st"]})

            feature_df = pd.DataFrame(feature_rows)
            predicted_power_ratio = model.predict(feature_df[features])
            predicted_power_ratio = np.clip(predicted_power_ratio, 0, 1)
            predicted_power_kw = predicted_power_ratio * target_config["rated_power"]
            P_with_neighbour = float(predicted_power_kw.sum())
            power_loss_kw = P_without_neighbour - P_with_neighbour
            wake_loss_pct = power_loss_kw / P_without_neighbour * 100

            if np.isfinite(wake_loss_pct):
                wake_losses.append(wake_loss_pct)
                power_losses.append(power_loss_kw)

        if not wake_losses:
            raise ValueError("No valid Monte Carlo results.")

        wake_losses = np.asarray(wake_losses)
        power_losses = np.asarray(power_losses)

        return {"wake_loss": float(np.mean(wake_losses)),
                "wake_uncertainty": float(np.std(wake_losses) if len(wake_losses) > 1 else 0),
                "power_loss": float(np.mean(power_losses) / 1000),
                "power_uncertainty": float(np.std(power_losses) / 1000 if len(power_losses) > 1 else 0)}

    except Exception as e:
        logging.exception( f"Monte Carlo simulation failed: {e}")
        raise CustomException( e, sys)
