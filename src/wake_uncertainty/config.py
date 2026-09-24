import os
import numpy as np


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ERA5_DIR = os.path.join(DATA_DIR, "era5")

OUTPUT_DIR = os.path.join(DATA_DIR, "train")

MODEL_DIR = os.path.join(PACKAGE_DIR, "models")

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

DEFAULT_FARM_FILE = os.path.join(DATA_DIR, "gdf.gpkg")


RANDOM_SEED = 42

N_SAMPLE_SCENARIOS = 50

N_MC = 50

KW_JENSEN = 0.05

SOURCE_EPSG = 4326

DEFAULT_PROJECTED_EPSG = 32631


TARGET_N_TURBINES = 44
TARGET_ROTOR_DIA_M = 164
TARGET_HUB_HEIGHT_M = 105
TARGET_RATED_POWER = 8400
TARGET_SPACING_D = 5


NB_DISTANCES_KM = [5, 50]

NB_DIRECTION_DEG = np.arange(0, 360, 30)

NB_SIZES = [20, 40, 80, 120, 160, 200]

NB_SPACINGS_D = [3, 5, 7, 9, 15]

TI_VALUES = [0.04, 0.06, 0.08, 0.10]

WS_SIM = (3.0, 25.0)

WD_SIM = (0.0, 360.0)


NEIGHBOUR_SPECS = [
    {
        "name": "V112-3.45",
        "diameter": 112,
        "hub_height": 84,
        "rated_power": 3450,
    }
]


FARM_CONFIGS = [
    {
        "name": "Dudgeon",
        "n_turbines": 67,
        "rotor_dia": 154,
        "hub_height": 95,
        "rated_power": 6000,
        "spacing_D": 7,
        "epsg": 32631,
    },
    {
        "name": "London Array",
        "n_turbines": 175,
        "rotor_dia": 107,
        "hub_height": 87,
        "rated_power": 3600,
        "spacing_D": 7,
        "epsg": 32631,
    },
    {
        "name": "Norther",
        "n_turbines": 44,
        "rotor_dia": 164,
        "hub_height": 105,
        "rated_power": 8400,
        "spacing_D": 7,
        "epsg": 32631,
    },
    {
        "name": "Kriegers Flak",
        "n_turbines": 72,
        "rotor_dia": 167,
        "hub_height": 100,
        "rated_power": 8400,
        "spacing_D": 7,
        "epsg": 32632,
    },
    {
        "name": "Rampion",
        "n_turbines": 116,
        "rotor_dia": 112,
        "hub_height": 84,
        "rated_power": 3450,
        "spacing_D": 7,
        "epsg": 32630,
    },
]


FEATURES = [
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


TARGET = "power_ratio"


OUTPUT_COLUMNS = [
    "scenario_id",
    "turbine_idx",
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

    "geom_n_blocking",
    "geom_blocking_ratio",
    "geom_blocking_distance",
    "geom_streamwise_dist_1st",
    "geom_lateral_dist_1st",
    "geom_rotor_dia_1st",

    "pi_n_blocking",
    "pi_blocking_ratio",
    "pi_blocking_distance",
    "pi_streamwise_dist_1st",
    "pi_lateral_dist_1st",
    "pi_rotor_dia_1st",

    "farm_name",
    "power_kw",
    "power_ratio",
    "ws_eff",
]