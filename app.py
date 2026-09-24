import os
import sys

import streamlit as st

from wake_uncertainty.config import FARM_CONFIGS, RANDOM_SEED
from wake_uncertainty.uncertainty import make_target_polygon, run_monte_carlo

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


st.set_page_config(
    page_title="Offshore Wake Uncertainty Explorer",
    page_icon="🌬️",
    layout="wide",
)

st.title("Offshore Wake Uncertainty Explorer")

st.write(
    """
    Estimate **inter-farm wake loss and layout uncertainty** using the trained
    XGBoost surrogate model.

    Choose either an **existing wind farm** from the project GeoPackage or
    define a **new target wind farm** from turbine specifications.
    """
)


def get_geopackage_path():
    candidates = [
        os.path.join(PROJECT_ROOT, "gdf.gpkg"),
        os.path.join(PROJECT_ROOT, "data", "gdf.gpkg"),
        os.path.join(PROJECT_ROOT, "data", "raw", "gdf.gpkg"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


@st.cache_data(show_spinner=False)
def load_geopackage(path):
    import geopandas as gpd
    return gpd.read_file(path)


def find_name_column(gdf):
    preferred_names = [
        "name",
        "farm_name",
        "windfarm",
        "wind_farm",
        "site",
        "project",
        "project_name",
    ]

    lower_map = {
        str(column).strip().lower(): column
        for column in gdf.columns
    }

    for candidate in preferred_names:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(
        "Could not identify the wind-farm name column. "
        f"Available columns: {list(gdf.columns)}"
    )


def get_farm_config(farm_name):
    for farm in FARM_CONFIGS:
        if farm["name"].strip().lower() == farm_name.strip().lower():
            return farm

    raise ValueError(
        f"No target turbine configuration was found for '{farm_name}'."
    )


def get_existing_farm_polygon(gdf, farm_name, target_epsg):
    name_col = find_name_column(gdf)

    mask = (
        gdf[name_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == farm_name.strip().lower()
    )

    selected = gdf.loc[mask].copy()

    if selected.empty:
        raise ValueError(
            f"'{farm_name}' was not found in the GeoPackage."
        )

    if selected.crs is None:
        raise ValueError(
            "The GeoPackage does not contain a coordinate reference system."
        )

    selected = selected.to_crs(epsg=int(target_epsg))

    try:
        polygon = selected.geometry.union_all()
    except AttributeError:
        polygon = selected.geometry.unary_union

    if polygon.is_empty:
        raise ValueError(
            f"The polygon for '{farm_name}' is empty."
        )

    return polygon


def make_target_config(
    n_turbines,
    diameter,
    hub_height,
    rated_power,
    spacing_D,
):
    return {
        "n_turbines": int(n_turbines),
        "diameter": float(diameter),
        "hub_height": float(hub_height),
        "rated_power": float(rated_power),
        "spacing_D": float(spacing_D),
    }


def make_neighbour_config(
    distance_km,
    direction_deg,
    n_turbines,
    spacing_values_D,
    diameter,
    hub_height,
    rated_power,
):
    return {
        "distance_km": float(distance_km),
        "direction_deg": float(direction_deg),
        "n_turbines": int(n_turbines),
        "spacing_values_D": [
            float(value)
            for value in spacing_values_D
        ],
        "diameter": float(diameter),
        "hub_height": float(hub_height),
        "rated_power": float(rated_power),
    }


def make_wind_config(ws, wd, ti):
    return {
        "ws": float(ws),
        "wd": float(wd),
        "ti": float(ti),
    }


def show_target_summary(target_config):
    columns = st.columns(5)

    columns[0].metric(
        "Target turbines",
        target_config["n_turbines"],
    )

    columns[1].metric(
        "Rotor diameter",
        f'{target_config["diameter"]:.0f} m',
    )

    columns[2].metric(
        "Hub height",
        f'{target_config["hub_height"]:.0f} m',
    )

    columns[3].metric(
        "Rated power",
        f'{target_config["rated_power"] / 1000:.2f} MW',
    )

    columns[4].metric(
        "Target spacing",
        f'{target_config["spacing_D"]:.1f}D',
    )


def show_results(result):
    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    col1.metric(
        "Inter-farm wake loss",
        f'{result["wake_loss"]:.2f} ± '
        f'{result["wake_uncertainty"]:.2f} %',
    )

    col2.metric(
        "Power loss",
        f'{result["power_loss"]:.2f} ± '
        f'{result["power_uncertainty"]:.2f} MW',
    )

    st.caption(
        "± is the standard deviation across Monte Carlo layout realisations. "
        "It represents layout-induced uncertainty, not a confidence interval "
        "and not total wake-model uncertainty."
    )


def neighbour_and_wind_inputs(prefix):
    st.markdown("#### Neighbouring farm")

    c1, c2, c3 = st.columns(3)

    nb_distance = c1.number_input(
        "Distance from target (km)",
        min_value=0.1,
        value=15.0,
        step=1.0,
        key=f"{prefix}_nb_distance",
    )

    nb_direction = c2.number_input(
        "Bearing from target (°)",
        min_value=0.0,
        max_value=359.9,
        value=270.0,
        step=5.0,
        key=f"{prefix}_nb_direction",
    )

    nb_turbines = c3.number_input(
        "Neighbour turbines",
        min_value=1,
        value=60,
        step=1,
        key=f"{prefix}_nb_turbines",
    )

    c1, c2, c3 = st.columns(3)

    nb_diameter = c1.number_input(
        "Neighbour rotor diameter (m)",
        min_value=1.0,
        value=178.0,
        step=1.0,
        key=f"{prefix}_nb_diameter",
    )

    nb_hub_height = c2.number_input(
        "Neighbour hub height (m)",
        min_value=1.0,
        value=119.0,
        step=1.0,
        key=f"{prefix}_nb_hub_height",
    )

    nb_rated_power = c3.number_input(
        "Neighbour rated power (kW)",
        min_value=1.0,
        value=10000.0,
        step=100.0,
        key=f"{prefix}_nb_rated_power",
    )

    spacing_values = st.multiselect(
        "Neighbour spacing values to sample (D)",
        options=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
        default=[3, 5, 7, 9],
        key=f"{prefix}_spacing_values",
    )

    st.markdown("#### Wind conditions")

    c1, c2, c3 = st.columns(3)

    ws = c1.number_input(
        "Wind speed (m/s)",
        min_value=0.1,
        value=10.0,
        step=0.5,
        key=f"{prefix}_ws",
    )

    wd = c2.number_input(
        "Wind direction (°)",
        min_value=0.0,
        max_value=359.9,
        value=270.0,
        step=5.0,
        key=f"{prefix}_wd",
    )

    ti = c3.number_input(
        "Turbulence intensity",
        min_value=0.001,
        max_value=0.5,
        value=0.06,
        step=0.01,
        format="%.3f",
        key=f"{prefix}_ti",
    )

    st.markdown("#### Monte Carlo")

    c1, c2 = st.columns(2)

    n_mc = c1.number_input(
        "Number of realisations",
        min_value=10,
        max_value=5000,
        value=500,
        step=10,
        key=f"{prefix}_n_mc",
    )

    seed = c2.number_input(
        "Random seed",
        min_value=0,
        value=int(RANDOM_SEED),
        step=1,
        key=f"{prefix}_seed",
    )

    return {
        "distance": nb_distance,
        "direction": nb_direction,
        "turbines": nb_turbines,
        "diameter": nb_diameter,
        "hub_height": nb_hub_height,
        "rated_power": nb_rated_power,
        "spacing_values": spacing_values,
        "ws": ws,
        "wd": wd,
        "ti": ti,
        "n_mc": n_mc,
        "seed": seed,
    }


st.divider()

target_mode = st.radio(
    "Choose target-farm input",
    [
        "Existing wind farm",
        "New wind farm",
    ],
    horizontal=True,
)


if target_mode == "Existing wind farm":

    gpkg_path = get_geopackage_path()

    if gpkg_path is None:
        st.error(
            "Could not find gdf.gpkg. Put it in the project root, "
            "`data/`, or `data/raw/`."
        )
        st.stop()

    try:
        farms_gdf = load_geopackage(
            gpkg_path
        )
    except Exception as exc:
        st.error(
            f"Could not load the GeoPackage: {exc}"
        )
        st.stop()

    configured_farms = [
        farm["name"]
        for farm in FARM_CONFIGS
    ]

    selected_farm = st.selectbox(
        "Select target wind farm",
        configured_farms,
    )

    try:
        farm_config = get_farm_config(
            selected_farm
        )

        target_config = make_target_config(
            n_turbines=farm_config["n_turbines"],
            diameter=farm_config["rotor_dia"],
            hub_height=farm_config["hub_height"],
            rated_power=farm_config["rated_power"],
            spacing_D=farm_config["spacing_D"],
        )

        target_epsg = int(
            farm_config["epsg"]
        )

        target_poly = get_existing_farm_polygon(
            farms_gdf,
            selected_farm,
            target_epsg,
        )

    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Target farm")

    st.write(
        f"Using the stored **{selected_farm}** boundary "
        f"from `{os.path.basename(gpkg_path)}`."
    )

    show_target_summary(
        target_config
    )

    with st.form(
        "existing_farm_form"
    ):

        values = neighbour_and_wind_inputs(
            "existing"
        )

        submitted = st.form_submit_button(
            "Run uncertainty analysis",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if not values["spacing_values"]:
            st.error(
                "Select at least one neighbour spacing value."
            )
            st.stop()

        neighbour_config = make_neighbour_config(
            distance_km=values["distance"],
            direction_deg=values["direction"],
            n_turbines=values["turbines"],
            spacing_values_D=values["spacing_values"],
            diameter=values["diameter"],
            hub_height=values["hub_height"],
            rated_power=values["rated_power"],
        )

        wind_config = make_wind_config(
            ws=values["ws"],
            wd=values["wd"],
            ti=values["ti"],
        )

        try:
            with st.spinner(
                "Running Monte Carlo uncertainty analysis..."
            ):
                result = run_monte_carlo(
                    target_config=target_config,
                    neighbour_config=neighbour_config,
                    wind_config=wind_config,
                    target_poly=target_poly,
                    n_mc=int(values["n_mc"]),
                    random_seed=int(values["seed"]),
                )

            show_results(
                result
            )

        except Exception as exc:
            st.error(
                f"Simulation failed: {exc}"
            )


else:

    st.subheader(
        "Define new target farm"
    )

    with st.form(
        "new_farm_form"
    ):

        st.markdown(
            "#### Target-farm specifications"
        )

        c1, c2, c3 = st.columns(3)

        target_turbines = c1.number_input(
            "Target turbines",
            min_value=1,
            value=44,
            step=1,
        )

        target_diameter = c2.number_input(
            "Target rotor diameter (m)",
            min_value=1.0,
            value=164.0,
            step=1.0,
        )

        target_hub_height = c3.number_input(
            "Target hub height (m)",
            min_value=1.0,
            value=105.0,
            step=1.0,
        )

        c1, c2 = st.columns(2)

        target_rated_power = c1.number_input(
            "Target rated power (kW)",
            min_value=1.0,
            value=8400.0,
            step=100.0,
        )

        target_spacing = c2.number_input(
            "Target spacing (D)",
            min_value=1.0,
            value=5.0,
            step=0.5,
        )

        values = neighbour_and_wind_inputs(
            "new"
        )

        submitted = st.form_submit_button(
            "Run uncertainty analysis",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if not values["spacing_values"]:
            st.error(
                "Select at least one neighbour spacing value."
            )
            st.stop()

        target_config = make_target_config(
            n_turbines=target_turbines,
            diameter=target_diameter,
            hub_height=target_hub_height,
            rated_power=target_rated_power,
            spacing_D=target_spacing,
        )

        neighbour_config = make_neighbour_config(
            distance_km=values["distance"],
            direction_deg=values["direction"],
            n_turbines=values["turbines"],
            spacing_values_D=values["spacing_values"],
            diameter=values["diameter"],
            hub_height=values["hub_height"],
            rated_power=values["rated_power"],
        )

        wind_config = make_wind_config(
            ws=values["ws"],
            wd=values["wd"],
            ti=values["ti"],
        )

        target_poly = make_target_polygon(
            n_turbines=target_config["n_turbines"],
            spacing_D=target_config["spacing_D"],
            diameter=target_config["diameter"],
        )

        try:
            with st.spinner(
                "Running Monte Carlo uncertainty analysis..."
            ):
                result = run_monte_carlo(
                    target_config=target_config,
                    neighbour_config=neighbour_config,
                    wind_config=wind_config,
                    target_poly=target_poly,
                    n_mc=int(values["n_mc"]),
                    random_seed=int(values["seed"]),
                )

            show_results(
                result
            )

        except Exception as exc:
            st.error(
                f"Simulation failed: {exc}"
            )