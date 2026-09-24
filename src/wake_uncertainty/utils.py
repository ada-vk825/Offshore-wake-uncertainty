# importing necessary libraries
import sys
import os
import numpy as np
import pandas as pd

from .config import KW_JENSEN, OUTPUT_COLUMNS, RANDOM_SEED
from .logger import logging
from .exception import CustomException


# Data Loading
def load_geodata(path):
    """
    Load geospatial data from a specified path and layer.

    Parameters:
    path : str 
        The file path to the geospatial data.

    Returns:
    gdf: 
        A GeoDataFrame containing the loaded geospatial data.
    """
    import geopandas as gpd
    logging.info(f"Loading geospatial data from {path}")
    # Check if the file exists
    if not os.path.exists(path):
        logging.error(f"The specified path does not exist: {path}")
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Load the geospatial data
    gdf = gpd.read_file(path)
    logging.info(f"Successfully loaded geospatial data with {len(gdf)} records.")
    return gdf


def load_csv(path, drop_unnamed=True):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    path : str 
        The file path to the CSV file.
    drop_unnamed :bool 
        Whether to drop unnamed columns. Default is True.

    Returns:
    df: A pandas DataFrame containing the loaded CSV data.
    """
    logging.info(f"Loading CSV data from {path}")
    # Check if the file exists
    if not os.path.exists(path):
        logging.error(f"The specified path does not exist: {path}")
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Load the CSV data
    df = pd.read_csv(path)
    
    # Optionally drop unnamed columns
    if drop_unnamed:
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        logging.info("Dropped unnamed columns from the DataFrame.")

    logging.info(f"Successfully loaded CSV data with {len(df)} records.")
    return df


def parse_windfarmpoly(filename, data_dir):
    """
    Parse wind farm polygon data from a CSV file.

    Parameters:
    filename :str
        The name of the CSV file containing wind farm polygon data.
    data_dir :str 
        The directory where the CSV file is located.

    Returns:
    df: A pandas DataFrame containing the parsed wind farm polygon data.
    """
    path = os.path.join(data_dir, filename)
    logging.info(f"Parsing wind farm polygon data from {path}")
    
    # Load the CSV data
    df = load_geodata(path)
    logging.info(f"Succesfully loaded wind farm polygon data with {len(df)} records.")
    farms = {}
    for _, row in df.iterrows():
        name = row.get("name")
        geometry = row.geometry
        if not isinstance(name, str) or not name.strip() or geometry is None or geometry.is_empty:
            continue
        centroid = geometry.centroid
        farms[name] = {
            "name": name,
            "polygon": geometry,
            "centroid_lat": float(centroid.y),
            "centroid_lon": float(centroid.x),
        }
    logging.info("Parsed %s named wind farms from %s", len(farms), path)
    return farms


def find_farm(all_farms, search_name):
    """
    Find a wind farm by name from a dictionary of all farms.

    Parameters:
    all_farms : dict
        A dictionary containing all wind farms.
    search_name : str
        The name of the wind farm to search for.

    Returns:
    dict or None: The wind farm data if found, otherwise None.
    """
    logging.info(f"Searching for wind farm: {search_name}")
    if not isinstance(search_name, str):
        logging.warning("Invalid search name provided.")
        return f"Invalid search name: {search_name}"
    query = search_name.strip().lower()
    for name, farm in all_farms.items():
        if name.strip().lower() == query:
            logging.info(f"Found wind farm: {name}")
            return name, farm
    for name, farm in all_farms.items():
        if query in name.strip().lower():
            logging.info(f"Found wind farm: {name}")
            return name, farm
    logging.warning(f"Wind farm not found: {search_name}")
    return f"Wind farm not found: {search_name}"


def require_farm(all_farms, search_name):
    """
    Require a wind farm by name from a dictionary of all farms.

    Parameters:
    all_farms : dict 
        A dictionary containing all wind farms.
    search_name : str
        The name of the wind farm to search for.

    Returns:
    dict: The wind farm data if found.

    Raises:
    CustomException: If the wind farm is not found.
    """
    name, farm = find_farm(all_farms, search_name)
    if isinstance(farm, str):
        logging.error(farm)
        raise CustomException(farm)
    logging.info(f"Successfully retrieved wind farm: {name}")
    return name, farm


def load_era5_wind_rose(data_dir, pattern = "era5_*.nc"):
    """
    Load ERA5 wind rose data from NetCDF files in a specified directory.

    Parameters:
    data_dir : str 
        The directory where the NetCDF files are located.
    pattern : str
        The pattern to match NetCDF files. Default is "era5_*.nc".

    Returns:
    df: A pandas DataFrame containing the loaded ERA5 wind rose data.
    """
    import xarray as xr
    from scipy.special import gamma
    from scipy.stats import weibull_min

    files = os.path.join(data_dir, pattern)
    logging.info(f"Loading ERA5 wind rose data from {files}")

    if not os.path.exists(data_dir):
        logging.error(f"The specified data directory does not exist: {data_dir}")
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")
    
    ds = xr.open_mfdataset([str(p) for p in files])
    logging.info(f"Successfully opened {len(ds)} NetCDF files.")
    try:
        u = ds['u100'].values.ravel()
        v = ds['v100'].values.ravel()
        logging.info("Extracted u and v wind components from the dataset.")
    finally:
        ds.close()
        logging.info("Closed the dataset after extraction.")

    valid = np.isfinite(u) & np.isfinite(v)
    logging.info(f"Filtered valid wind data points: {np.sum(valid)} out of {len(u)}")
    u, v = u[valid], v[valid]
    logging.info(f"Valid wind data points after filtering: {len(u)}")
    ws = np.sqrt(u**2 + v**2)
    wd = (270 - np.degrees(np.arctan2(v, u))) % 360
    logging.info("Calculated wind speed and direction.")

    rows = []
    for sector in np.arange(0, 360, 30):
        lower, upper = (sector -15) % 360, (sector + 15) % 360
        mask = (wd >= lower) | (wd < upper) if lower > upper else (wd >= lower) & (wd < upper)
        ws_sector = ws[mask]
        ws_sector = ws_sector[ws_sector > 0.5]
        logging.info(f"Sector {sector}°: {len(ws_sector)} valid wind speed data points after filtering.")
        if len(ws_sector) < 3:
            logging.warning(f"Not enough data for sector {sector}° to fit Weibull distribution.")
            continue
        k, _, a = weibull_min.fit(ws_sector, floc=0)
        logging.info(f"Fitted Weibull distribution for sector {sector}°: shape={k}, scale={a}")
        rows.append({"direction_deg": float(sector),
                     "frequency": float(mask.sum() / len(ws)),
                     "weibull_A": float(a),
                    "weibull_k": float(k),
                    "mean_ws": float(a * gamma(1 + 1/k))})
    df = pd.DataFrame(rows)
    logging.info(f"Successfully created wind rose DataFrame with {len(df)} sectors.")
    df["frequency"] /= df["frequency"].sum()
    return df


def download_era5_for_farm(farm_name, lat, lon, era5_dir, years= (2023, 2024, 2025)):
    """
    Download ERA5 wind data for a specific wind farm location.
    """
    import cdsapi
    farm_era5_dir = os.path.join(era5_dir, farm_name.lower().replace(' ','_'))
    os.makedirs(farm_era5_dir, exist_ok=True)
    logging.info(f"Downloading ERA5 data for {farm_name} at ({lat}, {lon}) into {farm_era5_dir}")
    
    files = []
    client = cdsapi.Client(quiet=True)

    for year in years:
        for month in range(1, 13):
            logging.info(f"Downloading {year:02d} - {month:02d} data")
            file_path = os.path.join(farm_era5_dir, f'era5_{farm_name}_100m_{year}_{month}.nc')
            if os.path.exists(file_path):
                logging.info(f"{year} - {month} already downloaded!")
            else:
                client.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind'],
                        'year': f'{year:02d}',
                        'month': f'{month:02d}',
                        'day': [f'{d:02d}' for d in range(1, 32)],
                        'time': [f'{h:02d}:00' for h in range(24)],
                        'area': [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5],
                        'data_format': 'netcdf',
                    },
                    file_path
                )
        files.append(file_path)
    logging.info(f"Downloaded ERA5 data for {farm_name} into {farm_era5_dir}")
    return files


# Geometry and turbine placement functions
def automatic_utm_epsg(lon, lat):
    """
    Automatically determine the UTM EPSG code based on longitude and latitude.

    Parameters:
    lon : float
        Longitude of the location.
    lat : float
        Latitude of the location.

    Returns:
    int: The EPSG code for the UTM zone.
    """
    zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700
    logging.info(f"Calculated UTM EPSG code {epsg_code} for coordinates ({lon}, {lat})")
    return epsg_code


def validate_polygon(polygon):
    """
    Validate a polygon geometry.
    Parameters:
    polygon: A shapely Polygon object to validate.
    Returns:
    polygon: The validated polygon if valid.
    Raises:
    ValueError: If the polygon is None, empty, has non-positive area, or is not valid.
    """
    if polygon is None or polygon.is_empty:
        logging.error("Invalid polygon: Polygon is None or empty.")
        raise ValueError("Invalid polygon: Polygon is None or empty.")

    if polygon.area <= 0:
        logging.error("Invalid polygon: Polygon area is non-positive.")
        raise ValueError("Invalid polygon: Polygon area is non-positive.")

    if not polygon.is_valid:
        logging.error("Invalid polygon: Polygon geometry is not valid.")
        raise ValueError("Invalid polygon: Polygon geometry is not valid.")
    return polygon


def project_polygon(polygon, target_epsg, source_epsg=4326):
    """
    Project a polygon from one coordinate reference system to another.

    If the target EPSG code is not provided, the appropriate UTM zone
    is automatically calculated from the polygon centroid.

    Parameters:
    polygon: A shapely Polygon geometry to project.
    target_epsg (int or None): EPSG code of the target coordinate
        reference system. If None, a UTM EPSG code is calculated
        automatically.
    source_epsg (int): EPSG code of the source coordinate reference
        system. Default is 4326.

    Returns:
    geom: The projected and validated polygon geometry.
    """
    import geopandas as gpd
    from pyproj import CRS

    validated_polygon = validate_polygon(polygon)
    logging.info(f"Projecting polygon to EPSG:{target_epsg}")
    if target_epsg is None:
        centroid = polygon.centroid
        target_epsg = automatic_utm_epsg(centroid.x, centroid.y)
    CRS.from_epsg(target_epsg)  # Validate EPSG code
    gdf = gpd.GeoDataFrame(geometry=[validated_polygon], crs=f"EPSG:{source_epsg}")
    geom = gdf.to_crs(f"EPSG:{target_epsg}").geometry.iloc[0]
    logging.info(f"Successfully projected polygon to EPSG:{target_epsg}")
    geom = validate_polygon(geom)
    return geom


def make_neighbour_polygon(centroid_x, centroid_y, distance_km, direction_deg,
                           n_turbines, spacing_D, rotor_dia):
    """
    Create a square polygon representing a neighbouring wind farm.

    The neighbouring farm centroid is positioned at a specified distance
    and direction from the target farm centroid. The polygon size is
    calculated from the number of turbines, rotor diameter, and turbine
    spacing.

    Parameters:
    centroid_x : float
        X-coordinate of the target farm centroid in metres.
    centroid_y : float
        Y-coordinate of the target farm centroid in metres.
    distance_km : float
        Distance between the target and neighbouring farm
        centroids in kilometres.
    direction_deg : float
        Bearing of the neighbouring farm from the target
        farm in degrees.
    n_turbines : int
        Number of turbines in the neighbouring farm.
    spacing_D : float
        Turbine spacing expressed in rotor diameters.
    rotor_dia : float
        Rotor diameter of the neighbouring turbines in metres.

    Returns:
    tuple: A tuple containing the neighbouring farm polygon, centroid
        x-coordinate, and centroid y-coordinate.
    """
    try:
        from shapely.geometry import Polygon

        if n_turbines <= 0 or spacing_D <= 0 or rotor_dia <= 0:
            raise ValueError("Number of turbines, spacing, and rotor diameter must be positive.")

        turbines_per_side = int(np.ceil(np.sqrt(n_turbines)))

        spacing_m = spacing_D * rotor_dia
        side_length = (turbines_per_side + 1) * spacing_m
        distance_m = distance_km * 1000
        direction_rad = np.radians(direction_deg)
        nb_x = (centroid_x + distance_m * np.sin(direction_rad))
        nb_y = (centroid_y + distance_m * np.cos(direction_rad))

        poly = Polygon([(nb_x - side_length / 2, nb_y - side_length / 2),
                        (nb_x + side_length / 2, nb_y - side_length / 2),
                        (nb_x + side_length / 2, nb_y + side_length / 2),
                        (nb_x - side_length / 2, nb_y + side_length / 2)])

        return poly, float(nb_x), float(nb_y)

    except Exception as e:
        logging.exception(f"Failed to create neighbour polygon: {e}")
        raise CustomException(e, sys)


def place_turbines(polygon, n_turbines, min_spacing_m, seed=42, max_expand_attempts=3):
    """
    Randomly place turbines inside a polygon while maintaining minimum spacing.

    Turbine locations are randomly sampled inside the polygon.
    A location is accepted only when it satisfies the minimum distance
    from all previously placed turbines. If all turbines cannot be placed,
    the polygon is expanded and placement is attempted again.

    Parameters:
    polygon: Polygon
        A shapely Polygon defining the wind farm boundary.
    n_turbines : int 
        Number of turbines to place.
    min_spacing_m : float
        Minimum allowed turbine spacing in metres.
    seed : int
        Random seed used for reproducible turbine placement.
        Default is 42.
    max_expand_attempts : int
        Maximum number of times the polygon may be
        expanded if all turbines cannot be placed. Default is 3.

    Returns:
    tuple: A tuple containing an array of x-coordinates, an array of
        y-coordinates, and the final polygon used for turbine placement.
    """
    try:
        from shapely.affinity import scale
        from shapely.geometry import Point

        current_poly = polygon
        rng = np.random.default_rng(seed)

        for expand_attempt in range(max_expand_attempts + 1):
            min_x, min_y, max_x, max_y = current_poly.bounds
            x_placed = []
            y_placed = []
            attempts = 0
            max_attempts = n_turbines * 200

            while (len(x_placed) < n_turbines and attempts < max_attempts):
                x = rng.uniform(min_x, max_x)
                y = rng.uniform(min_y, max_y)
                attempts += 1

                if not current_poly.contains(Point(x, y)):
                    continue

                too_close = False

                for xi, yi in zip(x_placed, y_placed):
                    distance_sq = ((x - xi) ** 2 + (y - yi) ** 2)
                    if (distance_sq < min_spacing_m ** 2):
                        too_close = True
                        break

                if not too_close:
                    x_placed.append(x)
                    y_placed.append(y)

            if len(x_placed) == n_turbines:
                return np.asarray(x_placed), np.asarray(y_placed), current_poly

            if (expand_attempt < max_expand_attempts):
                current_poly = scale(current_poly, xfact=1.25, yfact=1.25, origin="center")

        logging.warning(f"Placed only {len(x_placed)} / {n_turbines} turbines.")
        return np.asarray(x_placed), np.asarray(y_placed), current_poly
    
    except Exception as e:
        logging.exception(f"Failed to place turbines: {e}")
        raise CustomException(e, sys)


def compute_blocking_features(blocking_mask, streamwise_dist,lateral_dist,
                              neighbor_rotor_dia, target_rotor_dia, prefix):
    """
    Calculate wake-blocking features for a target turbine.

    The function uses a blocking mask to identify upstream turbines that
    may influence the target turbine and calculates blocking count,
    blocking ratio, average blocking distance, and first-blocker
    geometric properties.

    Parameters:
    blocking_mask : ndarray
        Boolean array identifying turbines classified as blockers.
    streamwise_dist : ndarray
        Streamwise distances between the
        surrounding turbines and the target turbine.
    lateral_dist : ndarray
        Lateral distances between the surrounding
        turbines and the target turbine.
    neighbor_rotor_dia : ndarray
        Rotor diameters of the surrounding
        turbines in metres.
    target_rotor_dia : float
        Rotor diameter of the target turbine in metres.
    prefix : str
        Prefix used for the returned feature names, such as
        'geom' or 'pi'.

    Returns:
    dict: Dictionary containing the number of blockers, blocking ratio,
        blocking distance, first-blocker streamwise distance, first-blocker
        lateral distance, and first-blocker rotor diameter.
    """

    n_blocking = int(blocking_mask.sum())
    logging.debug(f"Computing blocking features: {n_blocking} blocking turbines found.")
    if n_blocking == 0:
        return {
            f'{prefix}_n_blocking': 0,
            f'{prefix}_blocking_ratio': 0.0,
            f'{prefix}_blocking_distance': 0.0,
            f'{prefix}_streamwise_dist_1st': 0.0,
            f'{prefix}_lateral_dist_1st': 0.0,
            f'{prefix}_rotor_dia_1st': 0.0,
        }

    blocking_streamwise = streamwise_dist[blocking_mask]
    blocking_lateral = lateral_dist[blocking_mask]
    blocking_rotor_dia = neighbor_rotor_dia[blocking_mask]
    logging.debug(f"Blocking streamwise distances: {blocking_streamwise}")
    logging.debug(f"Blocking lateral distances: {blocking_lateral}")
    logging.debug(f"Blocking rotor diameters: {blocking_rotor_dia}")
    # blocking ratio
    overlap_per_turbine = np.clip((blocking_rotor_dia/2 + target_rotor_dia/2 - blocking_lateral) / target_rotor_dia, 0, 1)
    blocking_ratio = float(np.clip(overlap_per_turbine.sum(), 0, 1))
    logging.debug(f"Computed blocking ratio: {blocking_ratio}")
    # Blocking distance: mean streamwise distance of blockers
    blocking_distance = float(blocking_streamwise.mean())
    idx_closest = np.argmin(blocking_streamwise)

    return {
        f'{prefix}_n_blocking': n_blocking,
        f'{prefix}_blocking_ratio': blocking_ratio,
        f'{prefix}_blocking_distance': blocking_distance,
        f'{prefix}_streamwise_dist_1st': float(blocking_streamwise[idx_closest]),
        f'{prefix}_lateral_dist_1st': float(blocking_lateral[idx_closest]),
        f'{prefix}_rotor_dia_1st': float(blocking_rotor_dia[idx_closest]),
    }


def compute_geometric_features(target_x, target_y, target_rotor_dia, 
                               neighbour_x,neighbour_y, neighbour_rotor_dia,
                               wind_direction_deg, kw=KW_JENSEN):
    """
    Calculate geometric and physics-informed wake features for a target turbine.

    Turbine positions are transformed into wind-aligned streamwise and
    lateral distances. Two blocking definitions are then used: a simple
    geometric blocking region and a physics-informed expanding wake region.
    Blocking features are calculated for both definitions.

    Parameters:
    target_x : float
        X-coordinate of the target turbine in metres.
    target_y : float
        Y-coordinate of the target turbine in metres.
    target_rotor_dia : float
        Rotor diameter of the target turbine in metres.
    neighbour_x : ndarray
        X-coordinates of surrounding turbines.
    neighbour_y : ndarray
        Y-coordinates of surrounding turbines.
    neighbour_rotor_dia : ndarray)
        Rotor diameters of surrounding turbines in metres.
    wind_direction_deg : float
        Meteorological wind direction in degrees.
    kw : float 
        Wake expansion coefficient used for the physics-informed
        blocking calculation. Default is KW_JENSEN.

    Returns:
    dict: Dictionary containing geometric and physics-informed blocking
        features for the target turbine.
    """
    wind_going_rad = np.radians(wind_direction_deg + 180)
    logging.debug(f"Computing geometric features with wind direction {wind_direction_deg}° (going {wind_going_rad} rad)")
    ux = np.sin(wind_going_rad)
    uy = np.cos(wind_going_rad)
    logging.debug(f"Wind unit vector: ux={ux}, uy={uy}")

    dx = target_x - neighbour_x
    dy = target_y - neighbour_y
    logging.debug(f"Computed relative positions: dx={dx}, dy={dy}")

    streamwise_dst = dx * ux + dy *uy
    lateral_dst = np.abs(dx * (-uy) + (dy * ux))
    logging.debug(f"Computed streamwise distances: {streamwise_dst}")
    logging.debug(f"Computed lateral distances: {lateral_dst}")

    target_radius = target_rotor_dia / 2
    upstream_mask = streamwise_dst > 0

    blocking_radius_geom = neighbour_rotor_dia / 2 + target_rotor_dia
    blocking_mask_geom = upstream_mask & (lateral_dst < blocking_radius_geom)
    logging.debug(f"Blocking mask (geometric): {blocking_mask_geom}")
    geom_features = compute_blocking_features(blocking_mask_geom, streamwise_dst, lateral_dst, neighbour_rotor_dia, target_rotor_dia, prefix='geom')

    # Physics-informed features

    Dw = np.where(upstream_mask, neighbour_rotor_dia + 2 * kw * streamwise_dst, neighbour_rotor_dia)
    blocking_radius_pi = Dw / 2 + target_radius
    blocking_mask_pi = upstream_mask & (lateral_dst < blocking_radius_pi)
    logging.debug(f"Blocking mask (physics-informed): {blocking_mask_pi}")

    pi_feats = compute_blocking_features(blocking_mask_pi, streamwise_dst, lateral_dst, neighbour_rotor_dia, target_rotor_dia, prefix='pi')
    logging.debug(f"Computed geometric features: {geom_features}")
    logging.debug(f"Computed physics-informed features: {pi_feats}")

    features = {}
    logging.info(f"Computed features for target turbine at ({target_x}, {target_y}) with rotor diameter {target_rotor_dia}")
    features.update(geom_features)
    features.update(pi_feats)
    logging.info(f"Final computed features: {features}")
    return features


def sample_scenarios(nb_distance_km, nb_direction_deg, nb_size, nb_spacing_D,
                     ti_values, neighbour_specs, n_samples, ws_range=(3.0, 25.0),
                     wd_range=(0, 360), random_seed=RANDOM_SEED):
    """
    Randomly sample neighbouring wind-farm and atmospheric scenarios.

    Each scenario contains neighbouring farm distance, direction, turbine
    count, turbine spacing, turbine specification, turbulence intensity,
    wind speed, and wind direction.

    Parameters:
    nb_distance_km : sequence
        Minimum and maximum neighbouring farm
        distances in kilometres.
    nb_direction_deg : sequence
        Possible neighbouring farm directions
        in degrees.
    nb_size : sequence
        Range of neighbouring turbine counts.
    nb_spacing_D : sequence
        Range of neighbouring turbine spacings in
        rotor diameters.
    ti_values : sequence
        Possible turbulence intensity values.
    neighbour_specs : list
        List of dictionaries containing neighbouring
        turbine specifications.
    n_samples : int
        Number of scenarios to generate.
    ws_range : tuple
        Minimum and maximum free-stream wind speeds in m/s.
        Default is (3.0, 25.0).
    wd_range : sequence
        Possible wind directions in degrees.
        Default is (0, 360).
    random_seed : int
        Random seed used for reproducible scenario sampling.
        Default is RANDOM_SEED.

    Returns:
    scenarios : list
        A list of dictionaries containing the sampled
        scenario parameters.

    """
    rng = np.random.default_rng(random_seed)
    logging.info(f"Sampling {n_samples} scenarios with random seed {random_seed}")
    if n_samples <= 0:
        logging.error("Number of samples must be positive.")
        raise ValueError("Number of samples must be positive.")
    scenarios = []

    for i in range(n_samples):
        nb_spec = neighbour_specs[rng.integers(len(neighbour_specs))]
        scenario = {
            'distance_km': rng.uniform(nb_distance_km[0], nb_distance_km[-1]),
            'direction_deg': rng.choice(nb_direction_deg),
            'n_nb_turbines': int(rng.uniform(nb_size[0], nb_size[-1])),
            'nb_spacing_D': int(rng.uniform(nb_spacing_D[0], nb_spacing_D[-1])),
            'ti': rng.uniform(0.04, 0.1),
            'nb_spec': nb_spec,
            'nb_rotor_dia_m': nb_spec['diameter'],
            'nb_hub_height_m': nb_spec['hub_height'],
            'nb_rated_power_kw': nb_spec['rated_power'],
            'ws': float(rng.uniform(ws_range[0], ws_range[1])),
            'wd': rng.choice(wd_range)        
        }
        scenarios.append(scenario)
    logging.info(f"Successfully sampled {len(scenarios)} scenarios.")
    return scenarios


def create_turbine(spec):
    """
    Create a PyWake GenericWindTurbine from a turbine specification.

    Parameters:
    spec : dict
        Dictionary containing turbine properties. Expected keys
        include name, diameter, hub_height, and rated_power. An optional
        turbulence intensity value may also be provided using the 'ti' key.

    Returns:
    GenericWindTurbine: A PyWake GenericWindTurbine object created from
        the supplied turbine specification.
    """
    try:
        from py_wake.wind_turbines import GenericWindTurbine
    except ImportError as e:
        logging.error("py_wake is not installed. Please install it to use this function.")
        return CustomException("py_wake is not installed. Please install it to use this function.")
    return GenericWindTurbine(
        name=spec['name'],
        diameter=spec['diameter'],
        hub_height=spec['hub_height'],
        power_norm=spec['rated_power'] * 1000,
        turbulence_intensity=spec.get('ti', 0.06)
    )


def run_pywake_simulation(wind_rose_df, x_target, y_target, target_spec,
                        x_nb, y_nb, nb_spec, ws, wd, ti):
    """
    Run a PyWake simulation for target and neighbouring wind farm turbines.

    The function creates target and neighbouring turbine types, combines
    their coordinates, creates a PyWake site using the supplied wind rose,
    and runs the Nygaard_2022 wake model for the specified wind speed,
    wind direction, and turbulence intensity.

    Parameters:
    wind_rose_df : DataFrame)
        Wind rose containing direction,
        frequency, Weibull A, and Weibull k values.
    x_target : array
        X-coordinates of the target farm turbines.
    y_target : array
        Y-coordinates of the target farm turbines.
    target_spec : dict
        Target turbine specification containing name,
        diameter, hub height, and rated power.
    x_nb : array
        X-coordinates of neighbouring farm turbines.
    y_nb : array 
        Y-coordinates of neighbouring farm turbines.
    nb_spec : dict
        Neighbouring turbine specification containing name,
        diameter, hub height, and rated power.
    ws : float
        Free-stream wind speed in m/s.
    wd : float 
        Wind direction in degrees.
    ti :float
        Turbulence intensity.

    Returns:
    dict: Dictionary containing the PyWake simulation result, turbine
        power in kW, effective wind speed, combined turbine coordinates,
        turbine types, wind speed, wind direction, and number of target
        turbines.
    """
    try:
        from py_wake import Nygaard_2022
        from py_wake.site import XRSite
        from py_wake.wind_turbines import WindTurbines
    except ImportError as e:
        logging.error("py_wake is not installed. Please install it to use this function.")
        return CustomException("py_wake is not installed. Please install it to use this function.")

    logging.info(f"Running py_wake simulation for target turbine at ({x_target}, {y_target}) with wind speed {ws} m/s and direction {wd}°")
    logging.info(f"Neighbour turbines at positions: {list(zip(x_nb, y_nb))} with specs: {nb_spec}")
    target_wt = create_turbine(target_spec)
    nb_wt = create_turbine(nb_spec)
    turbines = WindTurbines.from_WindTurbine_lst([target_wt, nb_wt])
    logging.info(f"Created wind turbines: {turbines}")

    x_target = np.asarray(x_target)
    y_target = np.asarray(y_target)
    x_nb = np.asarray(x_nb)
    y_nb = np.asarray(y_nb)
    n_target = len(x_target)
    types_all = np.asarray([0] * n_target + [1] * len(x_nb), dtype=int)
    x_all = np.concatenate([x_target, x_nb])
    y_all = np.concatenate([y_target, y_nb])
    logging.info(f"Total turbines in simulation: {len(x_all)}")
    logging.info(f"Turbine types: {types_all}")

    ds = xr.Dataset(data_vars={"Sector_frequency": ("wd", wind_rose_df["frequency"].to_numpy()),
                               "Weibull_A": ("wd", wind_rose_df["weibull_A"].to_numpy()),
                               "Weibull_k": ("wd", wind_rose_df["weibull_k"].to_numpy()),
                               "TI": ti},
                    coords={"wd": wind_rose_df["direction_deg"].to_numpy()})
    site = XRSite(ds)
    logging.info("Created XRSite for simulation.")
    wake_model = Nygaard_2022(site, turbines)
    logging.info("Initialized Nygaard_2022 wake model.")
    ws_values = np.array([ws])
    wd_values = np.array([wd])
    sim_res = wake_model(x_all, y_all, ws=ws_values, wd=wd_values, ti=ti)
    logging.info("Simulation completed.")
    return {
        "simulation": sim,
        "power_kw": np.asarray(sim.Power.values, dtype=float) / 1000,
        "ws_eff": np.asarray(sim.WS_eff.values, dtype=float),
        "x_all": x_all,
        "y_all": y_all,
        "types_all": types_all,
        "ws": ws_values,
        "wd": wd_values,
        "n_target": n_target,
    }


def generate_training_data(wind_rose_df, target_poly, target_centroid, target_spec,
                           scenarios, random_seed=RANDOM_SEED, output_path=None):
    """
    Generate turbine-level training data from sampled wind-farm scenarios.

    The target turbine layout is generated inside the target polygon.
    For each scenario, a neighbouring farm polygon and turbine layout are
    created, PyWake is used to simulate turbine power, and geometric and
    physics-informed wake features are calculated for every target turbine.
    The resulting features and simulation outputs are stored as rows in a
    training DataFrame.

    Parameters:
    wind_rose_df : DataFrame
        Wind rose data used to define the PyWake site.
    target_poly: Shapely polygon defining the target wind-farm boundary.
    target_centroid: Shapely Point representing the target farm centroid.
    target_spec : dict
        Dictionary containing target turbine specifications,
        including turbine count, diameter, hub height, rated power, and
        spacing.
    scenarios : list
        List of scenario dictionaries generated by
        sample_scenarios().
    random_seed : int 
        Random seed used for reproducible target and
        neighbouring turbine placement. Default is RANDOM_SEED.
    output_path : str or None 
        Optional CSV file path used to save the
        generated training dataset. Default is None.

    Returns:
    df: A pandas DataFrame containing turbine-level scenario information,
        atmospheric variables, turbine specifications, wake topology
        features, simulated power, power ratio, and effective wind speed.
    """
    logging.info(f"Generating training data for {len(scenarios)} scenarios with random seed {random_seed}")
    rng = np.random.default_rng(random_seed)
    target_spacing_m = float(target_spec.get("spacing_D", 5.0) * target_spec["diameter"])
    n_target_requested = int(target_spec.get("n_turbines", 1))  
    x_target, y_target, _ = place_turbines(target_poly, n_target_requested, target_spacing_m, seed=random_seed)
    logging.info(f"Placed {len(x_target)} target turbines within the target polygon.")

    rows = []
    for i, scenario in enumerate(scenarios):
        logging.info(f"Processing scenario {i+1}/{len(scenarios)}: {scenario}")
        if i % 100 == 0:
            logging.info(f"Processed {i} scenarios so far.")
        spec = scenario['nb_spec']
        logging.info(f"Using neighbour turbine spec: {spec} to create neighbour polygon.")
        nb_poly, _, _ = make_neighbour_polygon(
            target_centroid.x,
            target_centroid.y,
            scenario['distance_km'],
            scenario['direction_deg'],
            scenario['n_nb_turbines'],
            scenario['nb_spacing_D'],
            spec['diameter']
        )
        logging.info(f"Created neighbour polygon for scenario {i+1}.")

        spacing_m = float(scenario['nb_spacing_D'] * spec['diameter'])
        x_nb, y_nb, _ = place_turbines(nb_poly, scenario['n_nb_turbines'], spacing_m, seed=random_seed+i)
        logging.info(f"Placed {len(x_nb)} neighbour turbines.")

        result = run_pywake_simulation(
            wind_rose_df=wind_rose_df,
            x_target=x_target,
            y_target=y_target,
            target_spec=target_spec,
            x_nb=x_nb,
            y_nb=y_nb,
            nb_spec=spec,
            ws=scenario['ws'],
            wd=scenario['wd'],
            ti=scenario['ti']
        )
        logging.info(f"Simulation results for scenario {i+1}: {result}")

        power = result["power_kw"]
        ws_eff = result["ws_eff"]
        x_all = result["x_all"]
        y_all = result["y_all"]

        for turbine_idx in range(result["n_target"]):
            mask = np.ones(len(x_all), dtype=bool)
            mask[turbine_idx] = False
            logging.debug(f"Computing geometric features for target turbine {turbine_idx+1}/{result['n_target']}.")
            for wd_idx, wd_value in enumerate(result["wd"]):
                features = compute_geometric_features(
                    target_x=x_all[turbine_idx],
                    target_y=y_all[turbine_idx],
                    target_rotor_dia=target_spec['diameter'],
                    neighbour_x=x_all[mask],
                    neighbour_y=y_all[mask],
                    neighbour_rotor_dia=np.array([target_spec['diameter']] * len(x_all[mask])),
                    wind_direction_deg=wd_value
                )

                logging.debug(f"Computed features for target turbine {turbine_idx+1}, wind direction {wd_value}°: {features}")

                for ws_idx, ws_value in enumerate(result["ws"]):
                    p_kw = float(power[turbine_idx, wd_idx, ws_idx])
                    eff = float(ws_eff[turbine_idx, wd_idx, ws_idx])
                    row: dict[str, Any] = {
                        "scenario_id": i,
                        "turbine_idx": turbine_idx,
                        "ws_free": float(ws_value),
                        "wd": float(wd_value),
                        "ti": float(scenario["ti"]),
                        "target_rotor_dia_m": float(target_spec["diameter"]),
                        "target_hub_height_m": float(target_spec["hub_height"]),
                        "target_rated_power_kw": float(target_spec["rated_power"]),
                        "nb_distance_km": float(scenario["distance_km"]),
                        "nb_direction_deg": float(scenario["direction_deg"]),
                        "nb_n_turbines": int(len(x_nb)),
                        "nb_spacing_D": float(scenario["nb_spacing_D"]),
                        "nb_rotor_dia_m": float(spec["diameter"]),
                        "nb_hub_height_m": float(spec["hub_height"]),
                        "nb_rated_power_kw": float(spec["rated_power"]),
                        "power_kw": p_kw,
                        "power_ratio": p_kw / float(target_spec["rated_power"]),
                        "ws_eff": eff,
                    }
                    logging.debug(f"Row data for scenario {i+1}, turbine {turbine_idx+1}, wind direction {wd_value}°, wind speed {ws_value} m/s: {row}")
                    row.update(features)
                    rows.append(row)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logging.info(f"Generated training data DataFrame with {len(df)} rows.")
    if output_path:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved training data to {output_path}")
    return df