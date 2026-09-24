import argparse
import sys
import pandas as pd

from wake_uncertainty.config import DATA_DIR, DEFAULT_FARM_FILE, DEFAULT_PROJECTED_EPSG, FEATURES, NB_SPACINGS_D, RANDOM_SEED, TARGET
from wake_uncertainty.model import train_model, evaluate_model, save_model, load_model
from wake_uncertainty.uncertainty import run_monte_carlo
from wake_uncertainty.utils import parse_windfarmpoly, find_farm, project_polygon
from wake_uncertainty.exception import CustomException
from wake_uncertainty.logger import logging


def train(args):
    try:
        df = pd.read_csv(args.data)
        bundle = train_model(df=df, features=FEATURES, target=TARGET, test_size=args.test_size, random_seed=args.seed)
        model_path = save_model(bundle)
        metrics = bundle["metrics"]
        logging.info("XGBoost training complete")
        print(f"R2: {metrics['r2']}")
        print(f"RMSE: {metrics['rmse']}")
        print(f"MAE: {metrics['mae']}")

    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise CustomException(e, sys)


def evaluate(args):
    try:
        df = pd.read_csv(args.data)
        bundle = load_model()
        if isinstance(bundle, dict):
            model = bundle["model"]
            features = bundle.get("features", FEATURES)
            target = bundle.get("target", TARGET)

        else:
            model = bundle
            features = FEATURES
            target = TARGET

        clean = df.dropna(subset=[target])
        metrics = evaluate_model(model, clean[features], clean[target])
        print(f"R2: {metrics['r2']}")
        print(f"RMSE: {metrics['rmse']}")
        print(f"MAE:{metrics['mae']}")

    except Exception as e:
        logging.exception(f"Evaluation failed: {e}")
        raise CustomException(e, sys)


def monte_carlo(args):
    try:
        target_config = {"n_turbines": args.target_turbines,
                         "diameter": args.target_diameter,
                         "hub_height": args.target_hub_height,
                         "rated_power": args.target_rated_power,
                         "spacing_D": args.target_spacing}

        neighbour_config = {"distance_km": args.nb_distance,
                            "direction_deg": args.nb_direction,
                            "n_turbines": args.nb_turbines,
                            "spacing_values_D": args.nb_spacing_values,
                            "diameter": args.nb_diameter,
                            "hub_height": args.nb_hub_height,
                            "rated_power": args.nb_rated_power}

        wind_config = {"ws": args.wind_speed,
                       "wd": args.wind_direction,
                       "ti": args.ti}

        target_poly = None

        if args.target_farm is not None:
            farms = parse_windfarmpoly(args.farm_file, DATA_DIR)

            _, farm = find_farm(farms, args.target_farm)

            if farm is None:
                raise ValueError(f"Target farm not found: {args.target_farm}")

            target_poly = project_polygon(farm["polygon"], args.epsg)

        result = run_monte_carlo(target_config=target_config,
                                 neighbour_config=neighbour_config,
                                 wind_config=wind_config,
                                 target_poly=target_poly,
                                 random_seed=args.seed)

        print(f"Wake loss: {result['wake_loss']:.2f} ± {result['wake_uncertainty']:.2f} %")

        print(f"Power loss: {result['power_loss']:.2f} ± {result['power_uncertainty']:.2f} MW")

    except Exception as e:
        logging.exception(f"Monte Carlo simulation failed: {e}")
        raise CustomException(e, sys)


def create_parser():
    parser = argparse.ArgumentParser(description=("Offshore inter-farm wake uncertainty modelling"))
    subparsers = (parser.add_subparsers(dest="command", required=True))
    train_parser = (subparsers.add_parser("train"))
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    train_parser.set_defaults(func=train)
    evaluate_parser = (subparsers.add_parser("evaluate"))
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.set_defaults(func=evaluate)

    mc_parser = subparsers.add_parser("monte-carlo")

    mc_parser.add_argument("--target-turbines", type=int, required=True)
    mc_parser.add_argument("--target-diameter", type=float, required=True)
    mc_parser.add_argument("--target-hub-height", type=float, required=True)
    mc_parser.add_argument("--target-rated-power", type=float, required=True)
    mc_parser.add_argument("--target-spacing", type=float, required=True)
    mc_parser.add_argument("--nb-distance", type=float, required=True)
    mc_parser.add_argument("--nb-direction", type=float, required=True)
    mc_parser.add_argument("--nb-turbines", type=int, required=True)
    mc_parser.add_argument("--nb-spacing-values", type=float, nargs="+", default=NB_SPACINGS_D)
    mc_parser.add_argument("--nb-diameter", type=float, required=True)
    mc_parser.add_argument("--nb-hub-height", type=float, required=True)
    mc_parser.add_argument("--nb-rated-power", type=float, required=True)
    mc_parser.add_argument("--wind-speed", type=float, required=True)
    mc_parser.add_argument("--wind-direction", type=float, required=True)
    mc_parser.add_argument("--ti", type=float, required=True)
    mc_parser.add_argument("--target-farm", type=str, default=None)
    mc_parser.add_argument("--farm-file", type=str, default=DEFAULT_FARM_FILE)
    mc_parser.add_argument("--epsg", type=int, default=DEFAULT_PROJECTED_EPSG)
    mc_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    mc_parser.set_defaults(func=monte_carlo)
    return parser


def main():
    try:
        parser = create_parser()
        args = parser.parse_args()
        logging.info(f"Running command: {args.command}")
        args.func(args)

    except Exception as e:
        logging.exception(f"Application failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

