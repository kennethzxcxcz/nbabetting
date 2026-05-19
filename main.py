import datetime
import csv
import json
import traceback
import warnings

from collections import defaultdict
from colorsys import rgb_to_hsv
from functools import lru_cache
from itertools import product
from urllib.parse import quote, unquote

import joblib
import numpy as np
import pandas as pd
from nicegui import app, ui

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app.add_static_files("./static", "static")

team_stats_data: defaultdict = defaultdict(list)
headers: list[str] = []

model_bundle: dict = {}
model = None
scaler = None
required_features: list[str] = []
scaler_features: list[str] = []
stats_tags: list[str] = []

# ---------------------------------------------------------------------------
# Bankroll and Kelly settings
# ---------------------------------------------------------------------------
BANKROLL: float = 75.0          # user bankroll in dollars
# ---------------------------------------------------------------------------
# Odds format selector (American / Decimal / Polymarket shares)
# ---------------------------------------------------------------------------
ODDS_FORMAT: str = "shares"   # options: "american", "decimal", "shares"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_team_stats(file_path: str = "./data/csv/averages.csv") -> None:
    global headers
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = [h.strip() for h in next(reader)]
        for row in reader:
            date_str, team = row[0].strip(), row[1].strip()
            team_stats_data[team].append((date_str, [v.strip() for v in row]))

    for team in team_stats_data:
        team_stats_data[team].sort(reverse=True)


# ---------------------------------------------------------------------------
# Feature-name extraction helpers
# ---------------------------------------------------------------------------

def _extract_feature_names_from_model(mdl) -> list[str]:
    if mdl is None:
        return []

    if hasattr(mdl, "feature_names_in_"):
        return list(mdl.feature_names_in_)

    if hasattr(mdl, "calibrated_classifiers_"):
        for cc in mdl.calibrated_classifiers_:
            inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            names = _extract_feature_names_from_model(inner)
            if names:
                return names

    if hasattr(mdl, "estimators_"):
        estimators = mdl.estimators_
        if isinstance(estimators, dict):
            estimators = list(estimators.values())
        for est in estimators:
            inner = est[1] if isinstance(est, tuple) else est
            names = _extract_feature_names_from_model(inner)
            if names:
                return names

    if hasattr(mdl, "final_estimator_"):
        names = _extract_feature_names_from_model(mdl.final_estimator_)
        if names:
            return names

    return []


def _feature_engineering_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("home_moneyline", "away_moneyline"):
        if col not in df.columns:
            continue
        clean = (
            df[col].astype(str)
            .str.replace("+", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        numeric_odds = pd.to_numeric(clean, errors="coerce")
        implied = np.where(
            numeric_odds >= 100,
            100 / (numeric_odds + 100),
            np.where(
                numeric_odds <= -100,
                np.abs(numeric_odds) / (np.abs(numeric_odds) + 100),
                np.where(
                    (numeric_odds > 1.0) & (numeric_odds < 100.0),
                    1.0 / numeric_odds,
                    np.nan,
                ),
            ),
        )
        df[f"{col}_implied_prob"] = implied

    req = ("home_moneyline_implied_prob", "away_moneyline_implied_prob")
    if all(c in df.columns for c in req):
        df["prob_diff"] = df[req[0]] - df[req[1]]
        df["prob_ratio"] = df[req[0]] / (df[req[1]] + 1e-5)

    if "home_elo" in df.columns and "away_elo" in df.columns:
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_ratio"] = df["home_elo"] / (df["away_elo"] + 1e-5)

    if "home_days_since_last_game" in df.columns and "away_days_since_last_game" in df.columns:
        df["rest_differential"] = (
            pd.to_numeric(df["home_days_since_last_game"], errors="coerce")
            - pd.to_numeric(df["away_days_since_last_game"], errors="coerce")
        )

    return df


def _derive_scaler_features_from_dataset(
    dataset_path: str = "./data/csv/dataset.csv",
) -> list[str]:
    cols_to_drop = {
        "winning_team",
        "date",
        "home_team",
        "away_team",
        "home_moneyline",
        "away_moneyline",
    }

    try:
        df = pd.read_csv(dataset_path, nrows=5)
    except FileNotFoundError:
        print(f"Warning: {dataset_path} not found; scaler feature list may be incomplete.")
        return []

    df = _feature_engineering_training(df)
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    return list(df.columns)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_stats_tags_from_importances(mdl, feat_list: list[str]) -> None:
    global stats_tags
    importance_scores = mdl.feature_importances_
    sorted_indices = np.argsort(importance_scores)[::-1]
    unique_stats: list[str] = []

    for i in sorted_indices:
        feat_name = feat_list[i] if i < len(feat_list) else ""
        stat_name = feat_name.replace("home_", "").replace("away_", "").replace("_avg", "")
        for suffix in ["_implied_prob", "_diff", "_ratio", "_home_avg", "_away_avg"]:
            stat_name = stat_name.replace(suffix, "")
        if stat_name and stat_name not in unique_stats and stat_name not in ["date", "team", "winning_team"]:
            unique_stats.append(stat_name)
        if len(unique_stats) == 15:
            break

    stats_tags = unique_stats


def _build_stats_tags_from_voting(mdl, feat_list: list[str]) -> None:
    global stats_tags
    estimators = getattr(mdl, "estimators_", [])
    if isinstance(estimators, dict):
        estimators = list(estimators.values())

    for est in estimators:
        inner = est[1] if isinstance(est, tuple) else est
        if hasattr(inner, "feature_importances_"):
            _build_stats_tags_from_importances(inner, feat_list)
            return

    stats_tags = list(
        dict.fromkeys(
            f.replace("home_", "").replace("away_", "")
            for f in feat_list[:30]
            if not any(f.endswith(s) for s in ["_implied_prob", "_diff", "_ratio"])
        )
    )[:15]


def load_model_bundle(model_path: str = "./model/nba.pkl") -> None:
    global model_bundle, model, scaler, required_features, scaler_features, stats_tags

    print(f"Loading model bundle from {model_path}...")
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")

    bundle_features = model_bundle.get("features", [])
    if bundle_features:
        required_features = [str(f).strip() for f in bundle_features]
    else:
        required_features = _extract_feature_names_from_model(model)
        if not required_features:
            raise RuntimeError(
                "Could not determine required_features from model bundle. "
                "Re-train the model or add a 'features' key to the bundle."
            )

    if scaler is not None:
        if hasattr(scaler, "feature_names_in_") and len(scaler.feature_names_in_) > 0:
            scaler_features = [str(f).strip() for f in scaler.feature_names_in_]
            print(f"Scaler feature list from feature_names_in_: {len(scaler_features)} features")
        else:
            scaler_features = _derive_scaler_features_from_dataset()
            if scaler_features:
                print(f"Scaler feature list derived from dataset.csv: {len(scaler_features)} features")
            else:
                scaler_features = required_features
                print("Warning: using required_features as scaler feature list (fallback).")
    else:
        scaler_features = required_features

    if hasattr(model, "feature_importances_"):
        _build_stats_tags_from_importances(model, required_features)
    else:
        _build_stats_tags_from_voting(model, required_features)

    print(
        f"Model loaded. "
        f"Scaler features: {len(scaler_features)}, "
        f"Model features (RFECV): {len(required_features)}, "
        f"Stats tags: {len(stats_tags)}"
    )


# ---------------------------------------------------------------------------
# Feature engineering (inference)
# ---------------------------------------------------------------------------

def parse_moneyline_to_implied_prob(ml_str: str) -> float:
    """
    Convert an odds string to implied probability (0-1).
    Respects the global ODDS_FORMAT setting.
    Supported formats: "american", "decimal", "shares".
    """
    if ml_str is None or str(ml_str).strip() == "":
        return np.nan

    s = str(ml_str).strip()
    fmt = ODDS_FORMAT  # use the global dropdown value

    if fmt == "shares":
        # Polymarket share price: 0.23 means 23% chance
        try:
            prob = float(s)
        except ValueError:
            return np.nan
        if 0.0 <= prob <= 1.0:
            return prob
        return np.nan

    if fmt == "decimal":
        try:
            odds = float(s)
        except ValueError:
            return np.nan
        if odds < 1.0:
            return np.nan
        return 1.0 / odds

    # ----- American format (default) -----
    s = s.replace("+", "").replace(",", "").replace(" ", "")
    try:
        odds = float(s)
    except ValueError:
        return np.nan

    if odds >= 100:
        return 100 / (odds + 100)
    if odds <= -100:
        return abs(odds) / (abs(odds) + 100)
    # For safety, treat numbers like 2.5 as decimal odds only if format is 'decimal'
    # (should not happen here, but fallback)
    if 1.0 < odds < 100:
        return 1.0 / odds
    return np.nan


def valid_moneylines(home_ml: str, away_ml: str) -> bool:
    return not np.isnan(parse_moneyline_to_implied_prob(home_ml)) and not np.isnan(
        parse_moneyline_to_implied_prob(away_ml)
    )


def feature_engineering_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("home_moneyline", "away_moneyline"):
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_implied_prob"] = df[col].apply(parse_moneyline_to_implied_prob)

    req = ("home_moneyline_implied_prob", "away_moneyline_implied_prob")
    if all(c in df.columns for c in req):
        hp = pd.to_numeric(df[req[0]], errors="coerce")
        ap = pd.to_numeric(df[req[1]], errors="coerce")
        df["prob_diff"] = hp - ap
        df["prob_ratio"] = hp / (ap + 1e-5)

    if "home_elo" in df.columns and "away_elo" in df.columns:
        he = pd.to_numeric(df["home_elo"], errors="coerce")
        ae = pd.to_numeric(df["away_elo"], errors="coerce")
        df["elo_diff"] = he - ae
        df["elo_ratio"] = he / (ae + 1e-5)

    if "home_days_since_last_game" in df.columns and "away_days_since_last_game" in df.columns:
        hr = pd.to_numeric(df["home_days_since_last_game"], errors="coerce")
        ar = pd.to_numeric(df["away_days_since_last_game"], errors="coerce")
        df["rest_differential"] = hr - ar

    return df


# ---------------------------------------------------------------------------
# Core inference utilities
# ---------------------------------------------------------------------------

def _align_to_feature_list(
    df: pd.DataFrame,
    feature_list: list[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    feature_list = [str(f).strip() for f in feature_list]

    data: dict[str, np.ndarray] = {}
    for feat in feature_list:
        if feat in df.columns:
            vals = pd.to_numeric(df[feat], errors="coerce")
            data[feat] = vals.fillna(fill_value).values
        else:
            data[feat] = np.full(len(df), fill_value, dtype=float)

    return pd.DataFrame(data, index=df.index)


def _prepare_model_input(df_features: pd.DataFrame) -> np.ndarray:
    if scaler is not None and scaler_features:
        df_for_scaler = _align_to_feature_list(df_features, scaler_features)
        try:
            x_scaled_full = scaler.transform(df_for_scaler.values)
        except Exception as e:
            print(f"Warning: scaler.transform failed ({e}); falling back to unscaled.")
            x_scaled_full = df_for_scaler.values

        df_scaled_full = pd.DataFrame(
            x_scaled_full,
            columns=scaler_features,
            index=df_features.index,
        )
    else:
        df_scaled_full = df_features.copy()

    df_model_input = _align_to_feature_list(df_scaled_full, required_features)
    return df_model_input.values


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

load_team_stats()
load_model_bundle()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

stat_to_full_name_desc: dict[str, str] = {
    "pts": "Points Per Game (PPG)",
    "fg": "Field Goals (FG)",
    "fga": "Field Goal Attempts (FGA)",
    "fg_pct": "Field Goal % (FG%)",
    "fg3": "3-Point Field Goals (3P)",
    "fg3a": "3-Point Field Goal Attempts (3PA)",
    "fg3_pct": "3-Point Field Goal % (3P%)",
    "fg2": "2-Point Field Goals (2P)",
    "fg2a": "2-Point Field Goal Attempts (2PA)",
    "fg2_pct": "2-Point Field Goal % (2P%)",
    "ft": "Free Throws (FT)",
    "fta": "Free Throw Attempts (FTA)",
    "ft_pct": "Free Throw % (FT%)",
    "orb": "Offensive Rebounds (ORB)",
    "drb": "Defensive Rebounds (DRB)",
    "trb": "Total Rebounds (TRB)",
    "ast": "Assists (AST)",
    "stl": "Steals (STL)",
    "blk": "Blocks (BLK)",
    "tov": "Turnovers (TOV)",
    "pf": "Personal Fouls (PF)",
    "ortg": "Offensive Rating (ORtg)",
    "drtg": "Defensive Rating (DRtg)",
    "pace": "Pace (Possessions per 48 minutes)",
    "ftr": "Free Throw Attempt Rate (FTr)",
    "3ptar": "3-Point Attempt Rate (3PAr)",
    "ts": "True Shooting % (TS%)",
    "trb_pct": "Total Rebound % (TRB%)",
    "ast_pct": "Assist % (AST%)",
    "stl_pct": "Steal % (STL%)",
    "blk_pct": "Block % (BLK%)",
    "efg_pct": "Effective Field Goal % (eFG%)",
    "tov_pct": "Turnover % (TOV%)",
    "orb_pct": "Offensive Rebound % (ORB%)",
    "ft_rate": "Free Throws Per Field Goal Attempt (FT/FGA)",
    "ast_tov": "Assist-to-Turnover (AST/TOV)",
    "ast_ratio": "Assist Ratio (ASTr)",
    "elo": "ELO Rating",
    "days_since_last_game": "Days Since Last Game",
    "is_back_to_back": "Back-to-Back Game",
    "rest_differential": "Rest Differential (Home - Away)",
}

team_color_codes: dict[str, list[str]] = {
    "Atlanta Hawks": ["#e03a3e", "#c1d32f"],
    "Boston Celtics": ["#007a33", "#ba9653", "#963821"],
    "Brooklyn Nets": ["#000000"],
    "Charlotte Hornets": ["#1d1160", "#00788c"],
    "Chicago Bulls": ["#ce1141"],
    "Cleveland Cavaliers": ["#860038", "#fdbb30"],
    "Dallas Mavericks": ["#00538c", "#002b5e"],
    "Denver Nuggets": ["#0e2240", "#fec524", "#8b2131", "#1d428a"],
    "Detroit Pistons": ["#c8102e", "#1d42ba", "#002d62"],
    "Golden State Warriors": ["#1d428a", "#ffc72c"],
    "Houston Rockets": ["#ce1141"],
    "Indiana Pacers": ["#002d62", "#fdbb30"],
    "Los Angeles Clippers": ["#c8102e", "#1d428a"],
    "Los Angeles Lakers": ["#552583", "#f9a01b"],
    "Memphis Grizzlies": ["#5d76a9", "#12173f", "#f5b112"],
    "Miami Heat": ["#98002e", "#f9a01b"],
    "Milwaukee Bucks": ["#00471b", "#0077c0"],
    "Minnesota Timberwolves": ["#0c2340", "#236192", "#78be20"],
    "New Orleans Pelicans": ["#0c2340", "#c8102e", "#85714d"],
    "New York Knicks": ["#006bb6", "#f58426"],
    "Oklahoma City Thunder": ["#007ac1", "#ef3b24", "#002d62"],
    "Orlando Magic": ["#0077c0"],
    "Philadelphia 76ers": ["#006bb6", "#ed174c", "#002b5c"],
    "Phoenix Suns": ["#1d1160", "#e56020", "#ffcd00", "#b95915"],
    "Portland Trail Blazers": ["#e03a3e"],
    "Sacramento Kings": ["#5a2d81", "#63727a"],
    "San Antonio Spurs": ["#c4ced4", "#000000"],
    "Toronto Raptors": ["#ce1141", "#b4975a"],
    "Utah Jazz": ["#753bbd"],
    "Washington Wizards": ["#002b5c", "#e31837"],
}

lower_better_stats: set[str] = {"tov", "pf", "drtg", "tov_pct", "tov_to_poss"}
today = datetime.date.today().isoformat()


def get_best_color_pair(team1: str, team2: str) -> tuple[str, str]:
    def hex_to_hsv(hex_color: str) -> tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        return rgb_to_hsv(r, g, b)

    colors_team1 = team_color_codes.get(team1, [])
    colors_team2 = team_color_codes.get(team2, [])
    if not colors_team1 or not colors_team2:
        return "#333333", "#666666"

    best_pair = None
    max_contrast = -1
    for color1, color2 in product(colors_team1, colors_team2):
        _, s1, v1 = hex_to_hsv(color1)
        _, s2, v2 = hex_to_hsv(color2)
        contrast = abs(v1 - v2) + abs(s1 - s2)
        if contrast > max_contrast:
            max_contrast = contrast
            best_pair = (color1, color2)

    return best_pair if best_pair else ("#333333", "#666666")


def to_json_safe_dict(data: dict) -> dict:
    safe = {}
    for key, value in data.items():
        try:
            if pd.isna(value):
                safe[key] = None
                continue
        except Exception:
            pass

        if isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = float(value)
        else:
            safe[key] = value
    return safe


# ---------------------------------------------------------------------------
# Schedule & stat helpers
# ---------------------------------------------------------------------------

def extract_games(date: str) -> list[dict[str, str]]:
    games = []
    try:
        with open("./data/csv/schedule.csv", "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["date"] == date:
                    games.append(
                        {
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                        }
                    )
    except FileNotFoundError:
        print("Warning: schedule.csv not found at ./data/csv/schedule.csv")
    return games


@lru_cache(maxsize=128)
def find_most_recent_stats(team_name: str, target_date: str):
    if team_name not in team_stats_data:
        return None, None
    for date_str, row in team_stats_data[team_name]:
        if date_str < target_date:
            return headers[2:], row[2:]
    return None, None


def build_game_data_for_date(date: str) -> list[dict]:
    games = []

    for game in extract_games(date):
        enriched_game = {
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "date": date,
        }

        stat_labels, home_stats = find_most_recent_stats(game["home_team"], date)
        if stat_labels and home_stats:
            for i, lbl in enumerate(stat_labels):
                enriched_game[f"home_{lbl}"] = home_stats[i]

        stat_labels, away_stats = find_most_recent_stats(game["away_team"], date)
        if stat_labels and away_stats:
            for i, lbl in enumerate(stat_labels):
                enriched_game[f"away_{lbl}"] = away_stats[i]

        games.append(enriched_game)

    return games


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------

def predict_game_proba(
    game_data: dict,
    manual_home_ml: str = "",
    manual_away_ml: str = "",
) -> tuple[float, float]:
    try:
        df = pd.DataFrame([game_data])

        if manual_home_ml:
            df["home_moneyline"] = manual_home_ml
        if manual_away_ml:
            df["away_moneyline"] = manual_away_ml

        df = feature_engineering_inference(df)

        for col in df.columns:
            if col.startswith(("home_", "away_")) and col not in [
                "home_team",
                "away_team",
                "home_moneyline",
                "away_moneyline",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        drop_cols = ["home_team", "away_team", "date", "winner", "home_prob", "away_prob"]
        df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        x = _prepare_model_input(df_features)
        proba = model.predict_proba(x)[0]
        return round(float(proba[0]) * 100, 1), round(float(proba[1]) * 100, 1)

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return 50.0, 50.0


# ---------------------------------------------------------------------------
# Edge & Kelly calculation (two‑bin strategy)
# ---------------------------------------------------------------------------

def _binned_kelly_bet(
    predicted_prob_pct: float,
    moneyline_str: str,
    opposing_moneyline_str: str,
    side: str,
    bankroll: float,
) -> dict | None:
    """
    Compute edge and bet amount using two‑bin Kelly strategy.
    Edge is computed against raw implied probabilities.
    side: 'home' or 'away'.
    Bins:
      2.5% ≤ edge < 5.5%  →  1.0 × Kelly (full Kelly)
      edge ≥ 5.5%         →  0.25 × Kelly
    """
    if not moneyline_str or not moneyline_str.strip():
        return None
    if not opposing_moneyline_str or not opposing_moneyline_str.strip():
        return None

    implied = parse_moneyline_to_implied_prob(moneyline_str)
    opposing_implied = parse_moneyline_to_implied_prob(opposing_moneyline_str)
    if np.isnan(implied) or np.isnan(opposing_implied) or implied <= 0 or opposing_implied <= 0:
        return None

    fair_prob = implied

    if np.isnan(fair_prob):
        return None

    p = predicted_prob_pct / 100.0
    edge = p - fair_prob
    edge_pct = edge * 100
    decimal_odds = 1.0 / implied if implied > 0 else 0.0

    if decimal_odds < 1.0:
        return None

    b = decimal_odds - 1.0
    if b <= 0:
        return None

    f = (b * p - (1.0 - p)) / b   # full Kelly
    if f < 0:
        f = 0.0

    has_edge = edge_pct >= 2.5
    if has_edge:
        if edge_pct < 5.5:
            kelly_fraction = 1.0
        else:
            kelly_fraction = 0.25
        f_adjusted = f * kelly_fraction
        bet_amount = f_adjusted * bankroll
    else:
        kelly_fraction = 0.0
        f_adjusted = 0.0
        bet_amount = 0.0

    return {
        "implied_prob": implied * 100,
        "edge_pct": edge_pct,
        "decimal_odds": decimal_odds,
        "full_kelly_fraction": f,
        "adjusted_kelly_fraction": f_adjusted,
        "kelly_multiplier": kelly_fraction,
        "bet_amount": bet_amount,
        "has_edge": has_edge,
    }


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

class GameCard(ui.card):
    _instances = []

    def __init__(self, game: dict, date: str) -> None:
        super().__init__()
        self.game = dict(game)
        self.date = date
        self.home_color, self.away_color = get_best_color_pair(
            self.game["home_team"], self.game["away_team"]
        )
        self.home_prob: float | None = None
        self.away_prob: float | None = None
        self.status_message = "Enter both moneylines to calculate win probabilities."

        self.classes("m-4 p-8 rounded-2xl shadow-md border w-[720px]").style(
            "background-color: #e3e4e6;"
        )

        GameCard._instances.append(self)

        with self:
            with ui.row(align_items="center").classes("items-center justify-between w-full"):
                ui.image(f"static/{self.game['home_team']}.png").classes("w-28")
                ui.image("static/vs.png").classes("w-14")
                ui.image(f"static/{self.game['away_team']}.png").classes("w-28")

            with ui.row().classes("w-full flex justify-center items-center mt-2"):
                ui.button(
                    "Details",
                    icon="info",
                    on_click=self.open_details,
                ).props("unelevated rounded color=grey-2 text-color=grey-5")

            ui.separator().classes("w-full my-2")

            # Dynamically set placeholders according to global format
            home_ph = {
                "american": "e.g. -150",
                "decimal": "e.g. 2.50",
                "shares": "e.g. 0.23"
            }[ODDS_FORMAT]
            away_ph = {
                "american": "e.g. +130",
                "decimal": "e.g. 1.80",
                "shares": "e.g. 0.81"
            }[ODDS_FORMAT]

            with ui.row().classes("w-full gap-3 items-end"):
                self.home_ml_input = (
                    ui.input(
                        f"{self.game['home_team']} ML (Home)",
                        placeholder=home_ph,
                        on_change=lambda e: self.calculate_prediction(),
                    )
                    .classes("flex-1")
                    .props("outlined clearable")
                )
                self.away_ml_input = (
                    ui.input(
                        f"{self.game['away_team']} ML (Away)",
                        placeholder=away_ph,
                        on_change=lambda e: self.calculate_prediction(),
                    )
                    .classes("flex-1")
                    .props("outlined clearable")
                )
                ui.button(
                    "Calculate",
                    on_click=self.calculate_prediction,
                    icon="play_arrow",
                ).props("rounded color=orange-14").classes("mb-1")

            self.render_prediction_panel()

            with ui.expansion().classes(
                "w-full shadow-md bg-gray-100 rounded-2xl overflow-hidden mx-auto mt-4"
            ).props("duration=550 hide-expand-icon") as expansion:

                def toggle_label() -> None:
                    label.set_text("Click to hide" if expansion.value else "Click for more")
                    icon.set_name("expand_less" if expansion.value else "expand_more")

                expansion.on("update:model-value", toggle_label)

                with expansion.add_slot("header"):
                    with ui.row().classes("w-full justify-center items-center"):
                        label = ui.label("Click for more").classes("text-md font-bold text-center")
                        icon = ui.icon("expand_more").classes("text-xl")

                with ui.row().classes("w-full"):
                    with ui.column().classes("items-start flex-1"):
                        for stat in stats_tags:
                            home_key = f"home_{stat}"
                            away_key = f"away_{stat}"
                            if home_key in self.game and away_key in self.game:
                                try:
                                    home_val = float(self.game[home_key])
                                    away_val = float(self.game[away_key])
                                    max_val = max(abs(home_val), abs(away_val), 1e-5)
                                    diff = abs(home_val - away_val) / max_val * 100
                                    is_lower_better = stat in lower_better_stats
                                    if diff >= 3:
                                        if (home_val > away_val and not is_lower_better) or (
                                            home_val < away_val and is_lower_better
                                        ):
                                            style = "text-green-600 font-bold"
                                        else:
                                            style = "text-red-600 font-bold"
                                    else:
                                        style = "text-black"
                                    ui.label(f"{home_val:.2f}").classes(f"text-left text-sm {style}")
                                except (ValueError, TypeError):
                                    ui.label("-").classes("text-left text-sm text-gray-400")
                            else:
                                ui.label("-").classes("text-left text-sm text-gray-400")

                    with ui.column().classes("items-center flex-3"):
                        for stat in stats_tags:
                            ui.label(stat_to_full_name_desc.get(stat, stat)).classes(
                                "text-center text-sm font-bold"
                            )

                    with ui.column().classes("items-end flex-1"):
                        for stat in stats_tags:
                            home_key = f"home_{stat}"
                            away_key = f"away_{stat}"
                            if home_key in self.game and away_key in self.game:
                                try:
                                    home_val = float(self.game[home_key])
                                    away_val = float(self.game[away_key])
                                    max_val = max(abs(home_val), abs(away_val), 1e-5)
                                    diff = abs(home_val - away_val) / max_val * 100
                                    is_lower_better = stat in lower_better_stats
                                    if diff >= 3:
                                        if (away_val > home_val and not is_lower_better) or (
                                            away_val < home_val and is_lower_better
                                        ):
                                            style = "text-green-600 font-bold"
                                        else:
                                            style = "text-red-600 font-bold"
                                    else:
                                        style = "text-black"
                                    ui.label(f"{away_val:.2f}").classes(f"text-right text-sm {style}")
                                except (ValueError, TypeError):
                                    ui.label("-").classes("text-right text-sm text-gray-400")
                            else:
                                ui.label("-").classes("text-right text-sm text-gray-400")

    @classmethod
    def update_placeholders(cls):
        home_ph = {
            "american": "e.g. -150",
            "decimal": "e.g. 2.50",
            "shares": "e.g. 0.23"
        }[ODDS_FORMAT]
        away_ph = {
            "american": "e.g. +130",
            "decimal": "e.g. 1.80",
            "shares": "e.g. 0.81"
        }[ODDS_FORMAT]
        for card in cls._instances:
            card.home_ml_input.props(f'placeholder="{home_ph}"')
            card.away_ml_input.props(f'placeholder="{away_ph}"')

    def open_details(self) -> None:
        payload = self.serializable_game()
        ui.navigate.to(f"/{self.date}/{quote(json.dumps(payload))}")

    def serializable_game(self) -> dict:
        data = dict(self.game)
        data["home_prob"] = self.home_prob
        data["away_prob"] = self.away_prob
        data["home_moneyline"] = self.home_ml_input.value.strip() if self.home_ml_input.value else ""
        data["away_moneyline"] = self.away_ml_input.value.strip() if self.away_ml_input.value else ""
        return to_json_safe_dict(data)

    @ui.refreshable
    def render_prediction_panel(self) -> None:
        if self.home_prob is None or self.away_prob is None:
            with ui.column().classes("w-full items-center mt-3"):
                ui.label(self.status_message).classes("text-center text-sm text-grey-7")
            return

        # ---- VIG calculation (for display only) ----
        home_ml = self.game.get("home_moneyline", "")
        away_ml = self.game.get("away_moneyline", "")
        vig_pct = None
        if home_ml and away_ml:
            home_imp = parse_moneyline_to_implied_prob(home_ml)
            away_imp = parse_moneyline_to_implied_prob(away_ml)
            if not np.isnan(home_imp) and not np.isnan(away_imp):
                total_implied = home_imp + away_imp
                vig_pct = (total_implied - 1) * 100  # vig in percent

        with ui.column().classes("w-full gap-2 mt-3"):
            with ui.row(align_items="stretch").classes("justify-between w-full"):
                with ui.column(align_items="start"):
                    ui.label(self.game["home_team"]).classes("text-left text-lg font-bold")
                    ui.label(f"W {self.home_prob} %").classes("text-left text-lg font-bold")
                with ui.column(align_items="end"):
                    ui.label(self.game["away_team"]).classes("text-right text-lg font-bold")
                    ui.label(f"W {self.away_prob} %").classes("text-right text-lg font-bold")

            with ui.element("div").classes("flex w-full h-6"):
                ui.element("div").style(
                    f"flex: {self.home_prob}; background-color: {self.home_color}"
                ).classes("rounded-md mr-1")
                ui.element("div").style(
                    f"flex: {self.away_prob}; background-color: {self.away_color}"
                ).classes("rounded-md ml-1")

            # Vig display (very noticeable)
            if vig_pct is not None:
                with ui.row().classes("w-full justify-center mt-2"):
                    ui.label(f"Vig: {vig_pct:+.2f}%").classes(
                        "text-sm font-bold text-orange-800 bg-yellow-100 px-2 py-1 rounded"
                    )

            self._render_edge_section()

    def _render_edge_section(self) -> None:
        home_ml = self.game.get("home_moneyline", "")
        away_ml = self.game.get("away_moneyline", "")

        # ----- compute vig for the betting disallow rule -----
        vig_pct = None
        if home_ml and away_ml:
            home_imp = parse_moneyline_to_implied_prob(home_ml)
            away_imp = parse_moneyline_to_implied_prob(away_ml)
            if not np.isnan(home_imp) and not np.isnan(away_imp):
                total_implied = home_imp + away_imp
                vig_pct = (total_implied - 1) * 100

        home_res = (
            _binned_kelly_bet(self.home_prob, home_ml, away_ml, "home", BANKROLL)
            if self.home_prob is not None
            else None
        )
        away_res = (
            _binned_kelly_bet(self.away_prob, away_ml, home_ml, "away", BANKROLL)
            if self.away_prob is not None
            else None
        )

        # ----- disallow betting when vig > 6.5% -----
        vig_too_high = False
        if vig_pct is not None and vig_pct > 6.5:
            vig_too_high = True
            if home_res:
                home_res["has_edge"] = False
                home_res["bet_amount"] = 0.0
            if away_res:
                away_res["has_edge"] = False
                away_res["bet_amount"] = 0.0

        with ui.row().classes("w-full justify-between gap-4 mt-3"):
            # Home side
            if home_res is not None:
                has_edge = home_res["has_edge"]
                box_class = "bg-green-100" if has_edge else "bg-red-100"
                text_color = "text-green-700" if has_edge else "text-red-700"
                with ui.column(align_items="start").classes(f"{box_class} rounded p-2 flex-1"):
                    ui.label(f"🏠 {self.game['home_team']}").classes("font-bold text-sm")
                    ui.label(f"Edge: {home_res['edge_pct']:.2f}%").classes(f"text-xs {text_color}")
                    if has_edge:
                        mult = home_res["kelly_multiplier"]
                        ui.label(f"Bet (Kelly {mult:.2f}×): ${home_res['bet_amount']:.2f}").classes("text-xs font-semibold")
                    else:
                        reason = "No bet (vig > 6.5%)" if vig_too_high else "No bet (edge <2.5%)"
                        ui.label(reason).classes("text-xs text-grey-7")
            else:
                with ui.column(align_items="start").classes("bg-grey-3 rounded p-2 flex-1"):
                    ui.label(f"🏠 {self.game['home_team']}").classes("font-bold text-sm")
                    ui.label("Enter valid odds").classes("text-xs text-grey-7")

            # Away side
            if away_res is not None:
                has_edge = away_res["has_edge"]
                box_class = "bg-green-100" if has_edge else "bg-red-100"
                text_color = "text-green-700" if has_edge else "text-red-700"
                with ui.column(align_items="end").classes(f"{box_class} rounded p-2 flex-1"):
                    ui.label(f"✈️ {self.game['away_team']}").classes("font-bold text-sm")
                    ui.label(f"Edge: {away_res['edge_pct']:.2f}%").classes(f"text-xs {text_color}")
                    if has_edge:
                        mult = away_res["kelly_multiplier"]
                        ui.label(f"Bet (Kelly {mult:.2f}×): ${away_res['bet_amount']:.2f}").classes("text-xs font-semibold")
                    else:
                        reason = "No bet (vig > 6.5%)" if vig_too_high else "No bet (edge <2.5%)"
                        ui.label(reason).classes("text-xs text-grey-7")
            else:
                with ui.column(align_items="end").classes("bg-grey-3 rounded p-2 flex-1"):
                    ui.label(f"✈️ {self.game['away_team']}").classes("font-bold text-sm")
                    ui.label("Enter valid odds").classes("text-xs text-grey-7")

        # Show Kelly bin label only if there is an active bet
        active_home = home_res["has_edge"] if home_res and home_res["has_edge"] else None
        active_away = away_res["has_edge"] if away_res and away_res["has_edge"] else None
        if active_home or active_away:
            with ui.row().classes("w-full justify-center mt-1"):
                ui.label("Kelly bins: 2.5-5.5%→1.0×, ≥5.5%→0.25×").classes("text-xs text-grey-6")

    def calculate_prediction(self) -> None:
        home_ml = self.home_ml_input.value.strip() if self.home_ml_input.value else ""
        away_ml = self.away_ml_input.value.strip() if self.away_ml_input.value else ""

        if not home_ml or not away_ml:
            self.home_prob = None
            self.away_prob = None
            self.status_message = "Enter both moneylines to calculate win probabilities."
            self.render_prediction_panel.refresh()
            return

        if not valid_moneylines(home_ml, away_ml):
            self.home_prob = None
            self.away_prob = None
            self.status_message = "Enter valid moneylines like -150 and +130."
            self.render_prediction_panel.refresh()
            return

        hp, ap = predict_game_proba(self.game, home_ml, away_ml)
        self.home_prob = hp
        self.away_prob = ap
        self.game["home_moneyline"] = home_ml
        self.game["away_moneyline"] = away_ml
        self.game["home_prob"] = hp
        self.game["away_prob"] = ap
        self.status_message = ""
        self.render_prediction_panel.refresh()


class GameList:
    def __init__(self, date: str) -> None:
        self.date = date

    def render(self) -> None:
        try:
            games = build_game_data_for_date(self.date)

            if not games:
                ui.label("No games found for this date").classes("text-center text-lg text-white")
                return

            for game in games:
                GameCard(game, self.date)

        except FileNotFoundError as e:
            ui.label(f"Error: Could not find required files - {e}").classes("text-red-600")
        except KeyError as e:
            ui.label(f"Error: Missing expected data field - {e}").classes("text-red-600")
        except ValueError as e:
            ui.label(f"Error: Invalid data format - {e}").classes("text-red-600")
        except Exception as e:
            ui.label(f"Unexpected error: {e}").classes("text-red-600")
            traceback.print_exc()


class H2HPlot:
    def __init__(
        self,
        stat: str,
        date: str,
        window: int,
        team1: str,
        team2: str,
        home_color: str,
        away_color: str,
        csv_path: str,
    ) -> None:
        self.stat = stat
        self.date = date
        self.window = window
        self.team1 = team1
        self.team2 = team2
        self.home_color = home_color
        self.away_color = away_color
        self.path = csv_path
        self.plot_stat()

    @ui.refreshable
    def plot_stat(self) -> None:
        try:
            df = pd.read_csv(self.path, parse_dates=["date"])

            if self.stat not in df.columns:
                matching = [c for c in df.columns if self.stat in c.lower()]
                if matching:
                    self.stat = matching[0]
                else:
                    ui.label(f"Stat '{self.stat}' not found").classes("text-red-500")
                    return

            start_date = pd.to_datetime(self.date)

            def make_series(frame: pd.DataFrame, team: str, color: str) -> dict:
                df_team = (
                    frame[(frame["team"] == team) & (frame["date"] < start_date)]
                    .sort_values("date", ascending=False)
                    .head(self.window)
                    .sort_values("date", ascending=True)
                )
                data = []
                for _, row in df_team.iterrows():
                    try:
                        stat_value = float(row[self.stat])
                    except Exception:
                        continue
                    data.append([int(pd.Timestamp(row["date"]).timestamp() * 1000), stat_value])

                return {"name": team, "data": data, "color": color}

            series = [
                make_series(df, self.team1, self.home_color),
                make_series(df, self.team2, self.away_color),
            ]

            config = {
                "chart": {"type": "line", "spacingTop": 25, "spacingBottom": 25},
                "title": {
                    "text": f"{self.team1} vs {self.team2} - {stat_to_full_name_desc.get(self.stat, self.stat)}"
                },
                "xAxis": {"type": "datetime", "labels": {"format": "{value:%b %d}"}},
                "yAxis": {"title": {"text": stat_to_full_name_desc.get(self.stat, self.stat)}},
                "legend": {"layout": "horizontal", "align": "center", "verticalAlign": "top"},
                "tooltip": {"xDateFormat": "%b %d, %Y", "shared": True},
                "series": series,
            }
            ui.highchart(options=config).classes("rounded-lg")

        except Exception as e:
            ui.label(f"Plot error: {str(e)[:100]}").classes("text-red-500")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@ui.page("/")
def redirect() -> None:
    ui.navigate.to(f"/{today}")


@ui.page("/{date}")
def home(date: str) -> None:
    ui.add_css(".nicegui-content { margin: 0; padding: 0; height: 100vh; }")
    ui.add_css(".w-1\\/3, .w-2\\/3 { border: none; box-shadow: none; }")

    def change_odds_format(new_format: str):
        global ODDS_FORMAT
        ODDS_FORMAT = new_format
        GameCard.update_placeholders()

    with ui.element("div").classes("w-full h-full flex"):
        with ui.element("div").classes(
            "w-1/3 flex justify-center items-center fixed h-full"
        ).style("background-color: #333436;"):
            date_container = ui.element("div")

        with ui.element("div").classes("w-2/3 ml-auto h-full overflow-auto p-16").style(
            "background-color: #5a5f70;"
        ):
            cards_container = ui.element("div")

        with cards_container:
            games_list = GameList(date)
            with ui.column(align_items="center").classes("w-full"):
                games_list.render()

        with date_container:
            with ui.column(align_items="center").classes("gap-2"):

                # Bankroll setting
                ui.label("Bankroll ($)").style("color: #e3e4e6;").classes("text-sm")
                ui.number(
                    value=BANKROLL,
                    format="%.2f",
                    step=100,
                    min=1,
                    max=1_000_000,
                    on_change=lambda e: set_bankroll(e.value),
                ).props("outlined color=orange-14 bg-color=grey-2").classes("rounded w-[200px]")

                # Odds format selector (now includes Polymarket)
                ui.label("Odds Format").style("color: #e3e4e6;").classes("text-sm mt-2")
                format_options = {
                    "American (-150)": "american",
                    "Decimal (2.50)": "decimal",
                    "Polymarket (0.23)": "shares",
                }
                current_key = [k for k, v in format_options.items() if v == ODDS_FORMAT][0]
                ui.select(
                    list(format_options.keys()),
                    value=current_key,
                    on_change=lambda e: change_odds_format(format_options[e.value]),
                    with_input=True,
                ).props("outlined color=orange-14 bg-color=grey-2").classes("rounded w-[200px]")

                # Date picker
                date_picker = (
                    ui.date(date)
                    .bind_value_to(games_list, "date")
                    .style("border-radius: 16px; background-color: #e3e4e6;")
                    .props("minimal color=orange-14")
                    .classes("mt-4")
                )

                ui.button(
                    "Predict",
                    on_click=lambda: ui.navigate.to(f"/{date_picker.value}"),
                ).props("rounded push size=lg color=orange-14").classes("rounded-2xl mt-4")

                ui.label("Select any date. Games appear only if schedule.csv has entries for that date.").style(
                    "color: #e3e4e6;"
                ).classes("text-center max-w-xs text-sm")

                with ui.row().classes("mt-4 justify-center items-center gap-2"):
                    ui.label("Data provided by: ").style("color: #e3e4e6;")
                    ui.link("Basketball Reference", "https://www.basketball-reference.com").style(
                        "color: #e3e4e6;"
                    )


def set_bankroll(val: float) -> None:
    global BANKROLL
    BANKROLL = max(1.0, val)


@ui.page("/{date}/{game}")
def game(date: str, game: str) -> None:
    try:
        game_data = json.loads(unquote(game))
    except json.JSONDecodeError:
        ui.label("Invalid game data").classes("text-red-600")
        return

    ui.add_css(".nicegui-content { margin: 0; padding: 0; min-height: 100vh; }")
    ui.add_css(".nicegui-content { display: flex; flex-direction: column; }")
    ui.add_css(".nicegui-content { justify-content: center; align-items: center; }")
    ui.add_css(".nicegui-content { background-color: #5a5f70; }")

    home_color, away_color = get_best_color_pair(game_data["home_team"], game_data["away_team"])

    with ui.page_sticky("top-left", x_offset=32, y_offset=32).classes("mt-8 ml-8"):
        ui.icon("arrow_back").classes("cursor-pointer text-3xl").style(
            "color: #e3e4e6"
        ).on("click", lambda: ui.navigate.to(f"/{date}"))

    card = (
        ui.card()
        .classes("m-4 p-6 rounded-2xl shadow-md border w-[900px]")
        .style("background-color: #e3e4e6;")
    )

    with card:
        with ui.row(align_items="center").classes("items-center justify-between w-full"):
            ui.image(f"static/{game_data['home_team']}.png").classes("w-28")
            ui.image("static/vs.png").classes("w-12")
            ui.image(f"static/{game_data['away_team']}.png").classes("w-28")

        detail_state = {
            "home_prob": game_data.get("home_prob"),
            "away_prob": game_data.get("away_prob"),
            "status": "Enter both moneylines to calculate win probabilities."
            if game_data.get("home_prob") is None or game_data.get("away_prob") is None
            else "",
        }

        # Placeholder based on current format
        home_ph = {
            "american": "e.g. -150",
            "decimal": "e.g. 2.50",
            "shares": "e.g. 0.23"
        }[ODDS_FORMAT]
        away_ph = {
            "american": "e.g. +130",
            "decimal": "e.g. 1.80",
            "shares": "e.g. 0.81"
        }[ODDS_FORMAT]

        with ui.row().classes("w-full gap-3 items-end mt-4"):
            home_ml_input = (
                ui.input(
                    f"{game_data['home_team']} ML (Home)",
                    placeholder=home_ph,
                    value=game_data.get("home_moneyline", "") or "",
                )
                .classes("flex-1")
                .props("outlined clearable")
            )
            away_ml_input = (
                ui.input(
                    f"{game_data['away_team']} ML (Away)",
                    placeholder=away_ph,
                    value=game_data.get("away_moneyline", "") or "",
                )
                .classes("flex-1")
                .props("outlined clearable")
            )

        @ui.refreshable
        def render_detail_prediction() -> None:
            if detail_state["home_prob"] is None or detail_state["away_prob"] is None:
                ui.label(detail_state["status"]).classes("text-center text-sm text-grey-7 w-full")
                return

            # VIG calculation for detail page
            hm = home_ml_input.value.strip() if home_ml_input.value else ""
            aw = away_ml_input.value.strip() if away_ml_input.value else ""
            vig_pct = None
            if hm and aw:
                home_imp = parse_moneyline_to_implied_prob(hm)
                away_imp = parse_moneyline_to_implied_prob(aw)
                if not np.isnan(home_imp) and not np.isnan(away_imp):
                    total_implied = home_imp + away_imp
                    vig_pct = (total_implied - 1) * 100

            with ui.column().classes("w-full gap-2"):
                with ui.row(align_items="stretch").classes("justify-between w-full"):
                    with ui.column(align_items="start"):
                        ui.label(game_data["home_team"]).classes("text-left text-md font-bold")
                        ui.label(f"W {detail_state['home_prob']} %").classes("text-left text-md font-bold")
                    with ui.column(align_items="end"):
                        ui.label(game_data["away_team"]).classes("text-right text-md font-bold")
                        ui.label(f"W {detail_state['away_prob']} %").classes("text-right text-md font-bold")

                with ui.element("div").classes("flex w-full h-6"):
                    ui.element("div").style(
                        f"flex: {detail_state['home_prob']}; background-color: {home_color}"
                    ).classes("rounded-md mr-1")
                    ui.element("div").style(
                        f"flex: {detail_state['away_prob']}; background-color: {away_color}"
                    ).classes("rounded-md ml-1")

                if vig_pct is not None:
                    with ui.row().classes("w-full justify-center mt-2"):
                        ui.label(f"Vig: {vig_pct:+.2f}%").classes(
                            "text-sm font-bold text-orange-800 bg-yellow-100 px-2 py-1 rounded"
                        )

                _render_detail_edge()

        def _render_detail_edge() -> None:
            hm = home_ml_input.value.strip() if home_ml_input.value else ""
            aw = away_ml_input.value.strip() if away_ml_input.value else ""

            # ----- vig for betting disallow rule -----
            vig_pct = None
            if hm and aw:
                home_imp = parse_moneyline_to_implied_prob(hm)
                away_imp = parse_moneyline_to_implied_prob(aw)
                if not np.isnan(home_imp) and not np.isnan(away_imp):
                    total_implied = home_imp + away_imp
                    vig_pct = (total_implied - 1) * 100

            home_res = (
                _binned_kelly_bet(detail_state["home_prob"], hm, aw, "home", BANKROLL)
                if detail_state["home_prob"] is not None
                else None
            )
            away_res = (
                _binned_kelly_bet(detail_state["away_prob"], aw, hm, "away", BANKROLL)
                if detail_state["away_prob"] is not None
                else None
            )

            vig_too_high = False
            if vig_pct is not None and vig_pct > 6.5:
                vig_too_high = True
                if home_res:
                    home_res["has_edge"] = False
                    home_res["bet_amount"] = 0.0
                if away_res:
                    away_res["has_edge"] = False
                    away_res["bet_amount"] = 0.0

            with ui.row().classes("w-full justify-between gap-4 mt-3"):
                # Home side
                if home_res is not None:
                    has_edge = home_res["has_edge"]
                    box_class = "bg-green-100" if has_edge else "bg-red-100"
                    text_color = "text-green-700" if has_edge else "text-red-700"
                    with ui.column(align_items="start").classes(f"{box_class} rounded p-2 flex-1"):
                        ui.label(f"🏠 {game_data['home_team']}").classes("font-bold text-sm")
                        ui.label(f"Edge: {home_res['edge_pct']:.2f}%").classes(f"text-xs {text_color}")
                        if has_edge:
                            mult = home_res["kelly_multiplier"]
                            ui.label(f"Bet (Kelly {mult:.2f}×): ${home_res['bet_amount']:.2f}").classes("text-xs font-semibold")
                        else:
                            reason = "No bet (vig > 6.5%)" if vig_too_high else "No bet (edge <2.5%)"
                            ui.label(reason).classes("text-xs text-grey-7")
                else:
                    with ui.column(align_items="start").classes("bg-grey-3 rounded p-2 flex-1"):
                        ui.label(f"🏠 {game_data['home_team']}").classes("font-bold text-sm")
                        ui.label("Enter valid odds").classes("text-xs text-grey-7")

                # Away side
                if away_res is not None:
                    has_edge = away_res["has_edge"]
                    box_class = "bg-green-100" if has_edge else "bg-red-100"
                    text_color = "text-green-700" if has_edge else "text-red-700"
                    with ui.column(align_items="end").classes(f"{box_class} rounded p-2 flex-1"):
                        ui.label(f"✈️ {game_data['away_team']}").classes("font-bold text-sm")
                        ui.label(f"Edge: {away_res['edge_pct']:.2f}%").classes(f"text-xs {text_color}")
                        if has_edge:
                            mult = away_res["kelly_multiplier"]
                            ui.label(f"Bet (Kelly {mult:.2f}×): ${away_res['bet_amount']:.2f}").classes("text-xs font-semibold")
                        else:
                            reason = "No bet (vig > 6.5%)" if vig_too_high else "No bet (edge <2.5%)"
                            ui.label(reason).classes("text-xs text-grey-7")
                else:
                    with ui.column(align_items="end").classes("bg-grey-3 rounded p-2 flex-1"):
                        ui.label(f"✈️ {game_data['away_team']}").classes("font-bold text-sm")
                        ui.label("Enter valid odds").classes("text-xs text-grey-7")

            if (home_res and home_res["has_edge"]) or (away_res and away_res["has_edge"]):
                with ui.row().classes("w-full justify-center mt-1"):
                    ui.label("Kelly bins: 2.5-5.5%→1.0×, ≥5.5%→0.25×").classes("text-xs text-grey-6")

        def calculate_detail_prediction() -> None:
            home_ml = home_ml_input.value.strip() if home_ml_input.value else ""
            away_ml = away_ml_input.value.strip() if away_ml_input.value else ""

            if not home_ml or not away_ml:
                detail_state["home_prob"] = None
                detail_state["away_prob"] = None
                detail_state["status"] = "Enter both moneylines to calculate win probabilities."
                render_detail_prediction.refresh()
                return

            if not valid_moneylines(home_ml, away_ml):
                detail_state["home_prob"] = None
                detail_state["away_prob"] = None
                detail_state["status"] = "Enter valid moneylines like -150 and +130."
                render_detail_prediction.refresh()
                return

            hp, ap = predict_game_proba(game_data, home_ml, away_ml)
            detail_state["home_prob"] = hp
            detail_state["away_prob"] = ap
            detail_state["status"] = ""
            game_data["home_moneyline"] = home_ml
            game_data["away_moneyline"] = away_ml
            game_data["home_prob"] = hp
            game_data["away_prob"] = ap
            render_detail_prediction.refresh()

        with ui.row().classes("w-full justify-center mt-2"):
            ui.button(
                "Calculate Probability",
                on_click=calculate_detail_prediction,
                icon="play_arrow",
            ).props("color=orange-14 rounded")

        render_detail_prediction()

        selectors_section = ui.element("div").classes("w-full mt-6")
        plotting_section = ui.element("div").classes("w-full mt-4")

        with plotting_section:
            plot = H2HPlot(
                "pts",
                date,
                10,
                game_data["home_team"],
                game_data["away_team"],
                home_color,
                away_color,
                "./data/csv/averages.csv",
            )

        with selectors_section:
            with ui.grid(columns="1fr 1fr").classes("w-full"):
                ui.select(
                    stat_to_full_name_desc,
                    label="Select a stat:",
                    value="pts",
                    with_input=True,
                    on_change=plot.plot_stat.refresh,
                ).style("border-radius: 0.25rem;").classes("w-full").props(
                    "outlined color=grey-9 bg-color=grey-2"
                ).bind_value_to(plot, "stat")

                ui.select(
                    [n for n in range(5, 26)],
                    value=10,
                    label="Select game window:",
                    with_input=True,
                    on_change=plot.plot_stat.refresh,
                ).style("border-radius: 0.25rem;").classes("w-full").props(
                    "outlined color=grey-9 bg-color=grey-2"
                ).bind_value_to(plot, "window")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="NBA Betting AI", favicon="static/icon.png", reload=False)