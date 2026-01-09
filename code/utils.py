import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statsmodels.api as sm
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
from nba_api.stats.static import players
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def get_player_stats_from_api(csv_filepath, start_season_str='2020-21', end_season_str='2024-25'):
    """
    Fetches aggregated season statistics for players listed in the provided CSV,
    starting from the specified minimum season (2020-21) up to the end_season_str.

    Args:
        csv_filepath (str): Path to the input CSV file ('nba_5seasons_final_with_nulls.csv').
        start_season_str (str): The earliest NBA season to fetch data for (e.g., '2020-21').
        end_season_str (str): The final NBA season to fetch data for (e.g., '2024-25').

    Returns:
        pd.DataFrame: A consolidated DataFrame of all requested player season stats.
    """

    print("--- Step 1: Loading Data and Preparing Player List ---")

    # --- 1. Load and process player names and initial seasons from the local CSV ---
    df_local = pd.read_csv(csv_filepath)

    def season_to_int(season_id):
        # Converts '2022-23' to 2022
        return int(season_id.split('-')[0])

    df_local['Start_Year_Original'] = df_local['SEASON_ID'].apply(season_to_int)

    # Get the minimum documented start year for each unique player
    player_min_years = df_local.groupby('PLAYER_NAME')['Start_Year_Original'].min().reset_index()

    # --- 2. Determine the full list of NBA seasons to query ---
    min_query_year = season_to_int(start_season_str)  # 2020
    end_query_year = season_to_int(end_season_str)  # 2024

    # Determine the actual start year for fetching (min_query_year vs. player's actual start year)
    player_min_years['Start_Year_Actual'] = player_min_years['Start_Year_Original'].apply(
        lambda x: max(x, min_query_year)
    )

    # --- 3. Load all NBA players for ID lookup ---
    nba_players = players.get_players()
    player_id_map = {player['full_name']: player['id'] for player in nba_players}

    # Combine start year and NBA ID
    player_data = player_min_years.copy()
    player_data['Player_ID'] = player_data['PLAYER_NAME'].map(player_id_map)

    # Identify players who could not be matched
    unmatched_players = player_data[player_data['Player_ID'].isna()]
    if not unmatched_players.empty:
        print(f"Warning: Could not find NBA ID for {len(unmatched_players)} player(s).")
        print(f"Skipping: {list(unmatched_players['PLAYER_NAME'])}")
        player_data = player_data.dropna(subset=['Player_ID']).copy()

    player_data['Player_ID'] = player_data['Player_ID'].astype(int)

    print(f"Found {len(player_data)} players to process (up to {end_season_str}).")
    print("-" * 50)

    # --- Step 2: Fetching Stats from NBA API ---
    all_stats_dfs = []

    for index, row in player_data.iterrows():
        player_name = row['PLAYER_NAME']
        player_id = row['Player_ID']
        start_year_int = row['Start_Year_Actual']  # Use the adjusted start year (2020 or later)

        print(
            f"Processing ({index + 1}/{len(player_data)}): {player_name} (ID: {player_id}) starting from {start_year_int}-21 season.")

        # --- Generate seasons list for this player ---
        seasons_to_fetch = [
            f"{year}-{str(year + 1)[2:]}"
            for year in range(start_year_int, end_query_year + 1)
        ]

        try:
            # PlayerCareerStats fetches ALL regular season totals for ALL seasons.
            career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)

            # The 'CareerTotalsRegularSeason' table contains all season-by-season totals
            career_df = career_stats.get_data_frames()[0]

            # Filter the fetched data to include only the requested seasons
            career_df['In_Range'] = career_df['SEASON_ID'].apply(lambda x: x in seasons_to_fetch)
            career_df = career_df[career_df['In_Range']].drop(columns=['In_Range'])

            # Add player name and ID for clarity
            career_df['PLAYER_NAME'] = player_name
            career_df['PLAYER_ID'] = player_id

            all_stats_dfs.append(career_df)

            max_fetched_season = career_df['SEASON_ID'].max() if not career_df.empty else 'N/A'
            print(f"  -> Fetched {len(career_df)} seasons (up to {max_fetched_season})")

        except Exception as e:
            print(f"  -> ERROR fetching data for {player_name}: {e}")

        # --- IMPORTANT: Sleep to respect API rate limits (1 second pause) ---
        time.sleep(1)

        # --- Step 3: Consolidate and Save Data ---
    if all_stats_dfs:
        final_df = pd.concat(all_stats_dfs, ignore_index=True)
        # Reorder columns for better readability
        cols = ['PLAYER_NAME', 'PLAYER_ID', 'SEASON_ID'] + [col for col in final_df.columns if
                                                            col not in ['PLAYER_NAME', 'PLAYER_ID', 'SEASON_ID']]
        final_df = final_df[cols]

        # Save the combined results to a new CSV file
        output_filename = f'nba_player_stats_from_{start_season_str}_to_{end_season_str}.csv'
        # final_df.to_csv(output_filename, index=False)
        print("-" * 50)
        print(
            f"Success! Fetched stats for {len(player_data)} players and saved {len(final_df)} season records to '{output_filename}'")
        print(f"Data starts at {start_season_str} or the player's true start season, whichever is later.")
        return final_df
    else:
        print("Failed to fetch any player data.")
        return pd.DataFrame()

def per_game_if_needed(series, gp_series):
    """If stats look like totals (big numbers), divide by GP."""
    if series.max() > 50:  # heuristic
        return series / gp_series
    return series


def add_position_to_existing_data(data_filepath):
    """
    Reads an existing dataset (which must contain a 'PLAYER_NAME' column),
    fetches the primary position for each player using the NBA API, and
    merges it into the dataset under the column name 'POS'.

    Args:
        data_filepath (str): Path to the existing CSV file to be updated.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    print(f"--- Starting POS data lookup for file: {data_filepath} ---")
    try:
        df_existing = pd.read_csv(data_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {data_filepath}")
        return pd.DataFrame()
    except KeyError:
        print(f"Error: The file {data_filepath} must contain a 'PLAYER_NAME' column.")
        return pd.DataFrame()

    # 1. Identify unique players from the existing data
    player_names = df_existing['PLAYER_NAME'].unique()
    print(f"Found {len(player_names)} unique players to look up.")

    # 2. Get NBA IDs for lookup
    nba_players = players.get_players()
    player_id_map = {player['full_name']: player['id'] for player in nba_players}

    # Create a temporary DataFrame to hold player lookup data
    lookup_df = pd.DataFrame(player_names, columns=['PLAYER_NAME'])
    lookup_df['Player_ID'] = lookup_df['PLAYER_NAME'].map(player_id_map)

    # Handle unmatched players
    unmatched_players = lookup_df[lookup_df['Player_ID'].isna()]
    if not unmatched_players.empty:
        print(f"Warning: Could not find NBA ID for {len(unmatched_players)} player(s). Skipping position fetch for them.")
        lookup_df = lookup_df.dropna(subset=['Player_ID']).copy()

    lookup_df['Player_ID'] = lookup_df['Player_ID'].astype(int)

    # 3. Fetch Player Position (POS)
    player_info_list = []
    print("--- Fetching Player Position (POS) from NBA API ---")

    for index, row in lookup_df.iterrows():
        player_id = row['Player_ID']
        player_name = row['PLAYER_NAME']

        try:
            # Query the CommonPlayerInfo endpoint
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = info.get_data_frames()[0]

            info_dict = {
                'Player_ID': player_id,
                'PLAYER_NAME': player_name,
                'POS': info_df.loc[0, 'POSITION'], # Using 'POS' as the final column name
            }
            player_info_list.append(info_dict)

        except Exception as e:
            print(f"  -> ERROR fetching info for {player_name}: {e}")

        # Short sleep to prevent rate limiting
        time.sleep(0.5)

    position_df = pd.DataFrame(player_info_list).drop(columns=['Player_ID'])

    # 4. Merge and Save
    print("--- Merging Position Data and Saving New File ---")

    # Merge the position data (POS) onto the existing DataFrame using PLAYER_NAME
    # We drop any pre-existing 'POS' or 'POSITION' columns to ensure a clean update
    df_merged = pd.merge(
        df_existing.drop(columns=['POS', 'POSITION', 'ROSTERSTATUS'], errors='ignore'),
        position_df,
        on='PLAYER_NAME',
        how='left'
    )

    # Prepare output filename
    output_filename = data_filepath.replace('.csv', '_with_POS.csv')

    # Handle the case where the existing file might already be the output file
    if output_filename == data_filepath:
        output_filename = data_filepath.replace('.csv', '_POS_only.csv')

    df_merged.to_csv(output_filename, index=False)

    print("-" * 50)
    print(f"Success! Added POS data and saved the new file to '{output_filename}'")
    print(f"Total rows in new file: {len(df_merged)}")
    return df_merged

def generate_profiles_per_season(df_general,season):
    df = df_general[df_general["SEASON_ID"] == season].copy()
    results = []
    # 1. Swiss Army Knife (all-rounders)
    all_rounders = df[(df[[f"z_{c}" for c in ["PTS","REB","AST","STL","BLK","FG3M","FG%","FT%"]]].gt(0).sum(axis=1) >= 6)]
    swiss_top10 = all_rounders.sort_values("fantasy_z_9cat", ascending=False).head(10)
    results.append(swiss_top10)

    # 2. Best assists-turnover ratio (z-score style)
    df["ast_tov_ratio"] = df["AST"] / df["TOV"].replace(0, np.nan)
    ast_tov_top10 = df.sort_values("ast_tov_ratio", ascending=False).head(10)
    results.append(ast_tov_top10)

    # 3. Punt FT specialists (reb + fg% + blk, but weak FT%)
    punt_ft = df[(df["z_FT%"] < -0.5)].copy()
    punt_ft["bigman_combo"] = df["z_REB"] + df["z_BLK"] + df["z_FG%"]
    punt_ft_top10 = punt_ft.sort_values("bigman_combo", ascending=False).head(10)
    results.append(punt_ft_top10)

    # 4. Defensive kings (steals + blocks)
    df["defense_combo"] = df["z_STL"] + df["z_BLK"]
    defense_top10 = df.sort_values("defense_combo", ascending=False).head(10)
    results.append(defense_top10)

    # 5. Best usage players (points + assists + rebounds)
    df["usage_combo"] = df["z_PTS"] + df["z_AST"] + df["z_REB"]
    usage_top10 = df.sort_values("usage_combo", ascending=False).head(10)
    results.append(usage_top10)

    # 6. Combo guard team (points + assists + 3pt + steals)
    df["combo_guard"] = df["z_PTS"] + df["z_AST"] + df["z_FG3M"] + df["z_STL"]
    guards_top10 = df.sort_values("combo_guard", ascending=False).head(10)
    results.append(guards_top10)

    return results


def evaluate_baseline(df, target_col="next_fantasy_z_9cat"):
    """
    Evaluate naive baseline: predict next season's fantasy value = last season's value.

    Parameters
    ----------
    df : pd.DataFrame
        Data with player_id, season, fantasy_z_9cat, and target_col.
    target_col : str, default="next_fantasy_z_9cat"
        Column representing the actual next-season fantasy value.

    Returns
    -------
    dict : baseline performance metrics
    """
    # Make sure data is sorted
    df = df.sort_values(["PLAYER_NAME", "SEASON_ID"])

    # Baseline = current season's fantasy value
    df["baseline_pred"] = df["fantasy_z_9cat"]

    # Keep rows where we actually have a next-season target
    valid = df.dropna(subset=[target_col, "baseline_pred"])

    y_true = valid[target_col]
    y_pred = valid["baseline_pred"]

    n = len(y_true)
    p = len(df.columns.tolist())

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R-squared": r2,
        "Adjusted R-squared": adjusted_r2,
    }


## Helper function for filling the 2021/22 season and rookies from other seasoms rolling averages and deltas with zero (if there is no past season
def preprocess_linear_features(df, season_col="SEASON_ID", player_col="PLAYER_NAME"):
    """
    Preprocess linear feature columns:
      1. Fill rolling/delta NaNs with -999 (benefit: When a linear model sees a −999 it learns a unique coefficient for all these
          "no prior history" cases, effectively treating it as a new, separate category.)
      2. Add has_history flag: 0 if first season for player, else 1

    Args:
        df (pd.DataFrame): input dataframe with rolling & delta features
        season_col (str): season column name
        player_col (str): player column name

    Returns:
        pd.DataFrame: cleaned dataframe ready for modeling
    """
    df_clean = df.copy()

    # 1. Identify rolling & delta columns
    rolling_cols = [c for c in df_clean.columns if "_roll2" in c]
    delta_cols = [c for c in df_clean.columns if "_delta" in c]

    # 2. Fill rolling & delta features with -999
    df_clean[rolling_cols + delta_cols] = df_clean[rolling_cols + delta_cols].fillna(-999)

    # 3. Add has_history flag (0 if all deltas are 0, else 1)
    df_clean["has_history"] = (df_clean[delta_cols].abs().sum(axis=1) > 0).astype(int)

    return df_clean


def preprocess_features(df, season_col="SEASON_ID", player_col="PLAYER_NAME"):
    """
    Preprocess linear feature columns:
      1. Adding per36 features.
      2. Creating sample weights for games played

    Args:
        df (pd.DataFrame): input dataframe with rolling & delta features
        season_col (str): season column name
        player_col (str): player column name

    Returns:
        pd.DataFrame: cleaned dataframe ready for modeling
    """
    df_clean = df.copy()

    # 1. Per36 stats (selected features only)
    per36_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    for col in per36_cols:
        df_clean[f"{col}_per36"] = df_clean[col] / df_clean["MIN"] * 36

    # 2. Adding sample-weight - Give more weight to examples where the season is more “reliable” (many games).
    df_clean["sample_weight"] = np.sqrt(df_clean["GP"].fillna(0))
    # Min-max normalize weights to [0.2,1] if you want to avoid extremely large weights:
    w = df_clean["sample_weight"].values
    w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    df_clean["sample_weight_norm"] = 0.2 + 0.8 * w
    df_clean.drop(columns=['sample_weight'], inplace=True)

    return df_clean

def evaluate(y_true, y_pred, p):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    n = len(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return {"MAE" :mae,"RMSE": rmse,"R-squared": r2, "Adjusted R-squared": adjusted_r2}

def plot_residaul_analysis(model, name, linear_X_test, y_test):
    best_model = model  # or PCA+LR
    y_pred = best_model.predict(linear_X_test)

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{name} - Residuals vs Predicted")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title(f"{name} - Distribution of Residuals")
    plt.show()

def plot_feature_importance(model, feature_names, title):
    features = model[0].get_feature_names_out()
    feat_imp = pd.Series(model[1].feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    feat_imp.head(15).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


def backward_elimination_ols_named(X_initial, y, feature_names=None, p_threshold=0.05):
    """
    Backward elimination OLS. Returns (ols_model, final_feature_list).
    X_initial: DataFrame or ndarray (rows must align with y)
    y: Series or ndarray
    """
    # Prepare X DataFrame
    if isinstance(X_initial, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names required when X_initial is ndarray")
        X = pd.DataFrame(X_initial.copy(), columns=feature_names)
    elif isinstance(X_initial, pd.DataFrame):
        X = X_initial.copy()
    else:
        raise TypeError("X_initial must be numpy array or DataFrame")

    # Ensure y is Series and aligned to X's index
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    y = y.reindex(X.index)

    # Iteratively remove worst p-value feature
    while True:
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()
        pvals = model.pvalues.drop('const', errors='ignore')
        if pvals.empty or pvals.max() <= p_threshold:
            break
        worst = pvals.idxmax()
        # If worst not in X (shouldn't happen) break
        if worst not in X.columns:
            break
        X = X.drop(columns=[worst])
        if X.shape[1] == 0:
            break

    final_features = list(X.columns)
    return model, final_features


def forecast_with_ols(ols_model, X_forecast_df, final_features):
    """
    ols_model: statsmodels fitted OLS
    X_forecast_df: DataFrame of preprocessor.get_feature_names_out() columns (any order)
    final_features: list of columns used during training (no 'const')
    Returns: pandas Series of predictions aligned to X_forecast_df.index
    """
    Xf = X_forecast_df.copy()
    # add missing training cols as zeros
    for c in final_features:
        if c not in Xf.columns:
            Xf[c] = 0.0
    # reduce to exactly final_features and cast to float
    Xf = Xf[final_features].astype(float)
    Xf_const = sm.add_constant(Xf, has_constant='add')
    preds = ols_model.predict(Xf_const)
    preds.index = Xf.index
    return preds


def train_models_per_target(train_df, test_df, raw_feature_columns, target_cols, preprocessor,
                            rf_params=None):
    """
    train_df/test_df: dataframes with rows and target cols present
    raw_feature_columns: columns used as input to preprocessor (raw features)
    target_cols: list of target columns to train e.g. ['z_PTS', 'z_REB', ..., 'fantasy_z_9cat']
    preprocessor: fitted ColumnTransformer/processor (must be fitted on training raw features)
    returns: dicts ols_models, rf_models, and performance summary
    """
    rf_params = rf_params or {"n_estimators": 300, "max_depth": 10, "random_state": 42}
    ols_models = {}
    rf_models = {}
    pca_lr_models = {}
    perf = []

    # Transform training and test raw features to named DataFrames
    X_train_arr = preprocessor.transform(train_df[raw_feature_columns])
    feat_names = list(preprocessor.get_feature_names_out())
    X_train_df = pd.DataFrame(X_train_arr, columns=feat_names, index=train_df.index)

    X_test_arr = preprocessor.transform(test_df[raw_feature_columns])
    X_test_df = pd.DataFrame(X_test_arr, columns=feat_names, index=test_df.index)

    # For PCA we will standardize inside PCA pipeline; here we use raw arrays
    for target in target_cols:
        # prepare y aligned
        y_train = train_df[target].reindex(X_train_df.index).copy()
        y_test = test_df[target].reindex(X_test_df.index).copy()

        # 1) OLS + backward elimination
        try:
            ols_model, final_features = backward_elimination_ols_named(X_train_df, y_train, feat_names,
                                                                       p_threshold=0.05)
            # prepare X_test subset for prediction
            X_test_for_ols = X_test_df.reindex(columns=final_features).astype(float).fillna(0.0)
            X_test_for_ols_const = sm.add_constant(X_test_for_ols, has_constant='add')
            y_pred_ols = ols_model.predict(X_test_for_ols_const)
            r2_ols = r2_score(y_test, y_pred_ols)
        except Exception as e:
            print(f"OLS failed for {target}: {e}")
            ols_model, final_features, r2_ols, y_pred_ols = None, [], np.nan, np.full(len(X_test_df), np.nan)

        ols_models[target] = {"model": ols_model, "features": final_features}

        # 2) Random Forest
        try:
            rf = RandomForestRegressor(**rf_params)
            rf.fit(X_train_df, y_train)
            y_pred_rf = rf.predict(X_test_df)
            r2_rf = r2_score(y_test, y_pred_rf)
        except Exception as e:
            print(f"RF failed for {target}: {e}")
            rf, r2_rf, y_pred_rf = None, np.nan, np.full(len(X_test_df), np.nan)
        rf_models[target] = {"model": rf}

        perf.append({
            "target": target,
            "r2_ols": float(r2_ols) if np.isfinite(r2_ols) else np.nan,
            "r2_rf": float(r2_rf) if np.isfinite(r2_rf) else np.nan,
        })

    perf_df = pd.DataFrame(perf).set_index("target")
    return ols_models, rf_models, perf_df, feat_names


def forecast_targets_on_predict_only(predict_df, preprocessor, ols_models, rf_models, feat_names, raw_feature_columns, target_cols):
    """
    predict_df: predict_only dataframe (2024-25)
    preprocessor: fitted transformer
    ols_models, rf_models, pca_lr_models: outputs from train_models_per_target
    feat_names: names returned by preprocessor.get_feature_names_out()
    raw_feature_columns: raw columns before preprocessor
    target_cols: list of target names
    Returns predict_df with appended prediction columns
    """
    # Transform predict set
    X_fore_arr = preprocessor.transform(predict_df[raw_feature_columns])
    X_fore_df = pd.DataFrame(X_fore_arr, columns=feat_names, index=predict_df.index)

    # prepare result df
    res = predict_df.copy()

    for target in target_cols:
        # OLS
        ols_meta = ols_models.get(target, {})
        ols_model = ols_meta.get("model", None)
        final_features = ols_meta.get("features", [])
        if ols_model is not None and len(final_features)>0:
            try:
                preds_ols = forecast_with_ols(ols_model, X_fore_df, final_features)
                res[f"pred_OLS_{target}"] = preds_ols
            except Exception as e:
                print(f"Failed OLS forecast for {target}: {e}")
                res[f"pred_OLS_{target}"] = np.nan
        else:
            res[f"pred_OLS_{target}"] = np.nan
        # RF
        rf_meta = rf_models.get(target, {})
        rf_model = rf_meta.get("model", None)
        if rf_model is not None:
            try:
                res[f"pred_RF_{target}"] = rf_model.predict(X_fore_df)
            except Exception as e:
                print(f"Failed RF forecast for {target}: {e}")
                res[f"pred_RF_{target}"] = np.nan
        else:
            res[f"pred_RF_{target}"] = np.nan

    return res


