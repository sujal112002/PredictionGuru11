import pandas as pd


SESSION_GAP_MINUTES = 30


def _add_session_ids(events_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create session ids when user inactivity crosses the session threshold."""
    dataframe = events_dataframe.copy()
    dataframe["previous_event_time"] = dataframe.groupby("visitorid")["timestamp"].shift(1)
    gap_minutes = (dataframe["timestamp"] - dataframe["previous_event_time"]).dt.total_seconds().div(60)

    # A 30-minute inactivity threshold is common in web analytics.
    starts_new_session = gap_minutes.isna() | (gap_minutes > SESSION_GAP_MINUTES)
    dataframe["session_id"] = starts_new_session.groupby(dataframe["visitorid"]).cumsum().astype("int64")
    return dataframe.drop(columns=["previous_event_time"])


def _keep_pre_conversion_events(events_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only behavior a model could observe before the outcome happens.

    Converters keep events strictly before their first transaction; non-converters keep
    everything. Transaction rows are always removed so no feature can encode the target.
    """
    first_transaction_at = (
        events_dataframe.loc[events_dataframe["event"] == "transaction"]
        .groupby("visitorid")["timestamp"]
        .min()
    )
    cutoff = events_dataframe["visitorid"].map(first_transaction_at)
    before_cutoff = cutoff.isna() | (events_dataframe["timestamp"] < cutoff)
    is_behavior_event = events_dataframe["event"] != "transaction"
    return events_dataframe.loc[before_cutoff & is_behavior_event].reset_index(drop=True)


def create_user_features(clean_events_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Build user-level features from cleaned event logs.
    """
    converted_users = clean_events_dataframe.loc[
        clean_events_dataframe["event"] == "transaction", "visitorid"
    ].unique()

    # Users who purchase with no prior behavior drop out here: there is nothing to predict from.
    dataframe = _add_session_ids(_keep_pre_conversion_events(clean_events_dataframe))

    user_event_counts = dataframe.groupby("visitorid").agg(
        total_events=("event", "size"),
        total_views=("event", lambda values: (values == "view").sum()),
        total_addtocart=("event", lambda values: (values == "addtocart").sum()),
        first_activity_at=("timestamp", "min"),
        last_activity_at=("timestamp", "max"),
        unique_items_viewed=("itemid", "nunique"),
    )

    session_summary = dataframe.groupby(["visitorid", "session_id"]).agg(
        session_start=("timestamp", "min"),
        session_end=("timestamp", "max"),
        events_in_session=("event", "size"),
    )
    session_summary["session_duration_minutes"] = (
        session_summary["session_end"] - session_summary["session_start"]
    ).dt.total_seconds().div(60)

    user_session_features = session_summary.groupby("visitorid").agg(
        total_sessions=("events_in_session", "size"),
        avg_session_duration_minutes=("session_duration_minutes", "mean"),
        median_session_duration_minutes=("session_duration_minutes", "median"),
        avg_events_per_session=("events_in_session", "mean"),
    )

    feature_dataframe = user_event_counts.join(user_session_features, how="left").reset_index()

    feature_dataframe["click_rate"] = (
        feature_dataframe["total_addtocart"] / feature_dataframe["total_events"].clip(lower=1)
    )
    feature_dataframe["add_to_view_ratio"] = (
        feature_dataframe["total_addtocart"] / feature_dataframe["total_views"].clip(lower=1)
    )
    feature_dataframe["session_duration"] = feature_dataframe["avg_session_duration_minutes"].fillna(0.0)
    feature_dataframe["pages_per_session"] = (
        feature_dataframe["unique_items_viewed"] / feature_dataframe["total_sessions"].clip(lower=1)
    )
    feature_dataframe["converted"] = feature_dataframe["visitorid"].isin(converted_users).astype(int)
    feature_dataframe["active_days"] = (
        feature_dataframe["last_activity_at"] - feature_dataframe["first_activity_at"]
    ).dt.total_seconds().div(86400)
    feature_dataframe["cart_to_view_rate"] = (
        feature_dataframe["total_addtocart"] / feature_dataframe["total_views"].clip(lower=1)
    )

    feature_dataframe = feature_dataframe.fillna(0.0)
    return feature_dataframe


def get_model_feature_columns() -> list[str]:
    """Features used by both training and inference."""
    return [
        "total_events",
        "total_views",
        "total_addtocart",
        "add_to_view_ratio",
        "total_sessions",
        "click_rate",
        "session_duration",
        "pages_per_session",
        "active_days",
        "cart_to_view_rate",
        "avg_events_per_session",
    ]