import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


def compute_derived_features(
    total_views: int,
    total_addtocart: int,
    total_sessions: int,
) -> dict:
    """Keep feature math consistent with src/features.py."""
    # Transactions are excluded from training features, so events are views plus add-to-carts.
    total_events = total_views + total_addtocart
    add_to_view_ratio = total_addtocart / max(total_views, 1)
    click_rate = total_addtocart / max(total_events, 1)
    cart_to_view_rate = add_to_view_ratio
    avg_events_per_session = total_events / max(total_sessions, 1)

    return {
        "total_events": total_events,
        "add_to_view_ratio": add_to_view_ratio,
        "click_rate": click_rate,
        "cart_to_view_rate": cart_to_view_rate,
        "avg_events_per_session": avg_events_per_session,
    }


def build_payload(
    total_views: int,
    total_addtocart: int,
    total_sessions: int,
    session_duration: float,
    pages_per_session: float,
    active_days: float,
) -> dict:
    derived_features = compute_derived_features(
        total_views=total_views,
        total_addtocart=total_addtocart,
        total_sessions=total_sessions,
    )

    return {
        "total_views": total_views,
        "total_addtocart": total_addtocart,
        "total_sessions": total_sessions,
        "session_duration": session_duration,
        "pages_per_session": pages_per_session,
        "active_days": active_days,
        **derived_features,
    }


def render_prediction_result(response_payload: dict) -> None:
    label = response_payload.get("label", "unknown")
    probability = float(response_payload.get("probability", 0.0))
    reason = response_payload.get("reason", "No explanation available.")

    if label == "convert":
        st.markdown(
            "<h3 style='color:#16a34a; margin-bottom:0;'>Prediction: convert</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h3 style='color:#dc2626; margin-bottom:0;'>Prediction: not_convert</h3>",
            unsafe_allow_html=True,
        )

    st.metric("Conversion Probability", f"{probability:.2%}")
    st.info(reason)


def main() -> None:
    st.set_page_config(page_title="User Conversion Prediction System", layout="centered")
    st.title("User Conversion Prediction System")
    st.caption("Predict conversion likelihood from user behavior signals.")

    st.subheader("User Behavior Inputs")
    left_column, right_column = st.columns(2)

    with left_column:
        total_views = st.slider("Total Views", min_value=0, max_value=100, value=15, step=1)
        total_addtocart = st.slider("Total Add To Cart", min_value=0, max_value=50, value=4, step=1)
        total_sessions = st.slider("Total Sessions", min_value=1, max_value=30, value=5, step=1)

    with right_column:
        session_duration = st.slider(
            "Session Duration (minutes)",
            min_value=0.0,
            max_value=120.0,
            value=12.0,
            step=0.5,
        )
        pages_per_session = st.slider(
            "Pages Per Session",
            min_value=0.0,
            max_value=30.0,
            value=4.0,
            step=0.1,
        )
        active_days = st.slider(
            "Active Days",
            min_value=0.0,
            max_value=365.0,
            value=14.0,
            step=1.0,
        )

    payload = build_payload(
        total_views=total_views,
        total_addtocart=total_addtocart,
        total_sessions=total_sessions,
        session_duration=session_duration,
        pages_per_session=pages_per_session,
        active_days=active_days,
    )

    with st.expander("Derived Features", expanded=False):
        st.json(
            {
                "total_events": payload["total_events"],
                "add_to_view_ratio": round(payload["add_to_view_ratio"], 4),
                "click_rate": round(payload["click_rate"], 4),
                "cart_to_view_rate": round(payload["cart_to_view_rate"], 4),
                "avg_events_per_session": round(payload["avg_events_per_session"], 4),
            }
        )

    st.divider()

    if st.button("Predict Conversion", type="primary", use_container_width=True):
        with st.spinner("Getting prediction from API..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                prediction_payload = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Ensure FastAPI is running at http://127.0.0.1:8000.")
                return
            except requests.exceptions.Timeout:
                st.error("API request timed out. Please try again.")
                return
            except requests.exceptions.RequestException as error:
                st.error(f"API request failed: {error}")
                return
            except ValueError:
                st.error("API returned an invalid response.")
                return

        st.subheader("Prediction Output")
        render_prediction_result(prediction_payload)


if __name__ == "__main__":
    main()
