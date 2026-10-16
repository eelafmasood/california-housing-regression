from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏡",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "california_housing_model.joblib"
RESULTS_PATH = Path(__file__).parent / "cross_validation_results.csv"


@st.cache_resource(show_spinner="Loading the trained model...")
def load_artifact(path_string: str):
    return joblib.load(path_string)


st.title("🏡 California Housing Value Predictor")
st.write(
    "Estimate a California district's median house value using a trained machine-learning model. "
    "This is an educational estimate based on historical census data, not a current property appraisal."
)

if not MODEL_PATH.exists():
    st.error("The model file is missing. Add california_housing_model.joblib to the project repository.")
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
    model = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    model_name = artifact.get("model_name", "Trained regression model")
except Exception as exc:
    st.error("The saved model could not be loaded. Check that scikit-learn matches the version used to save it.")
    st.code(str(exc))
    st.stop()

st.caption(f"Model loaded: **{model_name}**")

with st.form("prediction_form"):
    st.subheader("Enter district characteristics")
    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-122.23, step=0.01, format="%.2f")
        latitude = st.number_input("Latitude", min_value=32.0, max_value=43.0, value=37.88, step=0.01, format="%.2f")
        housing_median_age = st.number_input("Housing median age (years)", min_value=1, max_value=100, value=25, step=1)

    with col2:
        total_rooms = st.number_input("Total rooms", min_value=1, value=2000, step=100)
        total_bedrooms = st.number_input("Total bedrooms", min_value=1, value=400, step=25)
        population = st.number_input("Population", min_value=1, value=1000, step=50)

    with col3:
        households = st.number_input("Households", min_value=1, value=350, step=25)
        median_income = st.number_input("Median income (dataset units)", min_value=0.0, max_value=20.0, value=3.5, step=0.1, format="%.2f")
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
            index=3,
        )

    
    # Validate the input values
    invalid_bedrooms = total_bedrooms > total_rooms
    invalid_households = households > population

    if invalid_bedrooms:
        st.error("Total bedrooms cannot exceed total rooms.")

    if invalid_households:
        st.error("Households cannot exceed population.")

    submitted = st.form_submit_button(
        "Predict Median House Value",
        type="primary",
        use_container_width=True,
    )


if submitted and not invalid_bedrooms and not invalid_households:
    input_row = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    try:
        input_df = pd.DataFrame([{column: input_row[column] for column in feature_columns}])
        prediction = float(model.predict(input_df)[0])
        st.success("Prediction completed")
        st.metric("Estimated median house value", f"${prediction:,.0f}")
        st.caption("The result is a model estimate; actual market prices may differ.")
    except Exception as exc:
        st.error("Prediction failed. Please check the input values and saved model.")
        st.code(str(exc))

st.divider()
st.subheader("10-fold cross-validation: model comparison")
if RESULTS_PATH.exists():
    try:
        results = pd.read_csv(RESULTS_PATH)
        st.dataframe(results, use_container_width=True, hide_index=True)
        chart_col1, chart_col2 = st.columns(2)
        if "Model" in results.columns and "Mean R²" in results.columns:
            with chart_col1:
                st.write("**Mean R² (higher is better)**")
                st.bar_chart(results.set_index("Model")["Mean R²"])
        if "Model" in results.columns and "Mean RMSE" in results.columns:
            with chart_col2:
                st.write("**Mean RMSE (lower is better)**")
                st.bar_chart(results.set_index("Model")["Mean RMSE"])
    except Exception as exc:
        st.warning("Could not display the comparison CSV.")
        st.code(str(exc))
else:
    st.info("Add cross_validation_results.csv to display the model comparison table and charts.")

st.caption("Course project: Machine Learning Fundamentals (AIC354) | California Housing Prices dataset")
