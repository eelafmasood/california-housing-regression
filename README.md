# California Housing Regression — Streamlit Deployment

A Streamlit web application for estimating median California district house values using a trained regression pipeline. The app also displays the five models' 10-fold cross-validation results and charts.

## Files required to run the app

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `california_housing_model.joblib` — trained pipeline and feature metadata
- `cross_validation_results.csv` — cross-validation comparison table (optional for predictions, required for comparison charts)

The notebook and `housing.csv` should also be retained in the full assignment repository as supporting evidence, but this app does not need to read them to make predictions.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Use a compatible Python environment and the same scikit-learn version used to create the serialized model. This project pins scikit-learn to `1.6.1`; if the model was saved with a different version, adjust the pin to the training version and test before deployment.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Ensure `app.py` and `requirements.txt` are in the repository root.
3. Ensure `california_housing_model.joblib` is available to the app. Because it is large, use Git LFS rather than GitHub's browser uploader or a normal Git commit.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Create an app using your repository, branch `main`, and main file path `app.py`.
6. Test a prediction and verify the cross-validation table and charts.

The prediction is an educational estimate based on historical census data, not a current real-estate appraisal.
