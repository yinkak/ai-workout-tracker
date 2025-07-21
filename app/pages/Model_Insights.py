# pages/03_Model_Insights.py

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from src.utils import get_gsheet_client
from datetime import timedelta

# --- Google Sheet Configuration ---
TRANSFORMED_GSHEET_URL_KEY = "transformed_google_sheet" 
TRANSFORMED_GSHEET_TAB_NAME = "Processed Data"
MODEL_DIR_RELATIVE_TO_APP = os.path.join(os.path.dirname(__file__),"..", "..", "models")


@st.cache_resource
def load_ml_resources_for_insights():
    try:
        # Load the regressor and model features
        regressor = joblib.load(os.path.join(MODEL_DIR_RELATIVE_TO_APP, 'trained_regressor_model.joblib'))
        model_features = joblib.load(os.path.join(MODEL_DIR_RELATIVE_TO_APP, 'model_features.joblib'))

        # Load transformed data from Google Sheet
        gc = get_gsheet_client()
        sheet_url = st.secrets[TRANSFORMED_GSHEET_URL_KEY]["url"]
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet(TRANSFORMED_GSHEET_TAB_NAME)
        data = worksheet.get_all_records()
        df_history_transformed = pd.DataFrame(data)

        # Basic data cleaning/type conversion for the loaded DataFrame
        if not df_history_transformed.empty:
            if 'date' in df_history_transformed.columns:
                df_history_transformed['date'] = pd.to_datetime(df_history_transformed['date'], errors='coerce')
                df_history_transformed.dropna(subset=['date'], inplace=True) # Drop rows where date couldn't be parsed

            numeric_cols = ['weight_lbs', 'sets', 'reps', 'rpe', 'volume', 'target_reps',
                            'reps_over_target', 'ready_for_increase', 'next_weight_lbs', 'exercise_encoded',
                            'rir', 'reps_x_rpe'] # Add other features if they exist
            for col in numeric_cols:
                if col in df_history_transformed.columns:
                    df_history_transformed[col] = pd.to_numeric(df_history_transformed[col], errors='coerce')
            df_history_transformed.dropna(subset=['weight_lbs', 'reps', 'rpe', 'next_weight_lbs'], inplace=True)
            df_history_transformed.columns = df_history_transformed.columns.str.strip()

        return regressor, model_features, df_history_transformed

    except FileNotFoundError:
        st.error(f"ML resources not found! Expected files in {os.path.abspath(MODEL_DIR_RELATIVE_TO_APP)}. "
                 "Please ensure 'train_model.py' has been run and models are saved.")
        st.stop()
    except KeyError as e:
        st.error(f"Google Sheet URL not found in secrets (key: '{e}'). Please check your `secrets.toml` configuration.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data for model insights: {e}")
        st.stop()

def show_model_insights_page():
    st.title("🧠 How the AI Coach Works")

    # Load resources
    regressor, model_features, df_history_transformed = load_ml_resources_for_insights()

    st.markdown("""
    This application uses a **Random Forest Regressor** machine learning model to predict your next optimal workout weight.
    It learns from your past performance (weight, reps, RPE, and how those change over time) to make personalized recommendations.
    """)

    st.subheader("What the Model Considers Important (Feature Importances)")
    if hasattr(regressor, 'feature_importances_') and model_features:
        importances = regressor.feature_importances_
        feature_names = model_features

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis')
        ax.set_title('Feature Importances from Random Forest Model')
        ax.set_xlabel('Relative Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Feature importances not available. Ensure the model is trained correctly.")

    st.subheader("Model Performance: Actual vs. Predicted Weights")

    if not df_history_transformed.empty and 'next_weight_lbs' in df_history_transformed.columns:
        X_predict_processed = df_history_transformed[model_features]
        y_actual = df_history_transformed['next_weight_lbs']
        
        for col in X_predict_processed.columns:
            X_predict_processed[col] = pd.to_numeric(X_predict_processed[col], errors='coerce').fillna(0)


        try:
            y_predicted = regressor.predict(X_predict_processed)

            fig_ap, ax_ap = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_actual, y=y_predicted, ax=ax_ap, alpha=0.6)
            ax_ap.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
            ax_ap.set_title('Actual vs. Predicted Next Weight')
            ax_ap.set_xlabel('Actual Next Weight (lbs)')
            ax_ap.set_ylabel('Predicted Next Weight (lbs)')
            st.pyplot(fig_ap)
            plt.close(fig_ap)

            residuals = y_actual - y_predicted
            fig_res, ax_res = plt.subplots(figsize=(8, 4))
            sns.scatterplot(x=y_predicted, y=residuals, ax=ax_res, alpha=0.6)
            ax_res.axhline(y=0, color='r', linestyle='--')
            ax_res.set_title('Residuals Plot (Prediction Error)')
            ax_res.set_xlabel('Predicted Next Weight (lbs)')
            ax_res.set_ylabel('Residuals (Actual - Predicted)')
            st.pyplot(fig_res)
            plt.close(fig_res)

            st.markdown(f"""
            * **Mean Absolute Error (MAE):** {abs(residuals).mean():.2f} lbs
            * **Root Mean Squared Error (RMSE):** {(residuals**2).mean()**0.5:.2f} lbs
            """)

        except Exception as e:
            st.warning(f"Could not generate performance plots. Error: {e}")
            st.info("Ensure your 'model_features.joblib' is correctly saved and aligns with your data's column names, and that your transformed Google Sheet has all required columns.")
    else:
        st.info("No transformed history data available (or 'next_weight_lbs' column missing) to generate model performance plots. Please ensure your data is processed and available in the Google Sheet.")


    st.subheader("Simplified Decision Flow")
    st.markdown("""
    Beyond the complex machine learning, the AI Coach also applies common-sense rules to refine its recommendations.
    This ensures the advice is practical and aligns with safe, effective training principles:

    * **If your last session was easy (RPE 5-7) and you hit good reps:** The model will likely recommend an **increase in weight**.
    * **If your last session was challenging (RPE 8-9) but you still hit your reps:** The model might suggest **maintaining the weight** to build strength and volume.
    * **If you truly struggled (RPE 9-10) and reps dropped significantly:** The model might suggest a **slight decrease (deload)** to allow for recovery and reset for future progress.

    This combination of data-driven insights and practical rules helps you progressively overload effectively.
    """)

# Main entry point for this page
if __name__ == '__main__':
    show_model_insights_page()