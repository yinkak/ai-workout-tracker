# pages/03_Model_Insights.py

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Re-load resources for this page, or pass them if shared globally
# It's safer to re-load with caching on each page if memory is a concern,
# or ensure global singletons with st.cache_resource in app.py.
# For simplicity, let's re-load with caching here.

@st.cache_resource
def load_ml_resources_for_insights():
    MODEL_DIR = "../models"
    try:
        regressor = joblib.load(os.path.join(MODEL_DIR, 'trained_regressor_model.joblib'))
        model_features = joblib.load(os.path.join(MODEL_DIR, 'model_features.joblib'))
        df_history_raw = pd.read_csv("../data/transformed_workout_log.csv")
        df_history_raw['date'] = pd.to_datetime(df_history_raw['date'])
        return regressor, model_features, df_history_raw
    except FileNotFoundError:
        st.error("ML resources or transformed data not found for insights!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data for model insights: {e}")
        st.stop()

def show_model_insights_page():
    st.title("🧠 How the AI Coach Works")

    regressor, model_features, df_history_raw = load_ml_resources_for_insights()

    st.markdown("""
    This application uses a **Random Forest Regressor** machine learning model to predict your next optimal workout weight.
    It learns from your past performance (weight, reps, RPE, and how those change over time) to make personalized recommendations.
    """)

    st.subheader("What the Model Considers Important (Feature Importances)")
    # ... (Your existing feature importance plotting code) ...
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
    # ... (Your existing actual vs. predicted and residuals plotting code) ...
    # Ensure X_predict_processed is correctly built using model_features
    
    if not df_history_raw.empty:
        X_predict = df_history_raw.drop(columns=['next_weight_lbs', 'date', 'exercise', 'notes'], errors='ignore') # Ensure 'notes' is handled if it's there
        y_actual = df_history_raw['next_weight_lbs']

        X_predict_processed = pd.DataFrame(columns=model_features)
        for col in model_features:
            if col in X_predict.columns:
                X_predict_processed[col] = X_predict[col]
            else:
                X_predict_processed[col] = 0.0
        
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
            st.info("Ensure your 'model_features.joblib' is correctly saved and aligns with your data's column names.")
    else:
        st.info("No transformed history data available to generate model performance plots.")


    st.subheader("Simplified Decision Flow")
    # ... (Your existing simplified decision flow markdown) ...
    st.markdown("""
    Beyond the complex machine learning, the AI Coach also applies common-sense rules to refine its recommendations.
    This ensures the advice is practical and aligns with safe, effective training principles:

    * **If your last session was easy (RPE 5-7) and you hit good reps:** The model will likely recommend an **increase in weight**.
    * **If your last session was challenging (RPE 8-9) but you still hit your reps:** The model might suggest **maintaining the weight** to build strength and volume.
    * **If you truly struggled (RPE 9-10) and reps dropped significantly:** The model might suggest a **slight decrease (deload)** to allow for recovery and reset for future progress.

    This combination of data-driven insights and practical rules helps you progressively overload effectively.
    """)

# Main entry point for this page (optional, for direct testing)
if __name__ == '__main__':
    show_model_insights_page()