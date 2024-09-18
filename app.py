import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PROCESSED_DATA_PATH = 'mlb_processed_injury_data.pkl'
MODEL_PATH = 'mlb_injury_model.joblib'
SCALER_PATH = 'mlb_injury_scaler.joblib'

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            return pd.read_pickle(PROCESSED_DATA_PATH)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.error(f"Processed data not found at {PROCESSED_DATA_PATH}")
        return None

@st.cache_resource
def load_model() -> Tuple[Optional[object], Optional[object]]:
    model, scaler = None, None
    for path, obj_name in [(MODEL_PATH, "model"), (SCALER_PATH, "scaler")]:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if obj_name == "model":
                    model = obj
                else:
                    scaler = obj
            except Exception as e:
                st.error(f"Error loading {obj_name}: {e}")
        else:
            st.error(f"{obj_name.capitalize()} not found at {path}")
    return model, scaler

def predict_injury_risk(model, scaler, df_processed, input_data):
    if model is None or scaler is None:
        st.error("Model or scaler not available. Unable to make prediction.")
        return None, None

    new_data = pd.DataFrame([input_data])
    feature_names = df_processed.drop(['Injured', 'player_name'], axis=1).columns
    new_data = new_data.reindex(columns=feature_names, fill_value=0)
    
    # Handle the case where 'Position_Pitcher' is not in the columns
    if 'Position_Pitcher' not in new_data.columns and 'Position' in new_data.columns:
        new_data['Position_Pitcher'] = (new_data['Position'] == 'Pitcher').astype(int)
        new_data = new_data.drop('Position', axis=1)
    
    X_new_scaled = scaler.transform(new_data)
    risk_probability = model.predict_proba(X_new_scaled)[0][1]
    risk_prediction = model.predict(X_new_scaled)[0]
    return risk_prediction, risk_probability

def predict_injury_risk_for_all(model, scaler, df_processed):
    if model is None or scaler is None:
        st.error("Model or scaler not available. Unable to make predictions.")
        return None

    X = df_processed.drop(['Injured', 'player_name'], axis=1)
    X_scaled = scaler.transform(X)
    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)
    return pd.DataFrame({
        'player_name': df_processed['player_name'],
        'Injury_Risk_Prediction': y_pred,
        'Injury_Risk_Probability': y_probs
    })

def plot_feature_importance(model, feature_names, top_n=15):
    if model is None:
        st.error("Model not available. Unable to plot feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances')
    st.pyplot(fig)

def display_player_stats(player_data):
    st.subheader("Player Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Age:** {player_data['Age'].iloc[0] if 'Age' in player_data.columns else 'N/A'}")
        st.markdown(f"**Games:** {player_data['G'].iloc[0] if 'G' in player_data.columns else 'N/A'}")
        
        # Determine position based on available statistics
        if 'ERA' in player_data.columns and player_data['ERA'].iloc[0] > 0:
            position = 'Pitcher'
        elif 'BA' in player_data.columns and player_data['BA'].iloc[0] > 0:
            position = 'Position Player'
        else:
            position = 'Unknown'
        st.markdown(f"**Position:** {position}")
    
    with col2:
        if position == 'Pitcher':
            st.markdown("**Pitching Performance:**")
            st.metric("ERA", f"{player_data['ERA'].iloc[0]:.2f}" if 'ERA' in player_data.columns else 'N/A')
            st.metric("WHIP", f"{player_data['WHIP'].iloc[0]:.3f}" if 'WHIP' in player_data.columns else 'N/A')
            st.metric("SO9", f"{player_data['SO9'].iloc[0]:.2f}" if 'SO9' in player_data.columns else 'N/A')
        else:
            st.markdown("**Batting Performance:**")
            st.metric("BA", f"{player_data['BA'].iloc[0]:.3f}" if 'BA' in player_data.columns else 'N/A')
            st.metric("OPS", f"{player_data['OPS'].iloc[0]:.3f}" if 'OPS' in player_data.columns else 'N/A')
            st.metric("WAR", f"{player_data['WAR'].iloc[0]:.2f}" if 'WAR' in player_data.columns else 'N/A')


def get_injury_prevention_recommendation(injury_risk_pred):
    if injury_risk_pred == 1:
        return ("Injury Prevention Recommendation: "
                "Implement a tailored injury prevention program focusing on high-risk areas. "
                "This may include specialized strength and conditioning exercises, "
                "biomechanical analysis to identify and correct potential issues, "
                "and a carefully managed workload to prevent overuse injuries.")
    else:
        return ("Injury Prevention Recommendation: "
                "Continue with the current training regimen while monitoring for any signs of fatigue or discomfort. "
                "Regular check-ins with the medical staff, proper nutrition, and adequate rest "
                "will help maintain the player's low injury risk status.")

def main():
    st.set_page_config(page_title="MLB Injury Risk Prediction App", page_icon="⚾")
    st.title("MLB Injury Risk Prediction App ⚾")
    
    df_processed = load_data()
    model, scaler = load_model()
    
    if df_processed is None:
        st.warning("Data not available. Some features may be limited.")
    if model is None or scaler is None:
        st.warning("Model or scaler not available. Predictions cannot be made.")
    
    menu = ["Home", "Predict Injury Risk", "Player Risk Lookup", "Data Visualization", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Welcome to the MLB Injury Risk Prediction App")
        st.write("""
        This app uses machine learning to predict the injury risk for MLB players based on their statistics. 
        Use the sidebar to navigate through different features of the app:
        
        - **Predict Injury Risk**: Input player stats to get an injury risk prediction
        - **Player Risk Lookup**: Select a player to see their injury risk, stats, and prevention recommendations
        - **Data Visualization**: Explore feature importance and SHAP values
        - **About**: Learn more about how this app works
        
        Get started by selecting an option from the sidebar!
        """)
    
    elif choice == "Predict Injury Risk":
        st.subheader("Predict Injury Risk for a Player")
        
        if df_processed is None or model is None or scaler is None:
            st.error("Required components are missing. Unable to make predictions.")
            return

        with st.form("player_form"):
            player_name = st.text_input("Player Name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=50, value=25)
                games = st.number_input("Games Played", min_value=1, max_value=162, value=100)
                position = st.selectbox("Position", ["Pitcher", "Position Player"])
            with col2:
                if position == "Pitcher":
                    era = st.number_input("ERA", min_value=0.0, value=4.0)
                    whip = st.number_input("WHIP", min_value=0.0, value=1.3)
                    so9 = st.number_input("SO9", min_value=0.0, value=8.0)
                else:
                    ba = st.number_input("Batting Average", min_value=0.0, max_value=1.0, value=0.250)
                    ops = st.number_input("OPS", min_value=0.0, max_value=2.0, value=0.750)
                    war = st.number_input("WAR", min_value=-5.0, max_value=15.0, value=2.0)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = {
                'player_name': player_name,
                'Age': age,
                'G': games,
                'Position': position,
                'ERA': era if position == "Pitcher" else 0,
                'WHIP': whip if position == "Pitcher" else 0,
                'SO9': so9 if position == "Pitcher" else 0,
                'BA': ba if position == "Position Player" else 0,
                'OPS': ops if position == "Position Player" else 0,
                'WAR': war if position == "Position Player" else 0
            }
            
            risk_prediction, risk_probability = predict_injury_risk(model, scaler, df_processed, input_data)
            if risk_prediction is not None and risk_probability is not None:
                risk_label = 'High Risk' if risk_prediction == 1 else 'Low Risk'
                st.markdown(f"### Injury Risk Prediction for {player_name}: {risk_label}")
                st.markdown(f"#### Risk Probability: {risk_probability:.2f}")
                
                recommendation = get_injury_prevention_recommendation(risk_prediction)
                st.markdown("---")
                st.markdown(f"### {recommendation}")
            else:
                st.error("Unable to make prediction. Please check the input data and try again.")
    
    elif choice == "Player Risk Lookup":
        st.subheader("MLB Players Injury Risk Lookup")
        if df_processed is None or model is None or scaler is None:
            st.error("Required components are missing. Unable to perform player risk lookup.")
            return

        df_results = predict_injury_risk_for_all(model, scaler, df_processed)
        if df_results is not None:
            player_names = df_processed['player_name'].unique()
            selected_player = st.selectbox("Select a Player", sorted(player_names))
            player_data = df_processed[df_processed['player_name'] == selected_player]
            if not player_data.empty:
                player_result = df_results[df_results['player_name'] == selected_player].iloc[0]
                injury_risk_prob = player_result['Injury_Risk_Probability']
                injury_risk_pred = player_result['Injury_Risk_Prediction']
                risk_label = 'High Risk' if injury_risk_pred == 1 else 'Low Risk'
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"<h2 style='font-size: 24px;'>{selected_player}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 20px;'>Injury Risk: <strong>{risk_label}</strong></h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 20px;'>Risk Probability: <strong>{injury_risk_prob:.2f}</strong></h3>", unsafe_allow_html=True)
                
                with col2:
                    display_player_stats(player_data)
                
                recommendation = get_injury_prevention_recommendation(injury_risk_pred)
                st.markdown("---")
                st.markdown(f"### {recommendation}")
            else:
                st.write("Player data not found.")
        else:
            st.error("Unable to generate risk predictions for players.")

    elif choice == "Data Visualization":
        st.subheader("Data Visualization")
        
        if df_processed is None or model is None or scaler is None:
            st.error("Required components are missing. Unable to generate visualizations.")
            return

        st.write("""
        This section provides insights into how our model makes predictions. We use two main visualization techniques:
        Feature Importance and SHAP (SHapley Additive exPlanations) values.
        """)
        
        st.subheader("Feature Importance")
        st.write("""
        Feature importance shows how much each feature contributes to the model's predictions. 
        Features with higher importance have a greater impact on the injury risk prediction.
        """)
        plot_feature_importance(model, df_processed.drop(['Injured', 'player_name'], axis=1, errors='ignore').columns, top_n=15)
        
        st.subheader("SHAP Summary Plot")
        st.write("""
        SHAP values provide a more detailed view of feature importance. They show how each feature 
        impacts the model output for each prediction. 
        
        - Features are ranked by importance from top to bottom.
        - Colors indicate whether the feature value is high (red) or low (blue) for that observation.
        - The horizontal location shows whether the effect of that value caused a higher or lower prediction.
        """)
        explainer = shap.TreeExplainer(model)
        
        # Prepare the data for SHAP
        X = df_processed.drop(['Injured', 'player_name'], axis=1, errors='ignore')
        X_sample = X.sample(min(100, len(X)))  # Sample up to 100 rows
        X_sample_scaled = scaler.transform(X_sample)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # Ensure shap_values is a list with at least two elements
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, X_sample, feature_names=X_sample.columns, plot_type="bar", show=False)
        st.pyplot(fig)
    
    elif choice == "About":
        st.subheader("About")
        st.write("""
        This app predicts the injury risk of MLB players based on historical data and player statistics. It uses a Random Forest Classifier 
        trained on player performance data and simulated injury data.

        Key features of the app include:
        1. Injury risk prediction for individual players
        2. Player risk lookup for all players in the database
        3. Data visualizations to understand feature importance and model predictions
        4. Injury prevention recommendations based on predicted risk

        Please note:
        - This model is based on simulated injury data. In a real-world scenario, you would need actual historical injury data for more accurate predictions.
        - The predictions should be used as one of many tools in assessing player health and should not replace medical expertise or individual player assessments.
        - Always consult with medical professionals and team experts for comprehensive player evaluations and injury prevention strategies.

        Data sources:
        - Player statistics: MLB API and Baseball Reference (via pybaseball)
        - Injury data: Simulated for demonstration purposes

        For more information on the data processing and model training, please refer to the `mlb_data_processor.py` script.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("An error occurred in the Streamlit app")