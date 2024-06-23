import streamlit as my_st
import numpy as my_np
import pandas as my_pd
import joblib as my_joblib
import xgboost as my_xgb

# Load the trained model
model_filename = 'xgb_model.pkl'
model = my_joblib.load(model_filename)

# Function to make predictions
def predict_player_rating(features):
    features = my_np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Define tooltips
tooltips = {
    "Dribbling": "Technical skill when controlling the ball with feet.",
    "Potential": "Projected future skill level of the player.",
    "Mentality Composure": "Ability to remain calm and make good decisions under pressure.",
    "Passing": "Accuracy and effectiveness in passing the ball.",
    "Wage (EUR)": "Weekly salary of the player in Euros.",
    "Overall Mentality Rating": "Overall rating of mental attributes.",
    "Attacking Short Passing": "Skill in making short passes in attacking situations.",
    "Value (EUR)": "Estimated market value of the player in Euros.",
    "Movement Reactions": "Speed and efficiency of reacting to game situations."
}


# Streamlit app
def main():
    my_st.title("Player Rating Prediction")
    my_st.write("Enter player features to predict the overall rating.")

    # Input fields for player features
    dribbling = my_st.number_input("Dribbling", min_value=0, max_value=100, value=60 , help=tooltips["Dribbling"])
    potential = my_st.number_input("Potential", min_value=0, max_value=100, value=60, help=tooltips["Potential"])
    mentality_composure = my_st.number_input("Mentality Composure", min_value=0, max_value=100, value=60, help=tooltips["Mentality Composure"])
    passing = my_st.number_input("Passing", min_value=0, max_value=100, value=50,help=tooltips["Passing"])
    wage_eur = my_st.number_input("Wage (EUR)", min_value=0, value=1000, help=tooltips["Wage (EUR)"])
    overall_mentality_rating = my_st.number_input("Overall Mentality Rating", min_value=0, max_value=100, value=60, help=tooltips["Overall Mentality Rating"])
    attacking_short_passing = my_st.number_input("Attacking Short Passing", min_value=0, max_value=100, value=60, help=tooltips["Attacking Short Passing"])
    value_eur = my_st.number_input("Value (EUR)", min_value=0, value=1000, help=tooltips["Value (EUR)"])
    movement_reactions = my_st.number_input("Movement Reactions", min_value=0, max_value=100, value=60,help=tooltips["Movement Reactions"] )

    # Button to make predictions
    if my_st.button("Predict Rating"):
        features = [dribbling, potential, mentality_composure, passing, wage_eur,
                    overall_mentality_rating, attacking_short_passing, value_eur, movement_reactions]
        prediction = predict_player_rating(features)
        my_st.write(f"Predicted Player Rating: {prediction:.2f}")

if __name__ == "__main__":
    main()
