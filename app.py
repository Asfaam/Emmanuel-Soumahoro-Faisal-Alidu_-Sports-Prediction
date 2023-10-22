import pickle
import streamlit as st
import pandas as pd

# Load the trained model
model_filename = 'player_rating_predictor.pkl'
model = pickle.load(open(model_filename, 'rb'))

# Define a function for predicting player ratings
def predict_player_rating(input_data):
    input_data = input_data.values.reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Create a Streamlit web application
st.title("FIFA Player Rating Predictor")
st.write("Enter player information to predict their overall rating.")

# Create input fields for user data
features = ['potential', 'age', 'value_eur', 'wage_eur', 'international_reputation', 'weak_foot', 'skill_moves', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision']
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=0)

# Create a button to trigger the prediction
if st.button("Predict Rating"):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = predict_player_rating(input_df)
    st.write(f"Predicted Player Rating: {prediction:.2f}")
