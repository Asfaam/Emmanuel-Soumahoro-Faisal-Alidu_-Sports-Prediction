import pickle
import streamlit as st
import pandas as pd

# Load the trained model

filename = 'C:/Users/user/OneDrive - Ashesi University/Desktop/ALL IN ONE/2ND YEAR, 2ND SEMESTER/Introduction to Artificial Intelligence/7. Week 9/AI_Final_MidSem_Project/player_rating_predictor.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Define a function for predicting player ratings
# In the predict_player_rating function, ensure the model is a RandomForestRegressor
def predict_player_rating(input_data):
    if isinstance(loaded_model, RandomForestRegressor):
        input_data = input_data.values.reshape(1, -1)
        prediction = loaded_model.predict(input_data)
        return prediction[0]
    else:
        st.error("Model is not a RandomForestRegressor")

# Create a Streamlit web application
st.title("FIFA Player Rating Predictor")
st.write("Enter player information to predict their overall rating.")

# Create input fields for user data
features = ['movement_reactions', 'mentality_composure', 'passing', 'potential', 'release_clause_eur', 'dribbling', 'wage_eur', 'power_shot_power', 'value_eur', 'mentality_vision', 'attacking_short_passing']
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=0)

# Create a button to trigger the prediction
if st.button("Predict Rating"):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = predict_player_rating(input_df)
    st.write(f"Predicted Player Rating: {prediction:.2f}")
