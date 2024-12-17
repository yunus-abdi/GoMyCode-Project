import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
clf = joblib.load('road_risk_model.pkl')

# Function to classify danger based on user input
def classify_road_danger(victim_input, deaths_input, accident_time_input):
    # Prepare the input data in the format expected by the model
    user_input = np.array([[victim_input, deaths_input, accident_time_input]])
    
    # Make the prediction
    prediction = clf.predict(user_input)[0]

    # Map the model's prediction (0, 1, 2) to road danger labels
    if prediction == "Low":
        return "Low", "green"
    elif prediction == "Moderate":
        return "Moderate", "yellow"
    elif prediction == "Moderately High":
        return "Moderately High", "orange"
    elif prediction == "High":
        return "High", "red"
    else:
        return "Unknown", "gray"

# Streamlit app layout
st.title("Road Risk Classification")
st.write("Enter the information to classify the road risk.")

# Select victim category
victim_category = st.selectbox(
    "Victim Category",
    [
        "driver and other motorists", "driver and passengers", "drivers, passengers and pedestrians", 
        "motorist", "passenger", "passengers", "passengers and drivers", "passengers and pedestrians", 
        "pedestrian", "pedestrians"
    ]
)

# Map victim category to index (0-9)
victim_input = ["driver and other motorists", "driver and passengers", "drivers, passengers and pedestrians", 
                "motorist", "passenger", "passengers", "passengers and drivers", "passengers and pedestrians", 
                "pedestrian", "pedestrians"].index(victim_category)

# Input for number of deaths
deaths_input = st.number_input("Enter number of deaths confirmed:", min_value=0, step=1)

# Select accident time
accident_time = st.selectbox(
    "Accident Time",
    ["afternoon", "evening", "morning", "night"]
)

# Map accident time to index (0-3)
accident_time_input = ["afternoon", "evening", "morning", "night"].index(accident_time)

# Error handling for specific victim categories with deaths greater than 1
if victim_category in ["passenger", "pedestrian", "motorist"] and deaths_input > 1:
    st.markdown("<h3 style='color:red;'>Cannot have deaths greater than 1 for singular victims</h3>", unsafe_allow_html=True)
else:
    # When the user submits the data
    if st.button("Classify Road Risk"):
        try:
            # Call function to classify the road risk
            prediction_label, color = classify_road_danger(victim_input, deaths_input, accident_time_input)
            
            # Display the result with color
            st.markdown(f"<h2 style='color:{color};'>The road is classified as: {prediction_label}</h2>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}. Please try again.")
