import streamlit as st
from user_interface import (
    display_intro,
    user_input_form,
    preprocess_input,
    make_prediction
)
import joblib

# Load the trained model
def load_model():
    """
    Load the pre-trained Logistic Regression model.
    """
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():

    display_intro()

    model = load_model()

    if model:
        # Get user input
        user_input = user_input_form()

        # Preprocess the input data
        preprocessed_input = preprocess_input(user_input)

        # Display the input data
        st.write("### Employee Data")
        st.write(user_input)

        # Predict when the user clicks the button
        if st.button("Predict Attrition"):
            make_prediction(model, preprocessed_input)

if __name__ == "__main__":
    main()