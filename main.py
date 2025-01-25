import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Smart Loan Application",page_icon='🏦')


CREDENTIALS = {
    "user": {"username": "1", "password": "1"},
    "admin": {"username": "2", "password": "2"},
}

# Initialize session states
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "user_history" not in st.session_state:
    # Initialize an empty DataFrame to store user requests history
    st.session_state["user_history"] = pd.DataFrame(columns=[
        "Customer Name", "Credit History", "Education", "Applicant Income", 
        "Coapplicant Income", "Loan Amount Term", "Property Area", "Gender", 
        "Loan Amount", "Marital Status", "Dependents", "Self-Employment", 
        "Loan Status", "Approval Probability"
    ])

# Function for login screen
def login_screen():
    
    
    # Embed custom CSS for full-screen video background
    st.markdown(
        """
        <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        video {
            position: fixed;
            top: 0;
            left: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
        }
        .stApp {
            background: transparent;
        }
        </style>
        <video autoplay muted loop>
            <source src="https://cdn.pixabay.com/video/2021/10/30/93956-641767616_large.mp4" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )
    st.title("🏦Smart Loan Application")
    st.subheader("Your one-stop solution for all financial burden")
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Check user credentials
        if username == CREDENTIALS["user"]["username"] and password == CREDENTIALS["user"]["password"]:
            st.session_state["logged_in"] = True
            st.session_state["user_role"] = "user"
            st.success("User login successful!")
        elif username == CREDENTIALS["admin"]["username"] and password == CREDENTIALS["admin"]["password"]:
            st.session_state["logged_in"] = True
            st.session_state["user_role"] = "admin"
            st.success("Admin login successful!")
        else:
            st.error("Invalid username or password.")


# Main loan prediction and recommendation dashboard
def user_app():
    
    
    # Set Groq API client
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )

    
    # Load models and data
    @st.cache_resource
    def load_model(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    @st.cache_resource
    def load_scaler(scaler_path):
        with open(scaler_path, 'rb') as file:
            return pickle.load(file)

    # Paths
    scaler_path = 'scaler.pkl'
    model_path = 'Logistic_Regression_model.pkl'
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)

    # Define feature names in the correct order (as used during model training)
    feature_names = [
        'Credit_History', 'Education', 'ApplicantIncome', 
        'CoapplicantIncome', 'Loan_Amount_Term', 
        'Property_Area', 'Gender'
    ]

    # User input
    st.header("Provide Details for Loan Prediction")

    user_input = {}

    # Customer details
    user_input['Customer_Name'] = st.text_input("Customer Name:")

    # Dropdown for credit history
    credit_history = st.selectbox("Do you have a good credit history?", ['Yes', 'No'])
    user_input['Credit_History'] = 1 if credit_history == 'Yes' else 0

    # Dropdown for education
    education = st.selectbox("Education Level:", ['Graduate', 'Non-Graduate'])
    user_input['Education'] = 0 if education == 'Graduate' else 1

    # Slider for applicant and co-applicant income
    user_input['ApplicantIncome'] = st.number_input("Applicant Income ($/Monthly):", value=None, placeholder="Enter Income in Dollar")
    user_input['CoapplicantIncome'] = st.number_input("Co-applicant Income ($/Monthly):", value=None, placeholder="Enter Income in Dollar")

    # Dropdown for gender
    gender = st.selectbox("Gender:", ['Male', 'Female'])
    user_input['Gender'] = 1 if gender == 'Male' else 0

    # Loan amount term
    user_input['Loan_Amount_Term'] = st.number_input("Enter Loan Amount Term (in months):", value=None, step=1, placeholder="Enter Months in number...", label_visibility="visible", help="Enter Loan term amount in months. The value should be a positive integer. Example - 6, 12, 18, 24, 36 etc.")

    # Dropdown for property area
    property_area = st.selectbox("Property Area:", ['Urban', 'Semiurban', 'Rural'])
    property_area_mapping = {'Urban': 0.6584158416, 'Semiurban': 0.7682403433, 'Rural': 0.6145251397}
    user_input['Property_Area'] = property_area_mapping[property_area]

    # Extra fields
    user_input['Marital Status'] = st.selectbox("Marital Status:", ['Married', 'Unmarried'])

    
    user_input['Dependents'] = st.number_input("Number of Dependents:", min_value=0, step=1)

    user_input['Employment'] = st.selectbox("Self-Employment :", ['Yes', 'No'])

    # Loan amount slider
    user_input['Loan_Amount'] = st.number_input("Loan Amount (1000 x $):", value=None, placeholder="Enter Loan amount as per 1000$...")

    # Convert user input into a dataframe with the correct feature order
    feature_values = {
        'Credit_History': user_input['Credit_History'],
        'Education': user_input['Education'],
        'ApplicantIncome': user_input['ApplicantIncome'],
        'CoapplicantIncome': user_input['CoapplicantIncome'],
        'Loan_Amount_Term': user_input['Loan_Amount_Term'],
        'Property_Area': user_input['Property_Area'],
        'Gender': user_input['Gender'],
    }
    user_input_df = pd.DataFrame([feature_values])[feature_names]
    
    # Prediction button
    if st.button("Predict"):
        # Preprocess input
        scaled_features = scaler.transform(user_input_df[['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']])
        user_input_df[['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']] = scaled_features

        # Predict
        prediction = model.predict(user_input_df)
        prediction_proba = model.predict_proba(user_input_df)[:, 1]
        loan_status = "Approved" if prediction[0] == 1 else "Declined"

        # Display prediction results
        st.subheader("Prediction Result")
        st.write(f"**Loan Status:** {loan_status}")
        st.write(f"**Approval Probability:** {prediction_proba[0]:.2f}")

        # Save user request and prediction to history
        user_input['Loan Status'] = loan_status
        user_input['Approval Probability'] = prediction_proba[0]
        selected_columns = [
        "Customer_Name", "Credit_History", "Education", "ApplicantIncome",
        "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender",
        "Loan_Amount", "Marital Status", "Dependents", "Self-Employment",
        "Loan Status", "Approval Probability"
    ]
    # Create a cleaned DataFrame with the selected columns
        cleaned_user_input = {key: user_input[key] for key in selected_columns if key in user_input}
        cleaned_user_df = pd.DataFrame([cleaned_user_input])

    # Update the session state
        st.session_state["user_history"] = pd.concat(
        [st.session_state["user_history"], cleaned_user_df],
        ignore_index=True
    )

        # Suggestion and recommendation
        st.subheader("Suggestions and Recommendations")

        # Create prompt for Groq
        prompt = (
            f"The user named {user_input['Customer_Name']} - provided the following details for a loan application:\n\n"
            f"Credit History: {user_input['Credit_History']} - 0 stands for bad credit history and 1 stands for good credit history,\n"
            f"Education: {user_input['Education']} - 1 stands for non-graduate and 0 stands for graduate,\n"
            f"Applicant Income: {user_input['ApplicantIncome']} - Applicant Income is provided in Dollar per month ,\n"
            f"Coapplicant Income: {user_input['CoapplicantIncome']} - CoApplicant Income is provided in Dollar per month,\n"
            f"Loan Amount Term: {user_input['Loan_Amount_Term']} - Loan Tenture is Entered as total months,\n"
            f"Property Area: {user_input['Property_Area']}, -  Property type is defined as a number as followed - '0.6584158416'means Urban, '0.7682403433' means Semiurban , '0.6145251397' means Rural  \n"
            f"Gender: {user_input['Gender']}\n\n"
            f"The loan prediction model determined the loan status as "
            f"{loan_status} with a probability of {prediction_proba[0]:.2f}.\n\n"
            f"Please provide reasons for this decision and suggest ways to improve approval chances if declined."
            f"You dont need to add user exact information. Please state that using english word in bullet points"
            f"Based on this, please explain the main reason factor for this decision in simple terms according to user input, and provide suggestions on how the user can modify their inputs to improve the chances of loan approval in future if loan is rejected. No suggestion is needed if the loan is approved.\n"
            f"Give the response in structured format with two sections - Reasons and Suggestions."
        )

        try:
            # Groq API response
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )

            st.write(chat_completion.choices[0].message.content.strip())
            
        except Exception as e:
            st.error(f"An error occurred while fetching recommendations: {e}")
        col1, col2 = st.columns([1, 1]) 
        with col1:
            st.button("Logout", type='primary', on_click=lambda: st.session_state.update({"logged_in": False}))

        with col2:        
            if st.button("New Request", type='secondary'):
                # Clear user input session state
                st.session_state["user_history"] = pd.DataFrame(columns=[
                    "Customer Name", "Credit History", "Education", "Applicant Income", 
                    "Coapplicant Income", "Loan Amount Term", "Property Area", "Gender", 
                    "Loan Amount", "Marital Status", "Dependents", "Self-Employment", 
                    "Loan Status", "Approval Probability"
                ])
                
                # Reset user inputs (the session state for user input data)
                st.session_state["user_input"] = {}

                # Reset any other variables if needed
                st.session_state["logged_in"] = False  # Optional: Reset login status (if you want to reset user session)
                
                # Trigger rerun to clear the form fields and reset the app state
                st.experimental_rerun()

                # Display a success message
                st.success("All inputs have been cleared. You can now make a new loan request.")
def admin_app():
    st.header("Admin Dashboard")
    st.write("View and manage user loan application history.")

    # Display user history
    if not st.session_state["user_history"].empty:
        # Apply the required mappings to the columns
        user_history = st.session_state["user_history"]

        # Map the values to human-readable strings
        user_history["Credit_History"] = user_history["Credit_History"].map({1: "Yes", 0: "No"})
        user_history["Education"] = user_history["Education"].map({0: "Graduate", 1: "Undergraduate"})
        user_history["Property_Area"] = user_history["Property_Area"].map({
            0.6584158416: "Urban", 0.7682403433: "Semiurban", 0.6145251397: "Rural"
        })
        user_history["Gender"] = user_history["Gender"].map({1: "Male", 0: "Female"})
        user_history["Marital Status"] = user_history["Marital Status"].map({1: "Married", 0: "Unmarried"})

        # Specify columns to display
        columns_to_display = [
            "Customer_Name", "Credit_History", "Education", "ApplicantIncome",
            "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender",
            "Loan_Amount", "Marital Status", "Dependents", "Self-Employment",
            "Loan Status", "Approval Probability"
        ]
        
        # Display the dataframe with only the relevant columns
        st.dataframe(user_history[columns_to_display])  # Filter only the columns to display
        
        # Download report button
        csv = user_history[columns_to_display].to_csv(index=False)
        st.download_button("Download Report", data=csv, file_name="user_requests_history.csv", mime="text/csv")
    else:
        st.write("No user requests history available.")

    st.button("Logout", type='primary', on_click=lambda: st.session_state.update({"logged_in": False}))
    
    

# Check if user is logged in
if not st.session_state["logged_in"]:
    login_screen()
else:
    if st.session_state["user_role"] == "user":
        user_app()
    elif st.session_state["user_role"] == "admin":
        admin_app()


