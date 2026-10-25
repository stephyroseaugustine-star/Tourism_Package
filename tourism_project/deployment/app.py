import os
import joblib
import pandas as pd
import streamlit as st

# Load the trained model pipeline
model_path = os.path.join(
    os.path.dirname(__file__),
    "tourism_package_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

# App title and description
st.title("Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on their profile, travel preferences,
and sales interaction details.
""")

# Customer input fields
age = st.number_input("Age", 18, 100, 35)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox("City Tier", [1, 2, 3])

duration = st.number_input(
    "Duration of Pitch", 0.0, 120.0, 10.0, 1.0
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

persons = st.number_input(
    "Number of Persons Visiting", 1, 20, 2
)

followups = st.number_input(
    "Number of Follow-ups", 0, 20, 3
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

property_star = st.selectbox(
    "Preferred Property Star", [3.0, 4.0, 5.0]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced"]
)

trips = st.number_input("Number of Trips", 0.0, 100.0, 2.0)

passport = st.selectbox(
    "Has Passport?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

satisfaction = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

owncar = st.selectbox(
    "Owns a Car?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

children = st.number_input(
    "Number of Children Visiting", 0.0, 20.0, 0.0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

income = st.number_input(
    "Monthly Income", 0.0, 1000000.0, 25000.0, 1000.0
)

# Prepare input using the original training feature names
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": trips,
    "Passport": passport,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income
}])

# Predict purchase outcome
if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The model predicts that the customer is likely "
            "to purchase the tourism package."
        )
    else:
        st.info(
            "The model predicts that the customer is unlikely "
            "to purchase the tourism package."
        )

    # Display estimated purchase probability when supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = probabilities[classes.index(1)]
            st.metric(
                "Estimated Purchase Probability",
                f"{probability:.1%}"
            )
