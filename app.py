import streamlit as st
from src.model_loading import predict_marks
from exceptions import PredictionAppError, StudentInputError

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# App title and description
st.title(" Student Performance Predictor")

st.write(
    "Enter a student's study habits and academic activity "
    "to estimate their marks using a trained Machine Learning model."
)

st.info(
    "This application provides estimates, does not guaranteed exam results."
)

# Input form
with st.form("student_form"):
    st.subheader("Student Information")

    study_hours = st.number_input(
        "Daily study hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    assignments = st.number_input(
        "Assignments completed",
        min_value=0,
        max_value=10,
        value=8,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Marks",
        type="primary",
        use_container_width=True
    )

# Make prediction after form submission
if submitted:
    try:
        if study_hours < 0 or study_hours > 24:
            raise StudentInputError(
                "Study hours must be between 0 and 24."
            )

        if attendance < 0 or attendance > 100:
            raise StudentInputError(
                "Attendance must be between 0 and 100."
            )

        if assignments < 0 or assignments > 10:
            raise StudentInputError(
                "Assignments must be between 0 and 10."
            )

        predicted_marks = predict_marks(
            study_hours=study_hours,
            attendence=attendance,
            assignments=assignments
        )

        if predicted_marks >= 80:
            st.success("High predicted performance!")
        elif predicted_marks >= 60:
            st.info("Moderate predicted performance.")
        else:
            st.warning(
                "The model predicts a lower score. "
                "Consider reviewing the student's learning needs."
            )

        st.subheader("Prediction Result")
        st.metric(
            "Estimated Marks",
            f"{predicted_marks:.2f} / 100"
        )
        st.progress(predicted_marks / 100)

    except StudentInputError as error:
        st.warning(str(error))

    except FileNotFoundError:
        st.error(
            "The trained model is missing. "
            "Please train and save the model first."
        )

    except Exception as error:
        raise PredictionAppError(
            f"Prediction failed: {error}"
        )