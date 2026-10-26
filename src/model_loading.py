from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(model_name):
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run model_training.py first."
        )

    return joblib.load(model_path)


def predict_marks(study_hours, attendence, assignments):
    model = load_model("linear_regression.pkl")

    student = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendence": attendence,
        "Assignments": assignments
    }])

    prediction = model.predict(student)[0]
    prediction = max(0,min(100,prediction))

    return prediction



if __name__ == "__main__":
    test_students = [
        {"StudyHours": 1, "Attendence": 40, "Assignments": 2},
        {"StudyHours": 2, "Attendence": 55, "Assignments": 4},
        {"StudyHours": 4, "Attendence": 70, "Assignments": 6},
        {"StudyHours": 5, "Attendence": 85, "Assignments": 8},
        {"StudyHours": 7, "Attendence": 90, "Assignments": 9},
        {"StudyHours": 9, "Attendence": 98, "Assignments": 10},
        {"StudyHours": 9, "Attendence": 45, "Assignments": 8},
        {"StudyHours": 2, "Attendence": 98, "Assignments": 9},
        {"StudyHours": 7, "Attendence": 80, "Assignments": 3},
        {"StudyHours": 5, "Attendence": 100, "Assignments": 10},
    ]

    results = []

    for student in test_students:
        predicted_marks = predict_marks(
            study_hours=student["StudyHours"],
            attendence=student["Attendence"],
            assignments=student["Assignments"]
        )

        results.append({
            **student,
            "PredictedMarks": round(predicted_marks, 2)
        })

    results_df = pd.DataFrame(results)

    print("\nStudent Performance Predictions:\n")
    print(results_df.to_string(index=False))
