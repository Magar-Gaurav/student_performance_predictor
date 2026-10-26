# Student Performance Predictor

A small ML project that predicts a student's marks from 0 to 100 from three things:
- daily study hours
- attendance
- assignments completed

## Application Preview

![App screenshot](image.png)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## What each file do?

- create_data.py : Makes 200 fake students and saves them to data/student_dataset.csv
- train_model.py : Cleans the data, trains Linear Regression and Random Forest Regressor, saves the better one to  `models/model.pkl` 
- app.py : Streamlit web page where you enter a student's details and see the predicted marks
- requirements.text : The python libraries needed

## How to run (Follow this order)

```bash
pip install -r requirements.txt
python create_data.py
python train_model.py
streamlit run app.py
```

## Future Improvements

- Use real student performance datasets
- Add more features such as:
  - Sleep hours
  - Exam preparation time
  - Previous exam scores
- Deploy online using Streamlit Cloud
