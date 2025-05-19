# predict_new_data.py

import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load saved model and preprocessor
model = load_model('student_perf_model.keras')
preprocessor = joblib.load('preprocessor.pkl')

# Example new data (replace with your real new data)
# Make sure to match the format of the original training data (same columns!)
new_data = pd.DataFrame([{
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "A",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "health",
    "Fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "no",
    "paid": "yes",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 4,
    "G1": 15,
    "G2": 14
}])

# Preprocess new data
X_new = preprocessor.transform(new_data)

# Predict
predicted_grade = model.predict(X_new)
print(f"Predicted Final Grade (G3): {predicted_grade[0][0]:.2f}")
