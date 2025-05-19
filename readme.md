# Predicting Student Performance Using Artificial Neural Network

A Python project that trains an ANN on student data to predict final grades (G3) based on demographics, past performance, and social features. Includes end-to-end preprocessing, training, model persistence, and inference scripts.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training the Model](#training-the-model)
5. [Making Predictions](#making-predictions)
6. [Customization & Tips](#customization--tips)
7. [License](#license)

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional but recommended) virtualenv

---

## Installation

1. **Clone or unzip the repository**

```bash
git clone <your-repo-url>
cd predicting-student-performance-ann
```

_Or unzip the provided ZIP and `cd` into its folder._

2. **Create a virtual environment**

```bash
python3 -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

3. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install pandas matplotlib scikit-learn tensorflow joblib
```

4. **Verify dataset availability**
   Ensure `student-por.csv` (the UCI student performance file) is in the project root.

---

## Project Structure

```
├── student-por.csv          # Raw dataset (semicolon-delimited)
├── train_model.py           # Script: load data, preprocess, train ANN, save artifacts
├── predict_new_data.py      # Script: load saved model + preprocessor, run inference
├── student_perf_model.keras # (Generated) Trained Keras model
├── preprocessor.pkl         # (Generated) Fitted preprocessing pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Training the Model

Run the training pipeline to create (or update) your model and preprocessing objects:

```bash
python train_model.py
```

- **What happens**

1.  Reads and splits `student-por.csv` into features (X) and target (y).
2.  Applies scaling and one-hot encoding.
3.  Trains a 2-layer ANN (64 → 32 units) over 50 epochs.
4.  Saves:
    - `student_perf_model.keras`
    - `preprocessor.pkl`

- **Output**
- Training and validation loss curves (displayed in a Matplotlib window).
- Model & preprocessor files in the project root.

---

## Making Predictions

Once you’ve trained the model, you can predict grades for new students:

1. Edit the `new_data` dictionary in `predict_new_data.py` to supply feature values.
2. Run:

```bash
python predict_new_data.py
```

3. The script will print the predicted final grade (G3).

---

## Customization & Tips

- **Adjust hyperparameters**
  Modify layer sizes, activation functions, epochs, or batch size in `train_model.py` to explore different performance profiles.

- **Batch predictions**
  Load a CSV of unseen samples in `predict_new_data.py` and transform them all at once:

```python
raw_df = pd.read_csv("unseen_students.csv", sep=";")
X_new   = preprocessor.transform(raw_df)
preds   = model.predict(X_new)
```

- **Evaluation metrics**
  Beyond MSE, compute RMSE or R² on your held-out test set:

```python
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
rmse   = mean_squared_error(y_test, y_pred, squared=False)
r2     = r2_score(y_test, y_pred)
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
