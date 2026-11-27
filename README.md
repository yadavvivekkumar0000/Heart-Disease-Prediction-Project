# Heart Disease Prediction (Machine Learning)

A beginner-friendly ML project that trains a Random Forest classification pipeline on a Cleveland-style UCI Heart Disease dataset and provides a Streamlit form to make demonstration predictions.

## Disclaimer
This is an educational project only—not a medical diagnostic system. It is not validated for clinical use. Do not use predictions to make medical decisions.

## Features
- Dataset retrieval through OpenML
- Missing-value imputation, scaling and one-hot encoding
- Random Forest classifier
- Evaluation report and confusion matrix printed during training
- Streamlit web interface

## Requirements
Python 3.9+ recommended.

## Run in VS Code
1. Clone/download this repository and open the folder in VS Code.
2. Open the terminal (Terminal → New Terminal).
3. Create a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Train and save the model:
   ```bash
   python train_model.py
   ```
   This downloads the dataset the first time and creates `heart_model.joblib`.
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Streamlit opens the application in your browser.

## Features used
age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG, maximum heart rate, exercise angina, ST depression, ST slope, major vessels and thalassemia.

## Project structure
```
Heart_Disease_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```
`heart_model.joblib` is generated when training and is not committed by default unless you choose to include it.

## GitHub upload
```bash
git init
git add .
git commit -m "Initial heart disease prediction ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Heart_Disease_Prediction.git
git push -u origin main
```
## Application Screenshot

![Heart Disease Prediction App](screenshots/heart-disease-app.png)

