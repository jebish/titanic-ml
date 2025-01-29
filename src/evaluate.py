import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras
from src.data_loader import load_data
import pickle as pkl
import os


def load_and_evaluate(file_path="data/Titanic-Dataset.csv"):
    """Provide file path in case of data error"""
    X_train, X_test, y_train, y_test = load_data(file_path)

    #Load all the models
    lr_model=pkl.load(open("models/lr_model.pkl", 'rb'))
    rf_model=pkl.load(open("models/rf_model.pkl", 'rb'))
    nn_model=keras.models.load_model("models/nn_model.keras")

    #Predict on the test set
    y_pred_lr=lr_model.predict(X_test)
    y_pred_rf=rf_model.predict(X_test)
    y_pred_nn=(nn_model.predict(X_test)>0.5).astype("int32")

    reports={
    "Logistic Regression": classification_report(y_test, y_pred_lr,zero_division=0),
    "Random Forest Model": classification_report(y_test, y_pred_rf,zero_division=0),
    "Neural Network Model": classification_report(y_test, y_pred_nn,zero_division=0)
    }

    
    report_path="results/classification_report.txt"
    os.makedirs(os.path.dirname(report_path),exist_ok=True)

    with open(report_path,"w") as f:
        for model,report in reports.items():
            print(f"=== {model} ===\n{report}\n\n")
            f.write(f"=== {model} ===\n{report}\n\n")

    print(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    load_and_evaluate()