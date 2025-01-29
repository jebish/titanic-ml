import numpy as np
import pickle as pkl
from tensorflow import keras

def get_user_input():
    """Ask user for input features and returns them as a numpy array"""
    print("Enter passenger details:")
    pclass = int(input("Pclass (1/2/3): "))
    age = float(input("Age: "))
    sibsp = int(input("Siblings/Spouses Aboard: "))
    parch = int(input("Parents/Children Aboard: "))
    fare = float(input("Fare: "))
    sex = int(input("Sex (0=Female, 1=Male): "))
    embarked = int(input("Embarked (0=C, 1=Q, 2=S): "))

    return np.array([[pclass, age, sibsp, parch, fare, sex, embarked]])

def make_predictions():
    """Loads models and makes predictions on user input"""
    X_input= get_user_input()

    lr_model=pkl.load(open("models/lr_model.pkl", 'rb'))
    rf_model=pkl.load(open("models/rf_model.pkl", 'rb'))
    nn_model=keras.models.load_model("models/nn_model.keras")

    print("\nPredictions:")
    print(f"Logistic Regression: {'Survived' if lr_model.predict(X_input)[0] == 1 else 'Did not survive'}")
    print(f"Random Forest: {'Survived' if rf_model.predict(X_input)[0] == 1 else 'Did not survive'}")
    print(f"Neural Network: {'Survived' if nn_model.predict(X_input)[0] > 0.5 else 'Did not survive'}")


if __name__== "__main__":
    make_predictions()