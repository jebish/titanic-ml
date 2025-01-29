from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import numpy as np
import pickle as pkl
import os
from src.data_loader import load_data

#Load data, provide file path if needed
def create_nn_model(input_shape):
    
    Inputs=keras.layers.Input(shape=(input_shape,))
    X=keras.layers.Dense(16, activation='relu',kernel_initializer="glorot_uniform")(Inputs)
    X=keras.layers.Dropout(0.1)(X)
    X=keras.layers.Dense(16, activation='relu',kernel_initializer='glorot_uniform')(X)
    X=keras.layers.Dropout(0.1)(X)
    X=keras.layers.Dense(units=16, activation='relu')(X)
    X=keras.layers.Dense(units=8, activation='relu')(X)
    # X=keras.layers.Dropout(0.2)(X)
    X=keras.layers.Dense(units=4, activation='relu')(X)
    y=keras.layers.Dense(units=1, activation='sigmoid')(X)

    model=keras.models.Model(inputs=Inputs, outputs=y)

    return model


def train_and_save_model(file_path="data/Titanic-Dataset.csv"):
    """Provide file path in case of error"""
    
    X_train, X_test, y_train, y_test = load_data(file_path)
    os.makedirs("models", exist_ok=True)

    #Train a Logistic Regression Model
    print("Training Models...")
    print("Training Logistic Regression Model...")

    lr_model=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)


    with open("models/lr_model.pkl", 'wb') as file:
        pkl.dump(lr_model, file)

    #Train a Random Forest Model
    print("Training Random Forest Model...")
    rf_model=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    with open("models/rf_model.pkl", 'wb') as file:
        pkl.dump(rf_model, file)

    #Train a Neural Network Model


    input_shape=X_train.shape[1]
    nn_model=create_nn_model(input_shape)
    Adam=keras.optimizers.Adam(learning_rate=0.008)
    nn_model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])

    print("Training Neural Network Model...")
    # print(nn_model.summary())
    nn_model.fit(X_train, y_train, epochs=150, batch_size=64)

    nn_model.save("models/nn_model.keras")

    print("Models have been trained and saved!")

if __name__ == "__main__":
    train_and_save_model()