🚀 <h1>Titanic Survival Prediction - Machine Learning Pipeline</h1>

A complete, modular, and scalable ML system to predict passenger survival on the Titanic using Logistic Regression, Random Forest, and a Neural Network.

✅ Modular Code
✅ Training, Evaluation and Inference
✅ Results Saving

📂 <h1>Project Structure</h1>

titanic-ml-project/
│-- src/                  # Source code
│   │-- __init__.py       
│   │-- data_loader.py    # Handles data loading & preprocessing
│   │-- train.py          # Trains models & saves them
│   │-- evaluate.py       # Evaluates models & saves reports
│   │-- inference.py      # Runs inference using trained models
│   
│-- main.py               # Runs training, evaluation & inference together
│-- notebooks/            # Jupyter notebooks for trial of Neural Network (optional)
│-- models/               # Stores trained models
│-- results/              # Stores evaluation reports (text/images)
│-- data/                 # Directory for dataset (not included)
│-- requirements.txt      # Python dependencies
│-- README.md             # Project documentation

📊 <h1>Dataset</h1>

The dataset used is the Titanic Survival Dataset from [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data).

It contains 12 features, including whether the passenger survived or not.
However, some of the features aren't used and the data contains null values. All the preprocessing is done by data_loader.py.

<h2>Features Used</h2>
1. Pclass (Ticket Class)
2. Sex (Gender)
3. Age
4. SibSp (Number of siblings/spouses aboard)
5. Parch (Number of parents/ children aboard)
6. Fare (Ticket price)
7. Embarked (Port of embarkation)

🔸Target Variable: Survived (0 = No, 1 = Yes)

⚙️ <h1>Setup & Installation</h1>

1️⃣ Clone the Repository

git clone https://github.com/your-username/titanic-ml-project.git
cd titanic-ml-project

2️⃣ Create a Virtual Environment (Optional)

python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Download the Dataset

Downlaod & Extract the [dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data) and place the ".csv" file inside the data/ folder.

Alternatively, you can provide dataset path as argument.

🚀 <h1>Running the Project</h1>

1️⃣ Train Models

python -m src.train

2️⃣ Evaluate Models

python -m src.evaluate

3️⃣ Perform Inference (User Input)

python -m src.inference

🔥 Perform Train, Evaluate and Inference at once

python main.py #user data/Titanic-Dataset.csv as data
python main.py --data "your_data_location" #Provide another location for dataset

📈 <h1>Model Performance</h1>

=== Logistic Regression ===
              precision    recall  f1-score   support

           0       0.88      0.81      0.84       105
           1       0.76      0.84      0.79        74

    accuracy                           0.82       179
   macro avg       0.82      0.82      0.82       179
weighted avg       0.83      0.82      0.82       179


=== Random Forest Model ===
              precision    recall  f1-score   support

           0       0.83      0.87      0.85       105
           1       0.80      0.74      0.77        74

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.81       179
weighted avg       0.81      0.82      0.81       179


=== Neural Network Model ===
              precision    recall  f1-score   support

           0       0.80      0.84      0.82       105
           1       0.75      0.70      0.73        74

    accuracy                           0.78       179
   macro avg       0.78      0.77      0.77       179
weighted avg       0.78      0.78      0.78       179


💡</h1>Future Improvements</h1>

⌛ Hyperparameter tuning (Grid Search / Bayesian Optimization)
⌛ More advanced models (XGBoost, SVM)
⌛ Deployment (Flask API, Streamlit, or FastAPI)

📝 <h1>By Jebish7</h1>
