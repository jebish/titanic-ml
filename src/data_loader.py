import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path="data/Titanic-Dataset.csv"):
    df=pd.read_csv(file_path)

    #Drop columns that are not useful
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    #Fill missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True) #Use Median for Age
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) #Use Mode for Embarked
    df['Fare'].fillna(df['Fare'].median(), inplace=True) #Use Median for Fare

    #Check if missing values are present
    if (df.isnull().sum().sum()):
        print("Missing values in the dataset")

    #Encode Categorical Variables
    label_encoders= LabelEncoder()
    df['Sex']=label_encoders.fit_transform(df['Sex'])
    df['Embarked']=label_encoders.fit_transform(df['Embarked'])

    #Define X (Features) and y (Target)
    X=df.drop("Survived",axis=1)
    y=df['Survived']

    #Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    load_data()