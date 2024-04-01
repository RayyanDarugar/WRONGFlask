## Python Titanic Model, prepared for a wwchickendinner.py file
# Import the required libraries for the TitanicModel class
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import json
import os
# {
#     "name": "AI-90002",
#     "age": 22,
#     "sex": "male",
#     "favoritegame": "Maze",
#     "dominanthand": "left",
#     "operatingsystem": "PC",
#     "survivied": 1
#     "year":3
# }
class TBFTModel:
    """A class used to represent the Titanic Model for passenger survival prediction.
    """
    # a singleton instance of TitanicModel, created to train the model only once, while using it for prediction multiple times
    _instance = None
    # constructor, used to initialize the TitanicModel
    def __init__(self):
        # the wwchickendinner ML model
        self.model = None
        self.dt = None
        # define ML features and target
        # self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.features = ['age', 'sex', 'dominanthand', 'firstmove', 'firstattack']
        self.target = 'winorloss'
        # load the wwchickendinner dataset
        self.tbftML_data = pd.read_json('/Users/rayyandarugar/vscode/WRONGFlask/model/tbftML.json')
        print(self.tbftML_data)
        # one-hot encoder used to encode 'embarked' column
        self.encoder = OneHotEncoder(handle_unknown='ignore')
    # clean the tbftmodel dataset, prepare it for training
    def _clean(self):
        # Drop unnecessary columns
        # self.wwchickendinner_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        # Convert boolean columns to integers
        self.tbftML_data['sex'] = self.tbftML_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.tbftML_data['dominanthand'] = self.tbftML_data['dominanthand'].apply(lambda x: 1 if x == 'left' else 0)
        self.tbftML_data['firstmove'] = self.tbftML_data['firstmove'].apply(lambda x: 1 if x == 2 else 0)
        self.tbftML_data['firstattack'] = self.tbftML_data['firstattack'].apply(lambda x: 1 if x == 5 else 0)
        #self.wwchickendinner_data['alone'] = self.wwchickendinner_data['alone'].apply(lambda x: 1 if x == True else 0)
        # Drop rows with missing 'embarked' values before one-hot encoding
        # self.wwchickendinner_data.dropna(subset=['embarked'], inplace=True)
        # One-hot encode 'embarked' column
        # onehot = self.encoder.fit_transform(self.wwchickendinner_data[['embarked']]).toarray()
        # cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        # onehot_df = pd.DataFrame(onehot, columns=cols)
        # self.wwchickendinner_data = pd.concat([self.wwchickendinner_data, onehot_df], axis=1)
        # self.wwchickendinner_data.drop(['embarked'], axis=1, inplace=True)
        # Add the one-hot encoded 'embarked' features to the features list
        # self.features.extend(cols)
        # Drop rows with missing values
        self.tbftML_data.dropna(inplace=True)
    # train the wwchickendinner model, using logistic regression as key model, and decision tree to show feature importance
    def _train(self):
        # split the data into features and target
        X = self.tbftML_data[self.features]
        y = self.tbftML_data[self.target]
        # perform train-test split
        self.model = LogisticRegression(max_iter=1000)
        # train the model
        self.model.fit(X, y)
        # train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
    @classmethod
    def get_instance(cls):
        """ Gets, and conditionaly cleans and builds, the singleton instance of the TitanicModel.
        The model is used for analysis on wwchickendinner data and predictions on the survival of theoritical passengers.
        Returns:
            TitanicModel: the singleton _instance of the TitanicModel, which contains data and methods for prediction.
        """
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance
    def predict(self, passenger):
        """ Predict the survival probability of a passenger.
        Args:
            passenger (dict): A dictionary representing a passenger. The dictionary should contain the following keys:
                'pclass': The passenger's class (1, 2, or 3)
                'sex': The passenger's sex ('male' or 'female')
                'age': The passenger's age
                'sibsp': The number of siblings/spouses the passenger has aboard
                'parch': The number of parents/children the passenger has aboard
                'fare': The fare the passenger paid
                'embarked': The port at which the passenger embarked ('C', 'Q', or 'S')
                'alone': Whether the passenger is alone (True or False)
        Returns:
           dictionary : contains lose and win probabilities
        """
        print(passenger)
        # clean the passenger data
        passenger_df = pd.DataFrame(passenger, index=[0])
        # passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_df['dominanthand'] = passenger_df['dominanthand'].apply(lambda x: 1 if x == 'left' else 0)
        passenger_df['firstmove'] = passenger_df['firstmove'].apply(lambda x: 1 if x == 2 else 0)
        passenger_df['firstattack'] = passenger_df['firstattack'].apply(lambda x: 1 if x == 5 else 0)
        # passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x == True else 0)
        # onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        # cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        # onehot_df = pd.DataFrame(onehot, columns=cols)
        # passenger_df = pd.concat([passenger_df, onehot_df], axis=1)
        passenger_df.drop(['name'], axis=1, inplace=True)
        # predict the survival probability and extract the probabilities from numpy array
        lose, win = np.squeeze(self.model.predict_proba(passenger_df))
        # return the survival probabilities as a dictionary
        return {'lose': lose, 'win': win}
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.
        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)}
def initTBFTModel():
    """ Initialize the Titanic Model.
    This function is used to load the Titanic Model into memory, and prepare it for prediction.
    """
    TBFTModel.get_instance()
def testTBFTModel():
    """ Test the Titanic Model
    Using the TitanicModel class, we can predict the survival probability of a passenger.
    Print output of this test contains method documentation, passenger data, survival probability, and survival weights.
    """
    # setup passenger data for prediction
    print(" Step 1:  Define theoritical passenger data for prediction: ")
    passenger = {
        'name': ['Will Bartelt'],
        'sex': ['male'],
        'age': [16],
        'dominanthand': ['right'],
        'firstmove': [3],
        'firstattack': [5],
    }
    print("\t", passenger)
    print()
    # get an instance of the cleaned and trained Titanic Model
    tbftModel = TBFTModel.get_instance()
    print(" Step 2:", tbftModel.get_instance.__doc__)
    # print the survival probability
    print(" Step 3:", tbftModel.predict.__doc__)
    probability = tbftModel.predict(passenger)
    print('\t lose probability: {:.2%}'.format(probability.get('lose')))
    print('\t win probability: {:.2%}'.format(probability.get('win')))
    print()
    # print the feature weights in the prediction model
    print(" Step 4:", tbftModel.feature_weights.__doc__)
    importances = tbftModel.feature_weights()
    for feature, importance in importances.items():
        print("\t\t", feature, f"{importance:.2%}") # importance of each feature, each key/value pair
if __name__ == "__main__":
    print(" Begin:", testTBFTModel.__doc__)
    testTBFTModel()