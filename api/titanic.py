from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class TitanicAPI:
    class _CRUD(Resource):
        def post(self):
            titanic_data = sns.load_dataset('titanic')
            titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
            titanic_data.dropna(inplace=True)
            titanic_data['sex'] = titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
            titanic_data['alone'] = titanic_data['alone'].apply(lambda x: 1 if x == True else 0)
            
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(titanic_data[['embarked']])
            titanic_data = pd.get_dummies(titanic_data, columns=['embarked'])
            titanic_data.dropna(inplace=True)

            X = titanic_data.drop('survived', axis=1)
            y = titanic_data['survived']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            accuracy_dt = accuracy_score(y_test, y_pred_dt)

            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            y_pred_logreg = logreg.predict(X_test)
            accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

            body = request.get_json()
            pclass = int(body.get('pclass', 0))
            sex = body.get('sex', "male")
            age = int(body.get('age', 0))
            sibsp = int(body.get('sibsp', 0))
            parch = int(body.get('parch', 0))
            fare = float(body.get('fare', 0.0))
            embarked = body.get('embarked', "S")
            alone = bool(body.get('alone', False))

            sex_encoded = 1 if sex == 'male' else 0
            alone_encoded = 1 if alone else 0
            embarked_encoded = enc.transform([[embarked]]).toarray()

            new_passenger = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, alone_encoded]])
            new_passenger = np.concatenate((new_passenger, embarked_encoded), axis=1)

            dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))
            death_probability = '{:.2%}'.format(dead_proba)
            # survival_probability = '{:.2%}'.format(alive_proba)

            return jsonify({
                # dead_proba
                'death_probability': death_probability
                # 'survival_probability': survival_probability
            })

    api.add_resource(_CRUD, '/')

