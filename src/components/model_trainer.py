import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                # "Logistic Regression": LogisticRegression(),
                # "K-Neighbors Classifier": KNeighborsClassifier(),
                # "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                # "XGBClassifier": XGBClassifier(eval_metric='mlogloss'),
                # "CatBoost Classifier": CatBoostClassifier(verbose=False),
                # "AdaBoost Classifier": AdaBoostClassifier(),
                # "Gradient Boosting Classifier": GradientBoostingClassifier()
            }

            params = {
            #     "Logistic Regression": {
            #     'C': [0.1, 1, 10]
            # },
            #     "K-Neighbors Classifier": {
            #     'n_neighbors': [3, 5, 7]
            # },
            #     "Decision Tree": {
            #     'criterion': ['gini', 'entropy', 'log_loss'],
            #     'max_depth': [None, 5, 10, 20]
            # },
                "Random Forest Classifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20]
            },
            #     "Gradient Boosting Classifier": {
            #     'learning_rate': [0.1, 0.01, 0.05],
            #     'n_estimators': [50, 100, 200],
            #     'subsample': [0.6, 0.8, 1.0]
            # },
            #     "XGBClassifier": {
            #     'learning_rate': [0.1, 0.01, 0.05],
            #     'n_estimators': [50, 100, 200],
            #     'max_depth': [3, 5, 7]
            # },
            #     "CatBoost Classifier": {
            #     'depth': [6, 8, 10],
            #     'learning_rate': [0.01, 0.05, 0.1],
            #     'iterations': [50, 100, 200]
            # },
            #     "AdaBoost Classifier": {
            #     'n_estimators': [50, 100, 200],
            #     'learning_rate': [0.1, 0.5, 1.0]
            # }
        }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print("Best Model:", best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model Found")
            logging.info("Best found Model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_test_predict = best_model.predict(X_test)

            Accuracy = accuracy_score(y_test, y_test_predict)
            print("Accuracy", Accuracy)
            return Accuracy
            
        except Exception as e:
            raise CustomException(e, sys)   