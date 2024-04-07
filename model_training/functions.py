from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_models(models, model_names, X_train, y_train, X_test, y_test, rmspe):
    MSE=[]
    r_2=[]
    accuracy = []
    RMSE = []
    RMSPE=[]
    d={}
    for model in models:
        model.fit(X_train,y_train)
        y_pre=model.predict(X_test)
        MSE.append(round(mean_squared_error(y_true=y_test,y_pred=y_pre),5))
        r_2.append(r2_score(y_true=y_test,y_pred=y_pre))
        accuracy.append((model.score(X_test,y_test))*100)
        RMSE.append(sqrt(mean_squared_error(y_true=y_test,y_pred=y_pre)))
        RMSPE.append(rmspe(y_test,y_pre))
    d=pd.DataFrame({'Modelling Name':model_names,'MSE':MSE,"R_2":r_2,"Accuracy":accuracy,"RMSE":RMSE,"RMSPE":RMSPE})
    return d



def tune_and_fit_model(model, param_dist, X_train, y_train, X_test, scorer, n_iter=1, cv=5, random_state=42):
    # Perform randomized search for best hyperparameters
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state
    )
    random_search.fit(X_train, y_train)

    # Print the best parameters found
    print("Best parameters found:")
    print(random_search.best_params_)

    # Fit the model with the best parameters to the training data
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = best_model.predict(X_test)

    return best_model, y_pred



def plot_feature_importances(model, X_train):
    # Calculate feature importances
    feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=True)

    # Adjust the size of the labels
    plt.tick_params(axis='y', labelsize=6)

    # Plot feature importances
    plt.barh(feature_importances.index, feature_importances['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()