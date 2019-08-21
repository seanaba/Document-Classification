"""
Train classification models with no hyper-parameter tuning
Train classification models with hyper-parameter tuning
"""
import os
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle as pk


def not_tuned_models(x_tr, x_te, y_tr, y_te):
    print("Training Logistic Regression not tuned ...")
    lr = LogisticRegression(random_state=47)
    lr.fit(x_tr, y_tr)
    y_pred = lr.predict(x_te)
    print(classification_report(y_te, y_pred))

    print("Training Random Forest not tuned ...")
    rf = RandomForestClassifier(random_state=47)
    rf.fit(x_tr, y_tr)
    y_pred = rf.predict(x_te)
    print(classification_report(y_te, y_pred))


def best_model(x_tr, x_te, y_tr, y_te, model_path):
    print("GridSearch Cross Validation training ...")

    pipeline_pca_lr = Pipeline([('pca', PCA(n_components=1000)),
                                ('clf', LogisticRegression(random_state=47))])

    lr_params = [{'clf__penalty': ['l2'],
                  'clf__C': [1, 10]}]

    lr = GridSearchCV(estimator=pipeline_pca_lr,
                      param_grid=lr_params,
                      scoring='accuracy',
                      n_jobs=-1,
                      cv=2)

    pipeline_pca_rf = Pipeline([('pca', PCA(n_components=1000)),
                                ('clf', RandomForestClassifier(random_state=47))])

    rf_params = [{'clf__max_features': ['sqrt'],
                  'clf__n_estimators': [100, 400]}]

    rf = GridSearchCV(estimator=pipeline_pca_rf,
                      param_grid=rf_params,
                      scoring='accuracy',
                      n_jobs=-1,
                      cv=2)

    gscv_models = [lr, rf]
    gscv_ind = ['Logistic Regression', 'Random Forest']
    final_model = None
    best_acc = 0
    best_index = 0
    for index, gscv_model in enumerate(gscv_models):
        gscv_model.fit(x_tr, y_tr)
        y_pred = gscv_model.predict(x_te)
        acc = accuracy_score(y_te, y_pred)
        if acc > best_acc:
            final_model = gscv_model
            best_acc = acc
            best_index = index

    print("The best model selected is: ", gscv_ind[best_index])
    print("Best grid search cross validation params are: ",
          final_model.best_params_)
    print("Accuracy is: ", best_acc)
    y_pred = final_model.predict(x_te)
    print(classification_report(y_te, y_pred))

    pk.dump(final_model, open(os.path.join(model_path, 'model'), 'wb'))
