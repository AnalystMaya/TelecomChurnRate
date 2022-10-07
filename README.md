# Telecom_churn_prediction
I found LGBMClassifier to have the best results on my data set, with a roc_auc score of 0.91 on the test set.
You can find the jupyter notebook with EDA and data preprocessing, as well as training and testing of different models.

To avoid data leackage I used a pipeline for cross validating the models preformance. The features are upsampled and scaled before the classifier is fit and hyperparameters are tuned. 

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from lightgbm import LGBMClassifier

pipe_lgbm = Pipeline([('res', SMOTE(sampling_strategy='minority', random_state=54321)),
                      ('sc', StandardScaler()),
                      ('clf',LGBMClassifier(n_jobs=-1, class_weight='balanced'))])

param_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)],
              'clf__learning_rate': [float(x) for x in np.linspace(0.01, 0.2, num = 5)],
              'clf__num_leaves': [int(x) for x in np.linspace(20, 3000, num = 5)],
              'clf__max_depth': [int(x) for x in np.linspace(start = 3, stop = 12, num = 5)],
             }

lgbm = RandomizedSearchCV(pipe_lgbm, param_grid, n_jobs=-1)
lgbm.fit(X_train, y_train)

cross_validate(lgbm, X_train, y_train, cv=5, scoring=('roc_auc', 'accuracy'))

### Test score
metrics.roc_auc_score(np.array(y_test), lgbm.predict_proba(X_test)[:, 1])

The score 0.91 attests to the performance of the model at distinguishing between the positive and negative classes.
