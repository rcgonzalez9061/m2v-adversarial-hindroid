import os, pickle
from scipy.sparse import csr_matrix, load_npz
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

def create_model(outfolder, app_path):
    model_path = os.path.join(outfolder, 'model.pkl')
    
    all_apps = pd.read_csv("data/out/all-apps/all_apps.csv", index_col='app')
    test_apps = pd.read_csv(app_path, index_col = 'app')
    test_apps_mal = test_apps.join(all_apps, how = 'left')

    all_apps_features =  pd.read_csv('data/out/all-apps/features.csv', index_col='uid')
    all_apps_features['app'] = all_apps_features.index.map(
        pd.read_csv('data/out/all-apps/app_map.csv', index_col='uid').app
    )
    all_apps_features['malware'] = (all_apps_features['app'].map(all_apps.category)=='malware').astype(int)
    all_apps_features['category'] = all_apps_features.app.map(all_apps.category)

    train = pd.read_csv('data/out/training-sample/app_map.csv', usecols=['app'])
    train = all_apps_features.set_index('app').loc[train.app]

    test_sample = all_apps_features[np.logical_not(
        all_apps_features.app.apply(lambda x: x in train.index)
    )]
    test_sample['category'] = test_sample.app.map(all_apps.category)
    test_sample = test_sample[test_sample.category!='random-apps']



    X_train, y_train = train.drop(columns=['malware', 'category']), train.malware
    X_test, y_test = test_sample.drop(columns=['app', 'malware', 'category']), test_sample.malware


    model = RandomForestClassifier(max_depth=3, n_jobs=-1)  # probably overfit
    model.fit(X_train, y_train)

    class_train = classification_report(model.predict(X_train), y_train)
    class_test = classification_report(model.predict(X_test), y_test)
    output = test_sample[['app','malware','category']].join(y_test, how = 'left')
    m_score = f1_score(model.predict(X_test), y_test)
    output.to_csv('output.csv')
    with open(model_path, 'wb') as file:
            pickle.dump(class_train, file)
            pickle.dump(class_test,file)
            pickle.dump(m_score,file)