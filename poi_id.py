#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                  'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
                  'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 
                  'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
                 
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#print data_dict

### Task 2: Remove outliers
#Get labels and features
df = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = float)
df = pd.DataFrame(df, columns = features_list)

'''
print len(df[df['poi'] == 1.0])
print len(df[df['poi'] == 0.0])
'''
#Identify any outliers by name
#print data_dict.keys()
##Found 2 names that do not appear to be people:
##'TOTAL'
##'THE TRAVEL AGENCY IN THE PARK'

from outlier_cleaner_final2 import outlierCleaner
cleaned_data = outlierCleaner( df )

data_dict.pop("TOTAL", 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


### Task 3: Create new feature(s)
from computefraction import computeFraction

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


##Selecting best K features out of all
from sklearn.feature_selection import SelectKBest
K = 8
selector = SelectKBest(k=K)
selector.fit(features, labels)
#print selector.scores_

#List comprehension of the selector scores and features
fs = [[e, i] for e, i in zip(selector.scores_, features_list[1:])]

fs = sorted(fs, key=lambda fs_list: fs_list[0], reverse = True)
fs = fs[:K]
features_list = [e[1] for e in fs]
features_list = ['poi'] + features_list
#print '***************'
#print fs
#print features_list

#Re-create features, labels based on optimized features_list
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Assuming GaussianNB, below are the results for particular K's
#K = 5, Precision: 0.47626	Recall: 0.34100
#K = 6, Precision: 0.47247	Recall: 0.34750
#K = 7, Precision: 0.48092	Recall: 0.37800
#K = 8, Precision: 0.48026	Recall: 0.40750
#K = 9, Precision: 0.47995	Recall: 0.40700
#K = 10, Precision: 0.44684	Recall: 0.39300
#K = 8 is optimum



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest



GNB = GaussianNB()
parameters = {'kbest__k': [1,2,3,4,5,6,7,8]}
Min_Max_scaler = MinMaxScaler()
kbest = SelectKBest()
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('kbest', kbest), ('GNB', GNB)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

#Results - Outliers = TOTAL, 'THE TRAVEL AGENCY IN THE PARK'
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=8, score_func=<function f_classif at 0x116701de8>)), ('GNB', GaussianNB())])
#	Accuracy: 0.84650	Precision: 0.45370	Recall: 0.36500	F1: 0.40454	F2: 0.37985
#	Total predictions: 14000	True positives:  730	False positives:  879	False negatives: 1270	True negatives: 11121

#Results - Outliers = All
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=8, score_func=<function f_classif at 0x11677c758>)), ('GNB', GaussianNB())])
#	Accuracy: 0.84143	Precision: 0.43963	Recall: 0.40050	F1: 0.41915	F2: 0.40776
#	Total predictions: 14000	True positives:  801	False positives: 1021	False negatives: 1199	True negatives: 10979


'''
KNC = KNeighborsClassifier()
parameters = {'KNC__n_neighbors' : [1, 5, 10],
          'KNC__algorithm' : ('ball_tree', 'kd_tree'),
          'KNC__leaf_size' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          'KNC__p' : [1,2,3], 'kbest__k':range(1,9)}
Min_Max_scaler = MinMaxScaler()
kbest = SelectKBest()
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ("kbest", kbest), ('KNC', KNC)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

#Results - Outliers = All
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=3, score_func=<function f_classif at 0x116bff9b0>)), ('KNC', KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=5, p=1,
#           weights='uniform'))])
#	Accuracy: 0.88013	Precision: 0.64470	Recall: 0.22500	F1: 0.33358	F2: 0.25868
#	Total predictions: 15000	True positives:  450	False positives:  248	False negatives: 1550	True negatives: 12752
'''

'''
#Ensemble AdaBoost
ensemble = ensemble.AdaBoostClassifier()
parameters = {'ensemble__n_estimators': [10, 20, 40]}
Min_Max_scaler = MinMaxScaler()
#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('ensemble', ensemble)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

#Results - Outliers = TOTAL, 'THE TRAVEL AGENCY IN THE PARK'
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=2, whiten=False)), ('ensemble', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#          learning_rate=1.0, n_estimators=10, random_state=None))])
#	Accuracy: 0.83500	Precision: 0.26083	Recall: 0.12950	F1: 0.17307	F2: 0.14400
#	Total predictions: 15000	True positives:  259	False positives:  734	False negatives: 1741	True negatives: 12266

#Results - Outliers = All
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=2, whiten=False)), ('ensemble', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#          learning_rate=1.0, n_estimators=10, random_state=None))])
#	Accuracy: 0.83064	Precision: 0.33363	Recall: 0.18600	F1: 0.23884	F2: 0.20406
#	Total predictions: 14000	True positives:  372	False positives:  743	False negatives: 1628	True negatives: 11257
'''


'''
#Tree Classifier
tree = DecisionTreeClassifier()
parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[1, 2, 10, 20],
                'tree__max_depth':[10,15,20,25],
                'tree__max_leaf_nodes':[10,30,50,70]}


# use scaling in GridSearchCV
Min_Max_scaler = MinMaxScaler()
#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('tree', tree)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

#Results - Outliers = TOTAL, 'THE TRAVEL AGENCY IN THE PARK'
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=2, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
#            max_features=None, max_leaf_nodes=70, min_samples_leaf=1,
#            min_samples_split=1, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='random'))])
#	Accuracy: 0.81140	Precision: 0.23849	Recall: 0.18900	F1: 0.21088	F2: 0.19718
#	Total predictions: 15000	True positives:  378	False positives: 1207	False negatives: 1622	True negatives: 11793

#Results - Outliers = All
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=2, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
#            max_features=None, max_leaf_nodes=70, min_samples_leaf=1,
#            min_samples_split=1, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best'))])
#	Accuracy: 0.80171	Precision: 0.29834	Recall: 0.28700	F1: 0.29256	F2: 0.28920
#	Total predictions: 14000	True positives:  574	False positives: 1350	False negatives: 1426	True negatives: 10650
'''

#GridSearch
gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, labels)
clf = gs.best_estimator_




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)