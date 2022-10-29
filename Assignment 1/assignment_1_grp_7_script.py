# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pandas import DataFrame
from collections import namedtuple
import pandas as pd

# %%
def calculate_PRF(target, value):
    P = precision_score(target, value, average="macro")
    R = recall_score(target, value, average="macro")
    F = f1_score(target, value, average="macro")
    print(Test(P,R,F))
    return Test(P,R,F)

# %%

def classifiers(clf):
    test_prf = []

    # Feature 1 : Using Count 
    text_clf_using_count = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', clf),
    ])
    text_clf_using_count.fit(twenty_train.data, twenty_train.target)
    predicted_test_using_count = text_clf_using_count.predict(twenty_test_data)
    print(f"{clf} Precision, Recall, F1 scores:")
    test_prf.append(calculate_PRF(twenty_test.target, predicted_test_using_count))
    print(f"{clf}: accuracy of Count vector {np.mean(predicted_test_using_count == twenty_test.target)*100}")
    # print(metrics.classification_report(twenty_test.target, predicted_test_using_count, target_names=twenty_test.target_names))

    # Feature 2 : Using TF
    text_clf_using_tf = Pipeline([
        ('vect', CountVectorizer()),
        ('tdidf', TfidfTransformer(use_idf=False)),
        ('clf', clf),
    ])
    text_clf_using_tf.fit(twenty_train.data, twenty_train.target)
    predicted_test_using_tf = text_clf_using_tf.predict(twenty_test_data)
    print(f"{clf} Precision, Recall, F1 scores:")
    test_prf.append(calculate_PRF(twenty_test.target, predicted_test_using_tf))
    print(f"{clf}: accuracy of Tf {np.mean(predicted_test_using_tf == twenty_test.target)*100}")
    # print(metrics.classification_report(twenty_test.target, predicted_test_using_tf, target_names=twenty_test.target_names))

    # Feature 3 : Using TFIDF
    text_clf_using_tfidf = Pipeline([
        ('vect', CountVectorizer()),
        ('tdidf', TfidfTransformer(use_idf=True)),
        ('clf', clf),
    ])
    text_clf_using_tfidf.fit(twenty_train.data, twenty_train.target)
    predicted_test_using_tfidf = text_clf_using_tfidf.predict(twenty_test_data)
    print(f"{clf} Precision, Recall, F1 scores:")
    test_prf.append(calculate_PRF(twenty_test.target, predicted_test_using_tfidf))
    print(f"{clf}: accuracy of TfIdf {np.mean(predicted_test_using_tfidf == twenty_test.target)*100}")
    # print(metrics.classification_report(twenty_test.target, predicted_test_using_tfidf, target_names=twenty_test.target_names))

    return test_prf

# %%
def plot_data_classifier(data_frame,MarkerSize=8):
    fig,ax = plt.subplots(figsize=(20, 10))
    clr = ['orange', 'blue', 'green']
    for i,row in enumerate(data_frame.index):
        ax.plot(data_frame.columns, [object.Precision for object in data_frame.loc[row,]], marker='o', color=clr[i], markersize=MarkerSize, linewidth=0, label="Precision by "+str(row))
        ax.plot(data_frame.columns, [object.Recall for object in data_frame.loc[row,]], marker='x', color=clr[i], markersize=MarkerSize, linewidth=0, label="Recall by "+str(row))
        ax.plot(data_frame.columns, [object.F1 for object in data_frame.loc[row,]], marker='^', color=clr[i], markersize=MarkerSize, linewidth=0, label="F1 by "+str(row))
    # x-axis is adjusted randomly to prevent overlap while plotting
    for objects in ax.lines:
        xs = objects.get_xydata()[:,0]
        objects.set_xdata(xs + np.random.uniform(-0.25, 0.25, xs.shape))
    ax.relim()
    ax.autoscale(enable=True)
    plt.xlabel("Classifier", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(0.6, 1.0)
    plt.legend()
    plt.title("Comparison of Classifiers", fontsize=15)
    plt.show()

# %%
def run_classifiers():
        CLF_Names = ["Naive Bayes", "Linear-SVM", "Logistic Regression"]
        CLF = [
                MultinomialNB(), 
                SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None),
                SGDClassifier(loss='log_loss',random_state=42)
        ]

        accuracy_scores = dict()
        for i in range(len(CLF)):
                accuracy_scores[CLF_Names[i]]=classifiers(CLF[i])
        scores = pd.DataFrame(accuracy_scores, index=["Count","Tf","Tfidf"])

        plot_data_classifier(scores)

# %%
def countVectorizer_exp(case=True, lang=None, analyzer_word='word', ngram_val=(1,1), max_feature=None):
    text_clf_NB = Pipeline([
        ('vect', CountVectorizer(lowercase=case, stop_words=lang, analyzer=analyzer_word, ngram_range=ngram_val, max_features=max_feature)),
        ('tdidf', TfidfTransformer()),
        ('clf_NB', MultinomialNB()),
    ])

    text_clf_NB.fit(twenty_train.data, twenty_train.target)
    return text_clf_NB.predict(twenty_test.data)

# %%
def exp_cv_lowercase(val):
    pred = countVectorizer_exp(case=val) # Lowercase - True
    print(f"Accuracy if lowercase is {val} : {np.mean(pred == twenty_test.target)}")
    pass

def exp_cv_stopwords(val):
    pred = countVectorizer_exp(lang=val) # Lowercase - True
    print(f"Accuracy if stop_words is {val} : {np.mean(pred == twenty_test.target)}")
    pass

def exp_cv_analyzer(val, ngram):
    pred = countVectorizer_exp(analyzer_word=val, ngram_val=ngram) # Lowercase - True
    print(f"Accuracy if analyzer is {val} and n_gram_range is {ngram} : {np.mean(pred == twenty_test.target)}")

def exp_cv_max_feature(val):
    pred = countVectorizer_exp(max_feature=val) # Lowercase - True
    print(f"Accuracy if max_feature value is {val} : {np.mean(pred == twenty_test.target)}")

# %%
def run_cv_exp():
    lower_case = [True, False]
    stop_words_list = ['english', None]
    analyzer_list = ['word', 'char', 'char_wb']
    ngram_list = [(1,1), (1,2),(2,2)]
    max_feature = [None,10,100,1000]
    
    print(f"Count Vectorizer parameters experimentation:")
    for i in range(4):
        if i < len(stop_words_list):
            exp_cv_lowercase(lower_case[i])
            exp_cv_stopwords(stop_words_list[i])
        
        if i < len(analyzer_list):
            exp_cv_analyzer(analyzer_list[i], ngram_list[i])

        if i < len(max_feature):
            exp_cv_max_feature(max_feature[i])    

# %%
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

twenty_train_data = twenty_train.data
twenty_test_data = twenty_test.data

Test  = namedtuple("Test", ["Precision","Recall","F1"])

run_classifiers()
# for plotting the x-axis is adjusted randomly to prevent overlap
run_cv_exp()


