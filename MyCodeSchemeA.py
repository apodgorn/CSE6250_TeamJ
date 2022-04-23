import pandas as pd
import numpy as np
from numpy.random import choice
import recordlinkage as rl
from recordlinkage.preprocessing import phonetic
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import *
import matplotlib.pyplot as plt

RANDOM_STATE = 604

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def true_links(df):
    #generate true link object to match true_links properties of RL toolkit
    df["rec_id"] = df.index.values.tolist()
    id_list1 = []
    id_list2 = []
    for match_id in df["match_id"].unique():
        if match_id != -1:
            records = df.loc[df['match_id'] == match_id]
            for i in range(len(records)-1):
                for j in range(i+1, len(records)):
                    id_list1 = id_list1 + [records.iloc[i]["rec_id"]]
                    id_list2 = id_list2 + [records.iloc[j]["rec_id"]]
    links = pd.MultiIndex.from_arrays([id_list1,id_list2])
    return links

def false_links(df, size):
    #generate pseudo false pairs for training purpose
    df["rec_id"] = df.index.values.tolist()
    id_list1 = []
    id_list2 = []
    unique_match_id = df["match_id"].unique()
    for i in range(size):
            false_pair_ids = choice(unique_match_id, 2)
            candidate_1_cluster = df.loc[df['match_id'] == false_pair_ids[0]]
            candidate_1 = candidate_1_cluster.iloc[choice(range(len(candidate_1_cluster)))]
            candidate_2_cluster = df.loc[df['match_id'] == false_pair_ids[1]]
            candidate_2 = candidate_2_cluster.iloc[choice(range(len(candidate_2_cluster)))]    
            id_list1 = id_list1 + [candidate_1["rec_id"]]
            id_list2 = id_list2 + [candidate_2["rec_id"]]  
    links = pd.MultiIndex.from_arrays([id_list1,id_list2])
    return links

def extract_features(df, links):
    comp = rl.Compare()
    comp.string('given_name', 'given_name', method='jarowinkler', label='y_name')
    comp.string('given_name_soundex', 'given_name_soundex', method='jarowinkler', label='y_name_soundex')
    comp.string('given_name_nysiis', 'given_name_nysiis', method='jarowinkler', label='y_name_nysiis')
    comp.string('surname', 'surname', method='jarowinkler', label='y_surname')
    comp.string('surname_soundex', 'surname_soundex', method='jarowinkler', label='y_surname_soundex')
    comp.string('surname_nysiis', 'surname_nysiis', method='jarowinkler', label='y_surname_nysiis')
    comp.exact('street_number', 'street_number', label='y_street_number')
    comp.string('address_1', 'address_1', method='levenshtein', threshold=0.7, label='y_address1')
    comp.string('address_2', 'address_2', method='levenshtein', threshold=0.7, label='y_address2')
    comp.exact('postcode', 'postcode', label='y_postcode')
    comp.exact('day', 'day', label='y_day')
    comp.exact('month', 'month', label='y_month')
    comp.exact('year', 'year', label='y_year')
    
    return comp.compute(links,df,df)

def train_model(alg, hyperParam, x, y, modelParam):
    #train base learners
    #SVM
    if alg == 'svm':
        model = svm.SVC(C=hyperParam, kernel=modelParam, random_state=RANDOM_STATE)
        model.fit(x, y)
    #Logistic Regression
    elif alg == 'lg':
        model = LogisticRegression(C=hyperParam, penalty=modelParam, max_iter=5000, multi_class='ovr', random_state=RANDOM_STATE)
        model.fit(x, y)
    #Neural Network
    elif alg == 'nn':
        model = MLPClassifier(solver='lbfgs', alpha=hyperParam, hidden_layer_sizes=(256, ), activation=modelParam, max_iter=10000, random_state=RANDOM_STATE)
        model.fit(x, y)
    return model

def eval_model(test_labels, result):
    #currently using the same eval function used in the paper
    precision = precision_score(test_labels,result)
    sensitivity = recall_score(test_labels,result)
    confusionMatrix = confusion_matrix(test_labels,result)
    no_links_found = np.count_nonzero(result)
    no_false = np.sum(np.logical_and(np.logical_not(test_labels), result)) + np.sum(np.logical_and(test_labels,np.logical_not(result)))
    Fscore = f1_score(test_labels,result)
    metrics_result = {'no_false':no_false, 'confusion_matrix':confusionMatrix ,'precision':precision,
                     'sensitivity':sensitivity ,'no_links':no_links_found, 'F-score': Fscore}
    return metrics_result

def main():
    #training data
    trainpath = 'febrl3_UNSW.csv'
    df_train = pd.read_csv(trainpath,index_col="rec_id")
    train_trueLinks = true_links(df_train)
    df_train['postcode'] = df_train['postcode'].astype(str)
    df_train['given_name_soundex'] = phonetic(df_train['given_name'], method='soundex')
    df_train['given_name_nysiis'] = phonetic(df_train['given_name'], method='nysiis')
    df_train['surname_soundex'] = phonetic(df_train['surname'], method='soundex')
    df_train['surname_nysiis'] = phonetic(df_train['surname'], method='nysiis')
    ones = extract_features(df_train, train_trueLinks)
    train_falseLinks = false_links(df_train, len(train_trueLinks))    
    mones = extract_features(df_train, train_falseLinks)
    x_train = ones.values.tolist() + mones.values.tolist()
    y_train = [1]*len(ones)+[0]*len(mones)
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    #testing data
    testpath = 'febrl4_UNSW.csv'
    df_test = pd.read_csv(testpath, index_col="rec_id")
    #test_trueLinks = true_links(df_test)
    df_test['postcode'] = df_test['postcode'].astype(str)
    df_test['given_name_soundex'] = phonetic(df_test['given_name'], method='soundex')
    df_test['given_name_nysiis'] = phonetic(df_test['given_name'], method='nysiis')
    df_test['surname_soundex'] = phonetic(df_test['surname'], method='soundex')
    df_test['surname_nysiis'] = phonetic(df_test['surname'], method='nysiis')
    
    blocking_fields = ["given_name", "surname", "postcode"]
    test_links = []
    for field in blocking_fields:
        block_indexer = rl.BlockIndex(on=field)
        lnks = block_indexer.index(df_test)
        test_links = lnks.union(test_links)
    df_x_test = extract_features(df_test, test_links)
    x_test = df_x_test.values.tolist()
    y_test = [0]*len(x_test)
    x_test_index = df_x_test.index
    for i in range(0, len(x_test_index)):
        if df_test.loc[x_test_index[i][0]]["match_id"]==df_test.loc[x_test_index[i][1]]["match_id"]:
            y_test[i] = 1
    x_test, y_test = shuffle(x_test, y_test, random_state=0)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    #base learners
    print("_______BASE LEARNERS PERFORMANCE______")
    alg = 'svm'# using 'svm', 'lg' or 'nn'
    modelParam = 'linear'# replace with 'linear' or 'rbf' for svm, 'l1' or 'l2' for lg, 'relu' or 'logistic' for nn
    param_range = [0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
    # parametric run results
    # 0.02 best C param for rbf-svm
    # 0.005 best C param for linear-svm (linear is better than rbf)
    # 0.02 best param for l2-lg
    # 100 best alpha for relu-NN
    precision = []
    sensitivity = []
    Fscore = []
    num_false = []
    
    for hyperParam in param_range:
        model = train_model(alg, hyperParam, x_train, y_train, modelParam)
        y_pred = model.predict(x_test)
        test_eval = eval_model(y_test, y_pred)
        precision += [test_eval['precision']]
        sensitivity += [test_eval['sensitivity']]
        Fscore += [test_eval['F-score']]
        num_false  += [test_eval['no_false']]
        
    print("False links:",num_false,"\n")
    print("Precision:",precision,"\n")
    print("Sensitivity:",sensitivity,"\n")
    print("F-score:", Fscore,"\n")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    title = "Learning Curves Best NN, relu"
    cv = cv = KFold(n_splits=10)
    estimator = MLPClassifier(solver='lbfgs', alpha=100, hidden_layer_sizes=(256, ),activation='relu', max_iter=10000, random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, x_train, y_train, axes=axes[:,0], ylim=(0.98, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves Best SVM, linear"
    cv = cv = KFold(n_splits=10)   
    estimator = svm.SVC(C=0.005, kernel='linear', random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, x_train, y_train, axes=axes[:,1], ylim=(0.98, 1.01), cv=cv, n_jobs=4
    )
    
    title = "Learning Curves Best Lg, l2"
    cv = cv = KFold(n_splits=10)   
    estimator = LogisticRegression(C=0.02, penalty='l2', max_iter=5000, multi_class='ovr', random_state=RANDOM_STATE)
    plot_learning_curve(
        estimator, title, x_train, y_train, axes=axes[:,2], ylim=(0.98, 1.01), cv=cv, n_jobs=4
    )
    
    plt.savefig('best base learners.png')
    
    
    #meta-ensemble
    #bagging
    algs = ['svm', 'nn', 'lg']    
    modelParams= ['linear', 'relu', 'l2']#optimal hyperparameters for the base learners
    hyperParams = [0.005, 100, 0.02]
    n = 10
    kf = KFold(n_splits=n)
    model_bagging_score = [0]*3
    for i in range(3):
        alg = algs[i]
        modelParam = modelParams[i]
        hyperParam = hyperParams[i]
        iFold = 0
        result_fold = [0]*n
        final_eval_fold = [0]*n
        for train_index, valid_index in kf.split(x_train):
            x_train_fold = x_train[train_index]
            y_train_fold = y_train[train_index]
            md = train_model(alg, hyperParam, x_train_fold, y_train_fold, modelParam)
            result_fold[iFold] = md.predict(x_test)
            final_eval_fold[iFold] = eval_model(y_test, result_fold[iFold])
            iFold = iFold + 1
        bagging_score = np.average(result_fold, axis=0)
        model_bagging_score[i] = bagging_score
        
    #stacking
    thres = .99
    stack_raw_score = np.average(model_bagging_score, axis=0)
    stack_result = np.copy(stack_raw_score)
    stack_result[stack_result > thres] = 1
    stack_result[stack_result <= thres] = 0
    
    ensemble_eval = eval_model(y_test, stack_result)    
    print("_______Ensemble Performance______")
    print(ensemble_eval)
    
    return

if __name__ == "__main__":
    main()

