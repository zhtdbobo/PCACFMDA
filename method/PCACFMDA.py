
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from deepforest import CascadeForestClassifier
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from method.util import transfer_label_from_prob, ad_compute_indexes
import scipy.io

def train(X_train, y_train, args):
    X_train = np.array(X_train)

    num_cross = args.fold
    all_performance = []
    probaresult = []
    ae_y_pred_probresult = []

    mean_tpr = 0.0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    y_real1 = []
    y_proba1 = []


    colors = ['blue', 'orange', 'lime', 'purple', 'green']
    linestyles = ['--'] * 5
    result_tpr = []
    result_fpr = []
    result_auc = []
    result_precision = []
    result_recall = []
    result_aupr = []
    for fold in range(num_cross):
        print(fold)
        train_X = np.array([x for i, x in enumerate(X_train) if i % num_cross != fold])
        test_X = np.array([x for i, x in enumerate(X_train) if i % num_cross == fold])

        train_label = np.array([x for i, x in enumerate(y_train) if i % num_cross != fold])
        train_Y = []
        for val in train_label:
            if val[0] == 1:
                train_Y.append(0)
            else:
                train_Y.append(1)

        test_label = np.array([x for i, x in enumerate(y_train) if i % num_cross == fold])
        test_Y = []
        for val in test_label:
            if val[0] == 1:
                test_Y.append(0)
            else:
                test_Y.append(1)

        clf = CascadeForestClassifier(random_state=1, verbose=0, n_jobs=-1)
        estimators = [
            RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=50),
            RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=50),
            AdaBoostClassifier(random_state=1, n_estimators=50),
            CatBoostClassifier(random_state=1, verbose=0, n_estimators=50, allow_writing_files=False)
        ]
        clf.set_estimator(estimators)
        predictor = SVC(random_state=1, probability=True, kernel='poly')
        clf.set_predictor(predictor)

        print('---' * 20)
        clf.fit(train_X, train_Y)
        y_pred_prob = clf.predict_proba(test_X)[:, 1]
        y_pred = transfer_label_from_prob(y_pred_prob)
        y_test = test_Y

        ae_y_pred_probresult.extend(y_pred_prob)
        probaresult.extend(y_pred)

        tp, fn, fp, tn = metrics.confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()
        acc, precision, sensitivity, specificity, MCC, f1_score, aupr_score = ad_compute_indexes(tp, fp, tn, fn, y_test,
                                                                                                 y_pred_prob)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
        auc = metrics.auc(fpr, tpr)
        result_fpr.append(fpr)
        result_tpr.append(tpr)
        result_auc.append(auc)
        print('---' * 20)
        print(metrics.confusion_matrix(y_test, y_pred, labels=[1, 0]))
        print(acc, precision, sensitivity, specificity, MCC, f1_score, aupr_score, auc)

        all_performance.append([acc, precision, sensitivity, specificity, MCC, f1_score, aupr_score, auc])

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc)

        precision2, recall2, _2 = metrics.precision_recall_curve(y_test, y_pred_prob)
        result_precision.append(precision2)
        result_recall.append(recall2)
        result_aupr.append(aupr_score)
        scipy.io.savemat(args.result_path + 'PCACFMDA_AUPR_%d.mat' % (fold)
                         , {'recall': recall2, 'precision': precision2, 'mean_aupr': aupr_score})
        y_real1.append(y_test)
        y_proba1.append(y_pred_prob)
        # AUC
        scipy.io.savemat(args.result_path + 'PCACFMDA_AUC_%d.mat' % (fold)
                         , {'fpr': fpr, 'tpr': tpr, 'mean_auc': auc})

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    scipy.io.savemat(args.result_path + 'PCACFMDA_AUC.mat'
                     , {'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'mean_auc': mean_auc})

    y_real1 = np.concatenate(y_real1)
    y_proba1 = np.concatenate(y_proba1)
    precision3, recall3, _3 = metrics.precision_recall_curve(y_real1, y_proba1)

    Mean_Result = np.mean(np.array(all_performance), axis=0)
    std_auc = np.std(tprs, axis=0)
    # plt.plot(recall3,precision3,color='b',label=r'Mean ROC (area=%0.4f)'%Mean_Result[7],lw=2,alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(1, 2, 1)
        if i != 5:
            plt.plot(result_fpr[i], result_tpr[i], lw=1, alpha=0.8, color=colors[i], linestyle=linestyles[i],
                     label=r'fold %d (AUC = %0.4f)' % (i + 1, result_auc[i]))
        else:
            plt.plot(mean_fpr, mean_tpr, lw=2, color='r', label=r'Mean (AUC = %0.4f)' % Mean_Result[7])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        if i != 5:
            plt.plot(result_recall[i], result_precision[i], lw=1, alpha=0.8, color=colors[i], linestyle=linestyles[i],
                     label=r'fold %d (AUPR = %0.4f)' % (i + 1, result_aupr[i]))
        else:
            plt.plot(recall3, precision3, lw=2, color='r', label='Mean (AUPR = %0.4f)' % Mean_Result[6])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
    plt.show()

    print('---' * 20)
    print('Mean-acc=', Mean_Result[0], '\n Mean-precision=', Mean_Result[1])
    print('Mean-sensitivity=', Mean_Result[2], '\n Mean-specificity=', Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4], '\n Mean-f1_score=', Mean_Result[5])

    print('Mean-aupr_score=', Mean_Result[6], '\n Mean-auc=', Mean_Result[7])

    stu = ['acc', 'precision', 'sensitivity', 'specificity', 'MCC', 'f1_score', 'aupr_score', 'auc']

    results = []
    results.append({
        'FPR': mean_fpr,
        'TPR': mean_tpr,
        'AUC': Mean_Result[7],
        'Recall': recall3,
        'Precision': precision3,
        'AUPR': Mean_Result[6]
    })
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.result_path + 'result_draw.csv', index=False)

    scipy.io.savemat(args.result_path + 'PCACFMDA_AUPR.mat'
                     , {'recall': recall3, 'precision': precision3, 'mean_aupr': Mean_Result[6]})
    Mean_Result = pd.DataFrame(Mean_Result, index=stu)
    Mean_Result.to_csv(args.result_path + 'PCACFMDA.csv', sep=',', index=True, header=False)


def test(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = CascadeForestClassifier(random_state=1, verbose=0, n_jobs=-1)
    estimators = [
        RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=50),
        RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=50),
        AdaBoostClassifier(random_state=1, n_estimators=50),
        CatBoostClassifier(random_state=1, verbose=0, n_estimators=50, allow_writing_files=False)
    ]
    clf.set_estimator(estimators)
    predictor = SVC(random_state=1, probability=True, kernel='poly')
    clf.set_predictor(predictor)

    probaresult = []
    ae_y_pred_probresult = []

    train_X = np.array(X_train)
    train_label = np.array(y_train)
    train_label_new = []  # 训练标签
    for val in train_label:
        if val[0] == 1:
            train_label_new.append(0)
        else:
            train_label_new.append(1)

    clf.fit(train_X, train_label_new)

    test_X = np.array(X_test)
    test_label = np.array(y_test)
    test_label_new = []  # 训练标签
    for val in test_label:
        if val[0] == 1:
            test_label_new.append(0)
        else:
            test_label_new.append(1)

    X_y_pred_prob = clf.predict_proba(test_X)[:, 1]
    Xproba = transfer_label_from_prob(X_y_pred_prob)

    proba = Xproba

    # 预测的最终概率，使用平均法
    ae_y_pred_prob = X_y_pred_prob
    # 结果保存
    probaresult.extend(proba)
    ae_y_pred_probresult.extend(ae_y_pred_prob)

    y_test = test_label_new
    y_pred = transfer_label_from_prob(ae_y_pred_prob)

    all_performance = []
    tprs = []
    aucs = []
    # 混淆矩阵
    tp, fn, fp, tn = metrics.confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()

    acc, precision, sensitivity, specificity, MCC, f1_score, aupr_score = ad_compute_indexes(tp, fp, tn, fn, y_test,
                                                                                             ae_y_pred_prob)
    fpr, tpr, _ = metrics.roc_curve(test_label_new, ae_y_pred_prob)
    auc = metrics.auc(fpr, tpr)

    print('---' * 20)
    print(metrics.confusion_matrix(y_test, y_pred, labels=[1, 0]))
    print(acc, precision, sensitivity, specificity, MCC, f1_score, aupr_score, auc)

    all_performance.append([acc, precision, sensitivity, specificity, MCC, f1_score, aupr_score, auc])

    # AUPR
    precision2, recall2, _2 = metrics.precision_recall_curve(test_label_new, ae_y_pred_prob)
    scipy.io.savemat(args.result_path + 'AUPR_PCACFMDA.mat'
                     , {'recall': recall2, 'precision': precision2, 'mean_aupr': aupr_score})

    scipy.io.savemat(args.result_path + 'AUC_PCACFMDA.mat'
                     , {'fpr': fpr, 'tpr': tpr, 'mean_auc': auc})
