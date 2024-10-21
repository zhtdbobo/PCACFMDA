import numpy as np
from matplotlib import pyplot as plt
from numpy import matlib
from sklearn import preprocessing
from keras import utils
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
def prepare_data(data_path):
    D_SSM1 = np.loadtxt(data_path + 'D_SSM1.txt')
    D_SSM2 = np.loadtxt(data_path + 'D_SSM2.txt')
    D_GSM = np.loadtxt(data_path + 'D_GSM.txt')
    M_FSM = np.loadtxt(data_path + 'M_FSM.txt')
    M_GSM = np.loadtxt(data_path + 'M_GSM.txt')
    D_SSM = (D_SSM1 + D_SSM2) / 2

    miRNA_dimension = M_FSM.shape[0]
    disease_dimension = D_SSM.shape[0]

    ID = np.zeros(shape=(disease_dimension, disease_dimension))
    IM = np.zeros(shape=(miRNA_dimension, miRNA_dimension))
    for i in range(disease_dimension):
        for j in range(disease_dimension):
            if D_SSM[i][j] == 0:
                ID[i][j] = D_GSM[i][j]
            else:
                ID[i][j] = (D_SSM[i][j] + D_GSM[i][j]) / 2
    for i in range(miRNA_dimension):
        for j in range(miRNA_dimension):
            if M_FSM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]
            else:
                IM[i][j] = (M_FSM[i][j] + M_GSM[i][j]) / 2

    print("loading data")

    A = np.loadtxt(data_path + "interaction.txt", dtype=int, delimiter=" ")
    interacation = np.transpose(A)

    R_B = matlib.repmat(interacation, miRNA_dimension, 1)
    sm = np.repeat(IM, repeats=disease_dimension, axis=0)
    train1 = np.concatenate((sm, R_B), axis=1)

    R_A = np.repeat(A, repeats=disease_dimension, axis=0)
    sd = matlib.repmat(ID, miRNA_dimension, 1)
    train2 = np.concatenate((R_A, sd), axis=1)
    label = A.reshape((miRNA_dimension * disease_dimension, 1))

    return train1, train2, label


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = utils.np_utils.to_categorical(y)
    return y, encoder

def feature_engineering(XY):
    feature_dimensen = XY.shape[1] - 1
    y = XY.iloc[:, feature_dimensen]

    fraud_indices = y[y == 1].index
    normal_indices = y[y == 0].index

    random_normal_indices = np.array(np.random.choice(normal_indices, len(y[y == 1]), replace=False))
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    under_sample_data = XY.iloc[under_sample_indices, :]
    under_sample_data = shuffle(under_sample_data)

    XX = under_sample_data.iloc[:, 0:feature_dimensen]
    yy = under_sample_data.iloc[:, feature_dimensen]
    yy = np.array(yy)


    yy, encoder = preprocess_labels(yy)
    ny = np.arange(len(yy))
    yy = yy[ny]
    return XX, yy

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label
def ad_compute_indexes(tp, fp, tn, fn,y_test,y_prob_score):
    from sklearn import metrics
    precision_1, recall_1, pr_threshods = metrics.precision_recall_curve(y_test,y_prob_score,pos_label=1)
    aupr_score = metrics.auc(recall_1, precision_1)
    acc = float(tp + tn) / (tp+tn+fp+fn)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    precision = float(tp) / (tp + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
    return acc, precision, sensitivity, specificity, MCC, f1_score,aupr_score

def draw_PCA(X, args):
    pca = PCA()
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    pca_output_dim = np.argmax(cumulative_variance >= args.pca_cumulative_variance / 100) + 1
    pca100 = np.argmax(cumulative_variance >= 0.99) + 100

    if pca_output_dim < X.shape[1]:
        pca = PCA(n_components=pca_output_dim)
        X = pca.fit_transform(X)
    else:
        print("原始数据维度已经低于或等于目标累积方差贡献率对应的主成分数量。")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, pca100), cumulative_variance[0:pca100 - 1] * 100, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')

    plt.axhline(y=95, color='r', linestyle='--', label=f'{args.pca_cumulative_variance}% Explained Variance Threshold')
    plt.text(pca_output_dim + 1, 92, pca_output_dim)
    plt.xticks(np.arange(0, pca100, 20))
    plt.grid(True)
    plt.legend()

    # plt.savefig(args.result_path + 'fig2.jpg', dpi=300)
    # plt.show()

    return X