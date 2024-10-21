import argparse
import os
from datetime import datetime

from sklearn.model_selection import train_test_split

from PCACFMDA import train,test
import numpy as np
import pandas as pd
from method.util import prepare_data, feature_engineering, draw_PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../datasets495/', help='Path to the data directory')
    parser.add_argument('--result_path', type=str, default='../result/495/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/', help='Path to the result directory')
    parser.add_argument('--pca_cumulative_variance', type=int, default=95, help='Cumulative variance contribution')
    parser.add_argument('--fold', type=int, default=5, help='cross-validation')

    args = parser.parse_args()

    os.makedirs(args.result_path)
    mtrain, dtrain, label = prepare_data(args.data_path)

    X, y = feature_engineering(pd.concat([pd.DataFrame(mtrain), pd.DataFrame(dtrain), pd.DataFrame(label)], axis=1))

    X = draw_PCA(X, args)
    X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    train(X_train, y_train, args)
    test(X, y, args)

if __name__ == '__main__':
    main()