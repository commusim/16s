import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import time
import argparse
import numpy as np

from module import preprocess_features, discretize_labels, X_all

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def train_and_evaluate_classification(model, model_name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # 加权平均 F1 分数
    training_time = end_time - start_time

    # 打印结果
    print(f"{model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cost Time: {training_time:.4f}s\n")

    # 混淆矩阵和分类报告
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 返回结果
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Time": training_time,
        "Confusion Matrix": cm,
        "Classification Report": report
    }

def train_and_evaluate_regression(model, model_name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    training_time = end_time - start_time
    # 打印结果
    print(f"{model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Cost Time: {training_time:.4f}s\n")
    
    # 返回结果
    return {
        "Model": model_name,
        "MSE": mse,
        "R2": r2,
        "Time": training_time
    }




def main(args):

    label = X_all[-1]
    X = np.empty((X_all[0].shape[0],0)) 
    for index in args.i:
        X_new = X_all[int(index)]
        preprocess = args.preprocess
        # print(preprocess)
        if preprocess:
            X_new = preprocess_features(X_new)
        if X.size == 0:  # 如果 X 是空的，直接赋值
                X = X_new
        else:
            X = np.concatenate([X, X_new], axis=1)
    
    if args.mode == "classification":
        label = discretize_labels(label, num_classes=5)
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)


    # 定义模型和超参数
    if args.mode == "classification":
        if args.method == "RF":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
        elif args.method == "DT":
            model = DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
        elif args.method == "XG":
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        save_dict = train_and_evaluate_classification(model, args.method, X_train, X_test, y_train, y_test)

    elif args.mode == "regression":
        if args.method == "LL":
            model = LinearRegression()
        elif args.method == "RF":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
        elif args.method == "DT":
            model = DecisionTreeRegressor(
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
        elif args.method == "XG":
            model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        save_dict = train_and_evaluate_regression(model, args.method, X_train, X_test, y_train, y_test)



    def save_dict_to_txt(data, file_path):
        """
        将任意字典保存到文本文件中。
        :param data: 要保存的字典
        :param file_path: 文件路径
        """
        def write_dict(file, dictionary, indent=""):
            """
            递归写入字典内容。
            :param file: 文件对象
            :param dictionary: 当前要写入的字典
            :param indent: 缩进字符串（用于递归时的层次感）
            """
            for key, value in dictionary.items():
                file.write(f"{indent}{key}: {value}\n")

        with open(file_path, "a") as file:
            file.write("Dictionary Contents:\n")
            file.write("=" * 50 + "\n")
            write_dict(file, data)
            file.write("-" * 50 + "\n")

    save_dict_to_txt(save_dict,f"model.txt")
    # print("Results saved to 'model_results.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Annotation Script")
    parser.add_argument('--i', type=list, default=[0,1,2,3,4,5,6,7],
                        help='Path to the input CSV file containing cell cluster markers.')
    parser.add_argument('--preprocess', type=bool, default=True)
    parser.add_argument('--mode', type=str, default="regression")
    parser.add_argument('--method', type=str, default="DT")
    args = parser.parse_args()

    main(args)

# XG:
# Accuracy: 0.1427
# F1 Score: 0.1256
# Cost Time: 792.4708s
# All data
# Accuracy: 0.1502
# F1 Score: 0.1320
# Cost Time: 1897.6569s
# XG:
# MSE: 197.9109
# R^2: 0.2091
# Cost Time: 126.7366s
# 1000; 50
# XG:
# MSE: 227.3182
# R^2: 0.0916
# Cost Time: 1086.2842s


# RF:
# Accuracy: 0.1483
# F1 Score: 0.1111
# Cost Time: 171.8737s

# DT:
# Accuracy: 0.1156
# F1 Score: 0.1102
# Cost Time: 65.2423s