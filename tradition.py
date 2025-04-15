import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data import CustomNpyDataset
import time
import argparse
import numpy as np

def select_highly_variable_features(features, top_k=500):
    """
    选择高变特征。
    :param features: 输入特征矩阵 (numpy 数组)
    :param top_k: 保留的高变特征数量
    :return: 筛选后的高变特征矩阵
    """
    # 计算每个特征的方差
    variances = np.var(features, axis=0)
    # 找到方差最大的 top_k 个特征的索引
    top_k_indices = np.argsort(variances)[-top_k:]
    # 提取高变特征
    selected_features = features[:, top_k_indices]
    return selected_features

def log_transform(features):
    """
    对非零值进行对数变换。
    :param features: 输入特征矩阵 (numpy 数组)
    :return: 转换后的特征矩阵
    """
    features = features.copy()
    features[features > 0] = np.log1p(features[features > 0])  # 对非零值取 log1p
    return features

def min_max_normalize(features):
    """
    使用 Min-Max 归一化。
    :param features: 输入特征矩阵 (numpy 数组)
    :return: 归一化后的特征矩阵
    """
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def z_score_standardize(features):
    """
    使用 Z-Score 标准化。
    :param features: 输入特征矩阵 (numpy 数组)
    :return: 标准化后的特征矩阵
    """
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return standardized_features

def preprocess_features(features, top_k=500, normalize_method="minmax"):
    """
    综合预处理流程：高变特征选择 -> 对数变换 -> 正则化。
    :param features: 输入特征矩阵 (numpy 数组)
    :param top_k: 保留的高变特征数量
    :param normalize_method: 正则化方法 ("minmax" 或 "zscore")
    :return: 预处理后的特征矩阵
    """
    # 1. 高变特征选择
    hvf_features = select_highly_variable_features(features, top_k=top_k)
    
    # 2. 对数变换
    log_hvf_features = log_transform(hvf_features)
    
    # 3. 正则化
    if normalize_method == "minmax":
        preprocessed_features = min_max_normalize(log_hvf_features)
    elif normalize_method == "zscore":
        preprocessed_features = z_score_standardize(log_hvf_features)
    else:
        raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    return preprocessed_features


sample = np.load("out/dataset.npy")
# 定义特征分组的列索引范围
label_col = 0  # 第一列为标签
id_col = 1     # 第二列为ID
diversity_cols = slice(2, 6)  # 第2-5列为多样性特征
metabolism_gene_cols = slice(6, 7243)  # 第6-7242列为代谢和基因特征
direct_microbe_cols = slice(7243, 82732)  # 第7243-82731列为直接检测到的菌特征
species_abundance_cols = slice(82732, 85881)  # 第82732-85880列为物种层级丰度特征
synthetic_features_cols = slice(85881, 85974)  # 第85881-85973列为合成特征
genus_species_cols = slice(85974, 88351)  # 第85974-88350列为属和种的物种层级丰度特征
pathway_cols = slice(88351, 98819)  # 第88351-98818列为基因和代谢通路特征
pathogen_cols = slice(98819, 98824)  # 后续补充的病原菌检测丰度特征


label = sample[:,label_col] 
id_ = sample[:,id_col]
# 提取各特征分组
diversity_features = sample[:,diversity_cols] # 1/4是0
metabolism_gene_features = sample[:,metabolism_gene_cols] # 大约1/3的数据是0
direct_microbe_features = sample[:,direct_microbe_cols] #大多都是0
species_abundance_features = sample[:,species_abundance_cols] #大多都是0
synthetic_features = sample[:,synthetic_features_cols] #1/100是0
genus_species_features = sample[:,genus_species_cols] #大多都是0
pathway_features = sample[:,pathway_cols] # 大约1/3的数据是0
pathogen_features = sample[:,pathogen_cols] #大多数是补0

X_all = [diversity_features, metabolism_gene_features, direct_microbe_features, species_abundance_features, synthetic_features, genus_species_features, pathway_features, pathogen_features]

def main(args):
    X = X_all[args.i]
    preprocess = args.preprocess
    print(preprocess)
    if preprocess:
        X = preprocess_features(X)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import numpy as np
    # # 设置图形风格
    # sns.set(style="whitegrid")
    # # 绘制直方图
    # plt.figure(figsize=(10,6))
    # sns.histplot(label, bins=30, kde=True, color="skyblue", stat="density", linewidth=0)
    # plt.title('Label Distribution with Seaborn')
    # plt.xlabel('Label Value')
    # plt.ylabel('Density')
    # # 显示图形
    # plt.savefig("./label.png")
    # import pdb; pdb.set_trace()
    eval = False
    def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        training_time = end_time - start_time
        
        import pdb; pdb.set_trace()
        # 打印结果
        print(f"{model_name}:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")
        print(f"Training and Prediction Time: {training_time:.4f} seconds\n")
        
        # 返回结果
        return {
            "Model": model_name,
            "MSE": mse,
            "R2": r2,
            "Time": training_time
        }

    # 1. 线性回归模型
    linear_result = train_and_evaluate(LinearRegression(), "Linear Regression", X_train, X_test, y_train, y_test)

    # 2. 随机森林回归模型
    rf_result = train_and_evaluate(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                                    "Random Forest Regressor", X_train, X_test, y_train, y_test)

    # 3. 决策树回归模型
    dt_result = train_and_evaluate(DecisionTreeRegressor(random_state=42), "Decision Tree Regressor",
                                    X_train, X_test, y_train, y_test)

    # 将结果保存到 txt 文件中
    with open("model_results.txt", "a") as file:
        file.write("Model Evaluation Results:\n")
        file.write("=" * 50 + "\n")
        
        for result in [linear_result, rf_result, dt_result]:
            file.write(f"Model: {result['Model']}\n")
            file.write(f"Mean Squared Error (MSE): {result['MSE']:.4f}\n")
            file.write(f"R^2 Score: {result['R2']:.4f}\n")
            file.write(f"Training and Prediction Time: {result['Time']:.4f} seconds\n")
            file.write("-" * 50 + "\n")

    print("Results saved to 'model_results.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Annotation Script")
    parser.add_argument('--i', type=int, default=0,
                        help='Path to the input CSV file containing cell cluster markers.')
    parser.add_argument('--preprocess', type=bool, default=True)

    args = parser.parse_args()

    main(args)
