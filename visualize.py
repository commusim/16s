import numpy as np
import pandas as pd
from data import CustomNpyDataset
import time
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from umap import UMAP
from data import CustomNpyDataset
import matplotlib.pyplot as plt
import seaborn as sns
from dmt_learn.dmt_learn import DMTLearn


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

def visualize(X, labels, title="UMAP Visualization"):
    """
    使用 UMAP 进行降维并可视化。
    :param X: 输入特征矩阵 (numpy 数组)
    :param labels: 对应的标签 (用于着色)
    :param title: 图表标题
    """
    # 初始化 UMAP 模型
    # reducer = UMAP(n_components=2, random_state=42)
    reducer = DMTLearn(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)  # 将数据降维到 2D

    # 创建 DataFrame 以便于绘图
    df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    df["Label"] = labels

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="Label",
        palette="viridis",
        data=df,
        legend="full",
        alpha=0.7,
        s=10
    )
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"./{title.replace(' ', '_')}.png")  # 保存图像
    plt.show()


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
names = ["diversity", "metabolism_gene", "direct_microbe", "species_abundance", "synthetic", "genus_species", "pathway", "pathogen"]

def main(args):

    X = X_all[args.i]
    X = preprocess_features(X)
    visualize(X,label,names[args.i]+"dmt")

    eval = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Annotation Script")
    parser.add_argument('--i', type=int, default=2,
                        help='Path to the input CSV file containing cell cluster markers.')
    parser.add_argument('--preprocess', type=bool, default=True)

    args = parser.parse_args()

    main(args)
