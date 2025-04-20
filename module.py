
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def discretize_labels(labels, num_classes=100):
    """
    将连续标签离散化为 0 到 num_classes-1 的类别。
    :param labels: 连续标签 (numpy 数组)
    :param num_classes: 类别数量
    :return: 离散化的类别标签
    """
    bins = np.linspace(0, 100, num_classes + 1)  # 创建等宽区间
    discrete_labels = np.digitize(labels, bins) - 1  # 将标签映射到区间索引
    return discrete_labels

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

X_all = [diversity_features, metabolism_gene_features, direct_microbe_features, species_abundance_features, synthetic_features, genus_species_features, pathway_features, pathogen_features, label]

# 提取各特征分组并存入字典
features_dict = {
    "diversity": sample[:, diversity_cols],  # 多样性特征
    "metabolism_gene": sample[:, metabolism_gene_cols],  # 代谢和基因特征
    "direct_microbe": sample[:, direct_microbe_cols],  # 直接检测到的菌特征
    "species_abundance": sample[:, species_abundance_cols],  # 物种层级丰度特征
    "synthetic": sample[:, synthetic_features_cols],  # 合成特征
    "genus_species": sample[:, genus_species_cols],  # 属和种的物种层级丰度特征
    "pathway": sample[:, pathway_cols],  # 基因和代谢通路特征
    "pathogen": sample[:, pathogen_cols]  # 病原菌检测丰度特征
}