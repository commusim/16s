import csv
import argparse

# 文件路径
input_file_path = '/zangzelin/data/guhe-microbiomedata/stage1sampledata-WL-clean.txt' # 输入的 txt 文件
output_file_path = '/zangzelin/data/guhe-microbiomedata/out_meta.csv'     # 输出的 csv 文件

def filter_label(label_path, filter_label_path):
    with open(label_path, 'r', encoding='utf-8') as infile, \
        open(filter_label_path, 'w', newline='', encoding='utf-8') as outfile:
        
        # 创建 CSV 写入器
        writer = csv.writer(outfile)
        
        # 逐行处理数据
        for iter,line in enumerate(infile):
            # 按制表符分割每行数据
            # print(id)
            parts = line.strip().split('\t')
            id,label = parts[0],parts[5]
            try:
                # 如果标签符合 年龄
                int(label)
                writer.writerow([id,label])  # 写入符合条件的行
            except ValueError:
                # 如果转换失败，跳过该行
                continue

def csv2dict(csv_path):
    dict = {}
    with open(csv_path, mode='r', encoding='utf-8') as label_file:
        reader = csv.reader(label_file)
        for row in reader:
            if row:  # 确保行不为空
                dict[row[0]] = row[1] 
    return dict

def filter_feature(filter_dict, input_path, out_path):
    filtered_count = 0
    total_count = 0
    # 打开输入文件并逐行读取
    with open(input_path, 'r', encoding='utf-8') as infile, \
        open(out_path, 'w', newline='', encoding='utf-8') as outfile:    
        # 创建 CSV 写入器
        writer = csv.writer(outfile)
        
        # 逐行处理数据
        for line in infile:
            total_count += 1  # 统计总行数
            print(total_count)
            # 按制表符分割每行数据
            parts = line.strip().split('\t')
            # 确保每行有正确的列数（第一列为 ID，后续为特征）
            if len(parts) > 98823:
                # 如果列数不足，补齐缺失的列（用空字符串或0填充）    
                # 写入表头（可选）
                header = parts[:98823]
                header[98822] = header[98822][:5]
                parts_data = parts[-98823:]
                parts_data[0] = parts_data[0][5:]
                # writer.writerow(header)

                current_id = parts_data[0]  # 当前行的 id
                
                # 检查当前 id 是否存在于 label_dict 中
                if current_id in filter_dict:
                    filtered_count += 1  # 统计筛选的行数
                    new_row = [filter_dict[current_id],*parts_data]  # 添加对应的 label
                    writer.writerow(new_row)

            # 确保每行有正确的列数（第一列为 ID，后续为特征）
            if len(parts) < 98823:
                # 如果列数不足，补齐缺失的列（用空字符串或0填充）
                parts += ['0'] * (98823 - len(parts))
            if(len(parts) == 98823):
                current_id = parts[0]
                # 写入 CSV 文件
                if current_id in filter_dict:
                    filtered_count += 1  # 统计筛选的行数
                    new_row = [filter_dict[current_id],*parts]  # 添加对应的 label
                    writer.writerow(new_row)
                # print(parts[0])

    print(f"数据已成功保存到 {output_file_path}")
    # 输出统计信息
    print(f"总行数: {total_count}")
    print(f"筛选行数: {filtered_count}")
    if total_count > 0:
        print(f"筛选比例: {filtered_count / total_count * 100:.2f}%")
    else:
        print("总行数为 0，无法计算比例。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Annotation Script")
    parser.add_argument('--input', type=str, default="../source_data/seurat_high2_data.h5ad",
    # parser.add_argument('--input', type=str, default="Exp1/05_cell_annotated/seurat_high2_data.h5ad",
                        help='Path to the input CSV file containing cell cluster markers.')
    parser.add_argument('--output', type=str, default="DMT_epi", help='Path to save the annotated cell data table.')
    parser.add_argument('--need_process', default=True, help='Whether the data need process.')
    parser.add_argument('--methods', default="dmt",choices=["pca","dmt"], help='Whether the data need process.')
    parser.add_argument('--seed', default=0)
    # parser.add_argument('--adata', type=str, required=True, help='Path to the AnnData object file for visualization.')

    args = parser.parse_args()

    filter_label()
    csv2dict()
    filter_feature()