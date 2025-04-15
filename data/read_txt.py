import csv

# 文件路径
input_file_path = '/zangzelin/data/guhe-microbiomedata/stage1sampledata-WL-clean.txt' # 输入的 txt 文件
output_file_path = '/zangzelin/data/guhe-microbiomedata/dataset.csv'     # 输出的 csv 文件


label_csv_path = '/root/commusim/16s/output.csv'   # 包含 label 的 CSV 文件
# 加载 label.csv 中的所有 id 和对应的 label 到一个字典中
label_dict = {}
with open(label_csv_path, mode='r', encoding='utf-8') as label_file:
    reader = csv.reader(label_file)
    next(reader)  # 跳过表头（如果有的话）
    for row in reader:
        if row:  # 确保行不为空
            label_dict[row[0]] = row[1]  # 假设第一列为 id，第二列为 label
import pdb; pdb.set_trace()
# 统计筛选的行数和总行数
filtered_count = 0
total_count = 0

# 打开输入文件并逐行读取
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:    
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
            if current_id in label_dict:
                filtered_count += 1  # 统计筛选的行数
                new_row = [label_dict[current_id],*parts_data]  # 添加对应的 label
                writer.writerow(new_row)

        # 确保每行有正确的列数（第一列为 ID，后续为特征）
        if len(parts) < 98823:
            # 如果列数不足，补齐缺失的列（用空字符串或0填充）
            parts += ['0'] * (98823 - len(parts))
        if(len(parts) == 98823):
            current_id = parts[0]
            # 写入 CSV 文件
            if current_id in label_dict:
                filtered_count += 1  # 统计筛选的行数
                new_row = [label_dict[current_id],*parts]  # 添加对应的 label
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