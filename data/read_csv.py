import csv

# # 文件路径
# input_file_path = '/zangzelin/data/guhe-microbiomedata/stage1metainfo-WL.txt' # 输入的 txt 文件
# output_file_path = 'label.csv'     # 输出的 csv 文件

# 打开输入文件并逐行读取
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
    
    # 创建 CSV 写入器
    writer = csv.writer(outfile)
    
    # 逐行处理数据
    for id,line in enumerate(infile):
        # 按制表符分割每行数据
        print(id)
        parts = line.strip().split('\t')
        writer.writerow([parts[0],parts[5]])

# print(f"数据已成功保存到 {output_file_path}")

# 输入文件路径
input_file_path = '/root/commusim/16s/label.csv'
# 输出文件路径
output_file_path = '/root/commusim/16s/output.csv'

# 打开输入文件进行读取
with open(input_file_path, mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    
    # 打开输出文件进行写入
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 遍历每一行
        for row in reader:
            if row:  # 确保行不为空
                try:
                    # 尝试将第一列的值转换为整数
                    int(row[1])
                    writer.writerow(row)  # 写入符合条件的行
                except ValueError:
                    # 如果转换失败，跳过该行
                    continue