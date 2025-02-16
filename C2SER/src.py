import json

# 输入文件路径
input_file = "./EMO-Emilia/EMO-Emilia-ALL.jsonl"
# 输出文件路径
output_file = "./EMO-Emilia/EMO-Emilia-ALL_new.jsonl"

# 打开输入文件并逐行读取 JSON 对象
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析 JSON 对象
        data = json.loads(line.strip())
        
        # 移除不需要的字段
        keys_to_remove = ["emo2vec_emo", "emo2vec_confidence", "sensevoice_emo", "qwen72b_emo"]
        for key in keys_to_remove:
            if key in data:
                del data[key]
        
        # 将处理后的 JSON 对象写入输出文件
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"处理完成，结果已保存到 {output_file}")