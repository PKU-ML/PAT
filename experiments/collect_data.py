import json
import os
import pandas as pd

# Define the target folder path
version = 'transfer'
folder_path = f"./eval/{version}/"

# Get all the files in the folder
files = os.listdir(folder_path)
json_files = [f for f in files if f.endswith('.json')]
json_files.sort()
print(json_files)

headers = ["offset", "jb", "em", "ori_jb", "ori_em"]

def remove_brackets(alist):
    for i in range(len(alist)):
        if isinstance(alist[i], list):
            alist[i] = alist[i][0]
    return alist

jb_list = []
em_list = []
ori_jb_list = []
ori_em_list = []
rec = []
bad_prompts = []
# 遍历所有JSON文件
for kkk, file in enumerate(json_files):
    with open(os.path.join(folder_path, file)) as f:
        data = json.load(f)  # 假设每个文件都只包含一个对象
        for k in data.keys():
            data = data[k]
        jb_list = []
        em_list = []
        jb_list.extend(data.get('jb', [])) 
        em_list.extend(data.get('em', [])) 

    jb_list = remove_brackets(jb_list)
    em_list = remove_brackets(em_list)

    for i, jb in enumerate(jb_list):
        if jb:
            bad_prompts.append(kkk*10 + i)

    jb_count = {item: jb_list.count(item) for item in jb_list}
    em_count = {item: em_list.count(item) for item in em_list}

    # print the results
    print("jb counts:", jb_count)
    print("em counts:", em_count)

    try:
        with open(os.path.join(folder_path, file)) as f:
            ori_jb_list = []
            ori_em_list = []
            ori_jb_list.extend(data.get('ori_jb', []))  # 默认值为空列表，以处理可能不存在的属性
            ori_em_list.extend(data.get('ori_em', []))  # 默认值为空列表，以处理可能不存在的属性

        ori_jb_list = remove_brackets(ori_jb_list)
        ori_em_list = remove_brackets(ori_em_list)
        # print(jb_list)


        ori_jb_count = {item: ori_jb_list.count(item) for item in ori_jb_list}
        ori_em_count = {item: ori_em_list.count(item) for item in ori_em_list}

        print("ori_jb的数量统计:", ori_jb_count)
        print("ori_em的数量统计:", ori_em_count)
        # jb = ori_em_count[1] if 1 in ori_em_count else 0
        # defense = ori_jb_count[0]
        # print(jb, defense, len(ori_jb_list) - jb - defense)
    except:
        pass

    rec.append([json.dumps(jb_count), json.dumps(em_count), json.dumps(ori_jb_count), json.dumps(ori_em_count)])
data = pd.DataFrame(rec, columns=["jb", "em", "ori_jb", "ori_em"])
data.to_csv(f"./csv/{version}.csv")
print(bad_prompts)