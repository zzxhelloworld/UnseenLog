#!/usr/bin/env python3
import re


def parse_data(text):
    """
    解析数据文本，返回一个字典，其中 key 为文件名，value 为 (label0, label1) 数量，
    同时累计全局标签统计 (total_label0, total_label1)
    """
    data_dict = {}
    total_label0 = 0
    total_label1 = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 匹配格式：序号: 文件名 | label 0: 数字 | label 1: 数字 (顺序可能不同)
        m_file = re.search(r"^\d+:\s*([^|]+?)\s*\|", line)
        if not m_file:
            continue
        file_name = m_file.group(1).strip()

        # 提取 label 0 和 label 1 的数值，若不存在则默认为 0
        label0 = 0
        label1 = 0
        m0 = re.search(r"label 0:\s*(\d+)", line)
        if m0:
            label0 = int(m0.group(1))
        m1 = re.search(r"label 1:\s*(\d+)", line)
        if m1:
            label1 = int(m1.group(1))

        data_dict[file_name] = (label0, label1)
        total_label0 += label0
        total_label1 += label1

    return data_dict, total_label0, total_label1


# 原始数据文本（注意此处将训练集和测试集数据分别以标题区分）
data = """
train dataloader:
1: pub.developers.forum.infrastructure.cache.DbCacheServiceImpl | label 0: 120201 | label 1: 3 | 
2: pub.developers.forum.app.manager.UserManager | label 0: 83791 | label 1: 36462 | 
3: pub.developers.forum.facade.impl.UserApiServiceImpl | label 0: 120254 | 
4: pub.developers.forum.portal.support.GlobalViewInterceptor | label 0: 120257 | 
5: root | label 0: 120260 | 
6: pub.developers.forum.portal.support.CorsInterceptor | label 0: 120257 | 
7: pub.developers.forum.facade.impl.ApprovalApiServiceImpl | label 0: 34119 | 
8: pub.developers.forum.infrastructure.dal.dao.UserFollowDAO | label 0: 46425 | 
9: pub.developers.forum.infrastructure.ApprovalRepositoryImpl | label 0: 33398 | 
10: pub.developers.forum.app.manager.ApprovalManager | label 0: 34119 | 
11: pub.developers.forum.infrastructure.dal.dao.PostsDAO | label 0: 99044 | 
12: pub.developers.forum.infrastructure.PostsRepositoryImpl | label 0: 37819 | 
13: pub.developers.forum.facade.impl.CommentApiServiceImpl | label 0: 26972 | 
14: pub.developers.forum.app.manager.CommentManager | label 0: 26972 | 
15: pub.developers.forum.infrastructure.dal.dao.CommentDAO | label 0: 50255 | 
16: pub.developers.forum.infrastructure.CommentRepositoryImpl | label 0: 50255 | 
17: pub.developers.forum.infrastructure.UserRepositoryImpl | label 0: 32048 | 
18: pub.developers.forum.common.support.GlobalViewConfig | label 0: 40322 | 
19: pub.developers.forum.portal.controller.InterestController | label 0: 6325 | 
20: pub.developers.forum.facade.impl.PostsApiServiceImpl | label 0: 6432 | 
21: pub.developers.forum.infrastructure.dal.dao.UserFoodDAO | label 0: 6323 | 
22: pub.developers.forum.infrastructure.UserFoodRepositoryImpl | label 0: 6324 | 
23: pub.developers.forum.app.manager.PostsManager | label 0: 4440 | label 1: 1884 | 
24: pub.developers.forum.portal.support.WebUtil | label 0: 39369 | 
25: pub.developers.forum.infrastructure.dal.dao.UserDAO | label 0: 56461 | 
26: pub.developers.forum.facade.impl.TagApiServiceImpl | label 0: 18982 | label 1: 21234 | 
27: pub.developers.forum.app.manager.TagManager | label 0: 24983 | 
28: pub.developers.forum.facade.impl.ConfigApiServiceImpl | label 0: 26950 | 
29: pub.developers.forum.infrastructure.dal.dao.ConfigDAO | label 0: 26926 | 
30: pub.developers.forum.infrastructure.ConfigRepositoryImpl | label 0: 26926 | 
31: pub.developers.forum.app.manager.ConfigManager | label 0: 11715 | label 1: 15235 | 
32: pub.developers.forum.facade.impl.MessageApiServiceImpl | label 0: 37836 | 
33: pub.developers.forum.infrastructure.dal.dao.MessageDAO | label 0: 37703 | 
34: pub.developers.forum.infrastructure.MessageRepositoryImpl | label 0: 37724 | 
35: pub.developers.forum.app.manager.MessageManager | label 0: 36788 | label 1: 1048 | 
36: pub.developers.forum.infrastructure.FaqRepositoryImpl | label 0: 30758 | 
37: pub.developers.forum.app.manager.FaqManager | label 0: 30758 | 
38: pub.developers.forum.facade.impl.FaqApiServiceImpl | label 0: 30758 | 
39: pub.developers.forum.portal.controller.FaqListController | label 0: 19938 | 
40: pub.developers.forum.infrastructure.dal.dao.TagPostsMappingDAO | label 0: 61076 | 
41: pub.developers.forum.infrastructure.dal.dao.TagDAO | label 0: 62417 | 
42: pub.developers.forum.infrastructure.dal.dao.CacheDAO | label 0: 14935 | 
43: pub.developers.forum.facade.impl.ArticleApiServiceImpl | label 0: 28100 | 
44: pub.developers.forum.infrastructure.ArticleRepositoryImpl | label 0: 26580 | 
45: pub.developers.forum.infrastructure.dal.dao.ArticleTypeDAO | label 0: 26997 | 
46: pub.developers.forum.app.manager.ArticleManager | label 0: 28100 | 
47: pub.developers.forum.portal.controller.ArticleInfoController | label 0: 9440 | 
48: pub.developers.forum.infrastructure.TagRepositoryImpl | label 0: 36853 | 
49: pub.developers.forum.infrastructure.ArticleTypeRepositoryImpl | label 0: 17135 | 
50: pub.developers.forum.app.manager.AbstractPostsManager | label 0: 108 | 
51: pub.developers.forum.portal.controller.FaqInfoController | label 0: 2547 | 
52: pub.developers.forum.portal.controller.MessageController | label 0: 1020 | 
53: pub.developers.forum.infrastructure.dal.dao.OptLogDAO | label 0: 221 | 
54: pub.developers.forum.infrastructure.OptLogRepositoryImpl | label 0: 221 | 
55: pub.developers.forum.portal.controller.IndexController | label 0: 450 | 
56: pub.developers.forum.portal.controller.TagController | label 0: 437 | 
57: pub.developers.forum.portal.controller.UserController | label 0: 450 | 
58: pub.developers.forum.portal.controller.VueController | label 0: 285 | 

test dataloader:
1: pub.developers.forum.infrastructure.dal.dao.UserFollowDAO | label 0: 58604 | 
2: pub.developers.forum.infrastructure.UserRepositoryImpl | label 0: 59662 | 
3: pub.developers.forum.app.manager.UserManager | label 1: 59778 | label 0: 34194 | 
4: pub.developers.forum.facade.impl.UserApiServiceImpl | label 1: 55223 | label 0: 38750 | 
5: pub.developers.forum.infrastructure.dal.dao.CacheDAO | label 0: 59377 | 
6: pub.developers.forum.infrastructure.cache.DbCacheServiceImpl | label 1: 55248 | label 0: 38724 | 
7: pub.developers.forum.portal.support.GlobalViewInterceptor | label 1: 55249 | label 0: 38724 | 
8: root | label 0: 93973 | 
9: pub.developers.forum.portal.support.CorsInterceptor | label 1: 55249 | label 0: 38724 | 
10: pub.developers.forum.facade.impl.ApprovalApiServiceImpl | label 0: 25506 | 
11: pub.developers.forum.infrastructure.ApprovalRepositoryImpl | label 0: 25506 | 
12: pub.developers.forum.app.manager.ApprovalManager | label 0: 25506 | 
13: pub.developers.forum.infrastructure.dal.dao.PostsDAO | label 0: 86893 | 
14: pub.developers.forum.infrastructure.PostsRepositoryImpl | label 0: 25805 | 
15: pub.developers.forum.infrastructure.dal.dao.UserDAO | label 0: 55821 | 
16: pub.developers.forum.facade.impl.FaqApiServiceImpl | label 0: 46944 | 
17: pub.developers.forum.infrastructure.FaqRepositoryImpl | label 1: 13843 | label 0: 33102 | 
18: pub.developers.forum.infrastructure.dal.dao.TagPostsMappingDAO | label 0: 76013 | 
19: pub.developers.forum.infrastructure.dal.dao.TagDAO | label 0: 76060 | 
20: pub.developers.forum.app.manager.FaqManager | label 0: 46945 | 
21: pub.developers.forum.portal.controller.FaqInfoController | label 0: 13407 | 
22: pub.developers.forum.common.support.GlobalViewConfig | label 0: 38539 | 
23: pub.developers.forum.portal.support.WebUtil | label 0: 36591 | 
24: pub.developers.forum.facade.impl.CommentApiServiceImpl | label 0: 21949 | 
25: pub.developers.forum.infrastructure.dal.dao.CommentDAO | label 0: 54118 | 
26: pub.developers.forum.infrastructure.CommentRepositoryImpl | label 0: 54118 | 
27: pub.developers.forum.app.manager.CommentManager | label 0: 21949 | 
28: pub.developers.forum.facade.impl.TagApiServiceImpl | label 1: 31603 | label 0: 5034 | 
29: pub.developers.forum.infrastructure.TagRepositoryImpl | label 0: 66768 | 
30: pub.developers.forum.app.manager.TagManager | label 0: 36637 | 
31: pub.developers.forum.facade.impl.MessageApiServiceImpl | label 0: 38567 | 
32: pub.developers.forum.infrastructure.dal.dao.MessageDAO | label 0: 38567 | 
33: pub.developers.forum.infrastructure.MessageRepositoryImpl | label 0: 38567 | 
34: pub.developers.forum.app.manager.MessageManager | label 0: 38567 | 
35: pub.developers.forum.portal.controller.MessageController | label 0: 1947 | 
"""

# 指定测试集中需要计算标签 1 占比的目标文件列表
target_files = [
    "pub.developers.forum.infrastructure.cache.DbCacheServiceImpl",
    "pub.developers.forum.facade.impl.UserApiServiceImpl",
    "pub.developers.forum.portal.support.GlobalViewInterceptor",
    "pub.developers.forum.portal.support.CorsInterceptor",
    "pub.developers.forum.infrastructure.FaqRepositoryImpl"
    # "pub.developers.forum.app.manager.MessageManager"
    # "pub.developers.forum.app.manager.ArticleManager"
    # "pub.developers.forum.app.manager.ConfigManager",
    # "pub.developers.forum.app.manager.PostsManager"
]

# 分割原始文本为训练集和测试集两部分
parts = data.split("test dataloader:")
if len(parts) != 2:
    print("数据格式错误：无法分割训练集和测试集")
    exit(1)

train_text = parts[0]
test_text = parts[1]

# 分别解析训练集和测试集
train_data, train_total_label0, train_total_label1 = parse_data(train_text)
test_data, test_total_label0, test_total_label1 = parse_data(test_text)

# 打印训练集统计信息
print("=== 训练集统计 ===")
print("总体标签统计:")
print("  Label 0: {}".format(train_total_label0))
print("  Label 1: {}".format(train_total_label1))
total_train = train_total_label0 + train_total_label1
if total_train > 0:
    print("标签比例:")
    print("  Label 1/0: {}/{}={:.2%}".format(train_total_label1, train_total_label0, train_total_label1 / train_total_label0))
else:
    print("训练集没有标签数据。")

# print("\n各文件标签情况:")
# for file_name, (l0, l1) in train_data.items():
#     print("  {} -> Label 0: {}, Label 1: {}".format(file_name, l0, l1))

# 打印测试集统计信息
print("\n=== 测试集统计 ===")
print("总体标签统计:")
print("  Label 0: {}".format(test_total_label0))
print("  Label 1: {}".format(test_total_label1))
total_test = test_total_label0 + test_total_label1
if total_test > 0:
    print("标签比例:")
    print("  Label 1/0: {}/{}={:.2%}".format(test_total_label1, test_total_label0, test_total_label1 / test_total_label0))
else:
    print("测试集没有标签数据。")

# print("\n各文件标签情况:")
# for file_name, (l0, l1) in test_data.items():
#     print("  {} -> Label 0: {}, Label 1: {}".format(file_name, l0, l1))


# 分别统计每个目标文件的 Label 1 情况，并计算累计 Label 1 数量
aggregate_target_label1 = 0
print("\n=== 指定文件的测试集统计 ===")
for target_file in target_files:
    if target_file in test_data:
        file_label1 = test_data[target_file][1]
        aggregate_target_label1 += file_label1
        if test_total_label1 > 0:
            ratio = file_label1 / test_total_label1
            print("文件 '{}' 的 Label 1 数量为 {}，占测试集总 Label 1 数量 {} 的比例为：{:.2%}".format(
                target_file, file_label1, test_total_label1, ratio))
        else:
            print("测试集中总的 Label 1 数量为 0，无法计算比例。")
    else:
        print("测试集中未找到目标文件：{}".format(target_file))

# 输出累计的目标文件 Label 1 数量及其占比
if test_total_label1 > 0:
    aggregate_ratio = aggregate_target_label1 / test_total_label1
    print("\n累计未见过异常的 Label 1 数量/测试集总 Label 1 数量的比例为：{}/{}={:.2%}".format(
        aggregate_target_label1, test_total_label1, aggregate_ratio))
else:
    print("测试集中总的 Label 1 数量为 0，无法计算累计比例。")
