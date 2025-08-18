#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
番茄遮阳棚控制决策树算法 - 优化版
支持多文件数据处理和综合分析
"""

import csv
import math
import random
import os

class DecisionTreeNode:
    """决策树节点类"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 分割特征（0:温度, 1:湿度, 2:光照）
        self.threshold = threshold  # 分割阈值
        self.left = left           # 左子树（小于阈值）
        self.right = right         # 右子树（大于等于阈值）
        self.value = value         # 叶节点的预测值（是否开启遮阳棚）

class DecisionTree:
    """决策树类"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def calculate_gini(self, y):
        """
        计算基尼不纯度
        基尼不纯度越小，数据越纯
        """
        if len(y) == 0:
            return 0
        
        # 统计各类别的数量
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        # 计算基尼不纯度
        gini = 1.0
        for count in counts.values():
            prob = float(count) / len(y)
            gini -= prob * prob
        
        return gini
    
    def split_data(self, X, y, feature, threshold):
        """
        根据特征和阈值分割数据
        返回左子树和右子树的数据
        """
        left_indices = []
        right_indices = []
        
        for i, x in enumerate(X):
            if x[feature] < threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        return left_X, left_y, right_X, right_y
    
    def find_best_split(self, X, y):
        """
        找到最佳分割点
        遍历所有特征和可能的阈值，选择基尼不纯度减少最多的分割
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0]) if X else 0
        
        for feature in range(n_features):
            # 获取该特征的所有唯一值
            values = set(x[feature] for x in X)
            
            for threshold in values:
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature, threshold)
                
                # 如果分割后某个子树为空，跳过
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # 计算分割后的加权基尼不纯度
                left_gini = self.calculate_gini(left_y)
                right_gini = self.calculate_gini(right_y)
                
                # 加权平均
                weighted_gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)
                
                # 如果基尼不纯度减少更多，更新最佳分割
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        """
        # 如果达到最大深度或样本数太少，创建叶节点
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # 返回多数类作为预测值
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            most_common = max(counts.items(), key=lambda x: x[1])[0]
            return DecisionTreeNode(value=most_common)
        
        # 如果所有样本都属于同一类，创建叶节点
        if len(set(y)) == 1:
            return DecisionTreeNode(value=y[0])
        
        # 找到最佳分割点
        best_feature, best_threshold = self.find_best_split(X, y)
        
        # 如果找不到好的分割点，创建叶节点
        if best_feature is None:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            most_common = max(counts.items(), key=lambda x: x[1])[0]
            return DecisionTreeNode(value=most_common)
        
        # 根据最佳分割点分割数据
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature, best_threshold)
        
        # 递归构建左右子树
        left_child = self.build_tree(left_X, left_y, depth + 1)
        right_child = self.build_tree(right_X, right_y, depth + 1)
        
        # 创建当前节点
        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X, y):
        """
        训练决策树
        """
        self.root = self.build_tree(X, y)
    
    def predict_single(self, x, node=None):
        """
        对单个样本进行预测
        """
        if node is None:
            node = self.root
        
        # 如果是叶节点，返回预测值
        if node.value is not None:
            return node.value
        
        # 根据特征值决定走左子树还是右子树
        if x[node.feature] < node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)
    
    def predict(self, X):
        """
        对多个样本进行预测
        """
        return [self.predict_single(x) for x in X]

def load_data_from_file(filename):
    """
    从单个文件加载番茄生长数据
    返回特征矩阵X、标签y和原始数据
    """
    X = []  # 特征矩阵：[[温度, 湿度, 光照], ...]
    y = []  # 标签：是否应该开启遮阳棚
    raw_data = []  # 原始数据，包含日期等信息
    data_info = {
        'filename': filename,
        'total_records': 0,
        'shade_records': 0,
        'no_shade_records': 0,
        'temp_stats': {'min': float('inf'), 'max': float('-inf'), 'avg': 0},
        'humidity_stats': {'min': float('inf'), 'max': float('-inf'), 'avg': 0},
        'sunlight_stats': {'min': float('inf'), 'max': float('-inf'), 'avg': 0}
    }
    
    temp_sum = 0
    humidity_sum = 0
    sunlight_sum = 0
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 提取特征：最高温度、湿度、光照时长
                max_temp = float(row['Max_Temp_C'])
                humidity = float(row['Humidity_Percent'])
                sunlight = float(row['Sunlight_Hours'])
                
                # 保存原始数据
                raw_data.append({
                    'date': row['Date'],
                    'plant_id': row['Plant_ID'],
                    'growth_stage': row['Growth_Stage'],
                    'weather': row['Weather_Condition'],
                    'max_temp': max_temp,
                    'min_temp': float(row['Min_Temp_C']),
                    'humidity': humidity,
                    'sunlight': sunlight,
                    'precipitation': float(row['Precipitation_mm']),
                    'soil_moisture': float(row['Soil_Moisture_Percent'])
                })
                
                # 更新统计信息
                data_info['temp_stats']['min'] = min(data_info['temp_stats']['min'], max_temp)
                data_info['temp_stats']['max'] = max(data_info['temp_stats']['max'], max_temp)
                data_info['humidity_stats']['min'] = min(data_info['humidity_stats']['min'], humidity)
                data_info['humidity_stats']['max'] = max(data_info['humidity_stats']['max'], humidity)
                data_info['sunlight_stats']['min'] = min(data_info['sunlight_stats']['min'], sunlight)
                data_info['sunlight_stats']['max'] = max(data_info['sunlight_stats']['max'], sunlight)
                
                temp_sum += max_temp
                humidity_sum += humidity
                sunlight_sum += sunlight
                
                # 根据农业经验判断是否需要遮阳棚
                # 条件：高温(>30度) 或 强光照(>14小时) 或 低湿度(<50%)
                need_shade = 0  # 0: 不需要遮阳棚, 1: 需要遮阳棚
                
                if max_temp > 30 or sunlight > 14 or humidity < 50:
                    need_shade = 1
                
                # 特殊情况：如果温度适中但光照很强，也需要遮阳棚
                if 25 <= max_temp <= 30 and sunlight > 13:
                    need_shade = 1
                
                X.append([max_temp, humidity, sunlight])
                y.append(need_shade)
                
                data_info['total_records'] += 1
                if need_shade == 1:
                    data_info['shade_records'] += 1
                else:
                    data_info['no_shade_records'] += 1
                    
            except (ValueError, KeyError):
                # 跳过无效数据行
                continue
    
    # 计算平均值
    if data_info['total_records'] > 0:
        data_info['temp_stats']['avg'] = temp_sum / data_info['total_records']
        data_info['humidity_stats']['avg'] = humidity_sum / data_info['total_records']
        data_info['sunlight_stats']['avg'] = sunlight_sum / data_info['total_records']
    
    return X, y, data_info, raw_data

def load_all_data():
    """
    加载目录中所有CSV文件的数据
    """
    all_X = []
    all_y = []
    all_data_info = []
    all_raw_data = []
    
    # 查找CSV文件
    csv_files = []
    for filename in os.listdir('.'):
        if filename.endswith('.csv') and 'tomato' in filename.lower():
            csv_files.append(filename)
    
    if not csv_files:
        print "未找到番茄数据CSV文件"
        return [], [], [], []
    
    print "找到以下CSV文件:"
    for filename in csv_files:
        print "  - " + filename
    
    # 加载每个文件的数据
    for filename in csv_files:
        try:
            X, y, data_info, raw_data = load_data_from_file(filename)
            all_X.extend(X)
            all_y.extend(y)
            all_data_info.append(data_info)
            all_raw_data.extend(raw_data)
            print "成功加载 %s: %d 条记录" % (filename, data_info['total_records'])
        except Exception as e:
            print "加载 %s 失败: %s" % (filename, e)
    
    return all_X, all_y, all_data_info, all_raw_data

def analyze_data_distribution(data_info_list):
    """
    分析数据分布情况
    """
    print "\n=== 数据分布分析 ==="
    
    for data_info in data_info_list:
        print "\n文件: %s" % data_info['filename']
        print "总记录数: %d" % data_info['total_records']
        print "需要遮阳棚: %d (%.1f%%)" % (
            data_info['shade_records'], 
            float(data_info['shade_records']) / data_info['total_records'] * 100
        )
        print "不需要遮阳棚: %d (%.1f%%)" % (
            data_info['no_shade_records'],
            float(data_info['no_shade_records']) / data_info['total_records'] * 100
        )
        
        print "温度统计: 最小值=%.1f°C, 最大值=%.1f°C, 平均值=%.1f°C" % (
            data_info['temp_stats']['min'],
            data_info['temp_stats']['max'],
            data_info['temp_stats']['avg']
        )
        print "湿度统计: 最小值=%.1f%%, 最大值=%.1f%%, 平均值=%.1f%%" % (
            data_info['humidity_stats']['min'],
            data_info['humidity_stats']['max'],
            data_info['humidity_stats']['avg']
        )
        print "光照统计: 最小值=%.1f小时, 最大值=%.1f小时, 平均值=%.1f小时" % (
            data_info['sunlight_stats']['min'],
            data_info['sunlight_stats']['max'],
            data_info['sunlight_stats']['avg']
        )

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    predictions = model.predict(X_test)
    
    # 计算准确率
    correct = sum(1 for p, t in zip(predictions, y_test) if p == t)
    accuracy = float(correct) / len(y_test)
    
    # 计算精确率、召回率、F1分数
    tp = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, y_test) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(predictions, y_test) if p == 0 and t == 0)
    
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def print_tree(node, depth=0, feature_names=['温度', '湿度', '光照']):
    """
    打印决策树结构
    """
    if node is None:
        return
    
    indent = "  " * depth
    
    if node.value is not None:
        # 叶节点
        if node.value == 1:
            action = "开启遮阳棚"
        else:
            action = "关闭遮阳棚"
        print indent + "预测: " + action
    else:
        # 内部节点
        feature_name = feature_names[node.feature]
        print indent + "如果 " + feature_name + " < " + str(node.threshold) + ":"
        print_tree(node.left, depth + 1, feature_names)
        print indent + "否则:"
        print_tree(node.right, depth + 1, feature_names)

def print_confusion_matrix(metrics):
    """
    打印混淆矩阵
    """
    print "\n=== 混淆矩阵 ==="
    print "预测\\实际    开启遮阳棚    关闭遮阳棚"
    print "开启遮阳棚      %d          %d" % (metrics['tp'], metrics['fp'])
    print "关闭遮阳棚      %d          %d" % (metrics['fn'], metrics['tn'])

def generate_daily_recommendations(model, raw_data):
    """
    生成每一天的遮阳棚开启建议
    """
    print "\n=== 每日遮阳棚开启建议 ==="
    print "日期\t\t\t植物ID\t生长阶段\t\t温度(°C)\t湿度(%)\t光照(小时)\t建议\t\t原因"
    print "-" * 120
    
    # 按日期和植物ID排序
    sorted_data = sorted(raw_data, key=lambda x: (x['date'], int(x['plant_id'])))
    
    for record in sorted_data:
        # 使用模型预测
        features = [record['max_temp'], record['humidity'], record['sunlight']]
        prediction = model.predict_single(features)
        
        # 确定建议和原因
        if prediction == 1:
            recommendation = "开启遮阳棚"
            reasons = []
            if record['max_temp'] > 30:
                reasons.append("高温")
            if record['sunlight'] > 14:
                reasons.append("强光照")
            if record['humidity'] < 50:
                reasons.append("低湿度")
            if 25 <= record['max_temp'] <= 30 and record['sunlight'] > 13:
                reasons.append("温度适中但光照强")
            reason = ", ".join(reasons) if reasons else "综合环境因素"
        else:
            recommendation = "关闭遮阳棚"
            reason = "环境条件适宜"
        
        # 格式化输出
        print "%s\t%s\t%s\t\t%.1f\t\t%.1f\t\t%.1f\t\t%s\t%s" % (
            record['date'],
            record['plant_id'],
            record['growth_stage'][:8],  # 限制长度
            record['max_temp'],
            record['humidity'],
            record['sunlight'],
            recommendation,
            reason
        )

def save_daily_recommendations_to_file(model, raw_data, filename="daily_recommendations.csv"):
    """
    将每日建议保存到CSV文件
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['日期', '植物ID', '生长阶段', '天气', '最高温度(°C)', '最低温度(°C)', 
                        '湿度(%)', '光照(小时)', '降水量(mm)', '土壤湿度(%)', '遮阳棚建议', '原因'])
        
        # 按日期和植物ID排序
        sorted_data = sorted(raw_data, key=lambda x: (x['date'], int(x['plant_id'])))
        
        for record in sorted_data:
            # 使用模型预测
            features = [record['max_temp'], record['humidity'], record['sunlight']]
            prediction = model.predict_single(features)
            
            # 确定建议和原因
            if prediction == 1:
                recommendation = "开启遮阳棚"
                reasons = []
                if record['max_temp'] > 30:
                    reasons.append("高温")
                if record['sunlight'] > 14:
                    reasons.append("强光照")
                if record['humidity'] < 50:
                    reasons.append("低湿度")
                if 25 <= record['max_temp'] <= 30 and record['sunlight'] > 13:
                    reasons.append("温度适中但光照强")
                reason = ", ".join(reasons) if reasons else "综合环境因素"
            else:
                recommendation = "关闭遮阳棚"
                reason = "环境条件适宜"
            
            writer.writerow([
                record['date'],
                record['plant_id'],
                record['growth_stage'],
                record['weather'],
                record['max_temp'],
                record['min_temp'],
                record['humidity'],
                record['sunlight'],
                record['precipitation'],
                record['soil_moisture'],
                recommendation,
                reason
            ])
    
    print "\n每日建议已保存到文件: %s" % filename

def main():
    """
    主函数：训练和测试决策树模型
    """
    print "=== 番茄遮阳棚控制决策树算法 - 优化版 ==="
    print
    
    # 加载所有数据
    print "正在加载数据..."
    X, y, data_info_list, raw_data = load_all_data()
    
    if not X:
        print "没有加载到有效数据，程序退出"
        return
    
    print "总共加载 %d 条数据记录" % len(X)
    
    # 分析数据分布
    analyze_data_distribution(data_info_list)
    
    # 划分训练集和测试集
    print "\n划分训练集和测试集..."
    n_samples = len(X)
    n_train = int(0.8 * n_samples)  # 80%用于训练
    
    # 随机打乱数据
    indices = range(n_samples)
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    print "训练集大小: %d, 测试集大小: %d" % (len(X_train), len(X_test))
    
    # 训练决策树模型
    print "\n训练决策树模型..."
    model = DecisionTree(max_depth=5, min_samples_split=10)
    model.fit(X_train, y_train)
    
    # 评估模型
    print "评估模型性能..."
    metrics = evaluate_model(model, X_test, y_test)
    
    print "\n=== 模型性能 ==="
    print "准确率: %.2f%%" % (metrics['accuracy'] * 100)
    print "精确率: %.2f%%" % (metrics['precision'] * 100)
    print "召回率: %.2f%%" % (metrics['recall'] * 100)
    print "F1分数: %.2f" % metrics['f1']
    
    # 打印混淆矩阵
    print_confusion_matrix(metrics)
    
    # 打印决策树结构
    print "\n=== 决策树结构 ==="
    print_tree(model.root)
    
    # 生成每日建议
    generate_daily_recommendations(model, raw_data)
    
    # 保存每日建议到文件
    save_daily_recommendations_to_file(model, raw_data)
    
    # 测试一些具体案例
    print "\n=== 测试案例 ==="
    test_cases = [
        [32, 60, 15],  # 高温
        [25, 45, 12],  # 低湿度
        [28, 70, 16],  # 强光照
        [22, 65, 10],  # 正常条件
        [35, 80, 14],  # 高温高湿
        [18, 75, 8],   # 低温
        [30, 50, 13],  # 高温适中光照
    ]
    
    for i, case in enumerate(test_cases):
        prediction = model.predict_single(case)
        if prediction == 1:
            action = "开启遮阳棚"
        else:
            action = "关闭遮阳棚"
        print "案例 %d: 温度=%.1f°C, 湿度=%.1f%%, 光照=%.1f小时 -> %s" % (
            i+1, case[0], case[1], case[2], action
        )
    
    # 综合分析
    print "\n=== 综合分析 ==="
    total_shade_needed = sum(y)
    total_records = len(y)
    shade_percentage = float(total_shade_needed) / total_records * 100
    
    print "总体数据中需要遮阳棚的比例: %.1f%% (%d/%d)" % (
        shade_percentage, total_shade_needed, total_records
    )
    
    if len(data_info_list) > 1:
        print "\n不同年份数据对比:"
        for data_info in data_info_list:
            year_shade_percentage = float(data_info['shade_records']) / data_info['total_records'] * 100
            print "%s: 遮阳棚需求 %.1f%%" % (data_info['filename'], year_shade_percentage)

if __name__ == "__main__":
    main()
