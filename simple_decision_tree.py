#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
简化版决策树算法演示
专注于核心算法实现，便于理解决策树原理
"""

import random

class SimpleDecisionTree:
    """简化版决策树实现"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None
    
    def calculate_gini(self, labels):
        """计算基尼不纯度"""
        if len(labels) == 0:
            return 0
        
        # 统计各类别数量
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        
        # 计算基尼不纯度: Gini = 1 - Σ(p_i²)
        gini = 1.0
        for count in counts.values():
            prob = float(count) / len(labels)
            gini -= prob * prob
        
        return gini
    
    def find_best_split(self, data, labels):
        """寻找最佳分割点"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = len(data[0]) if data else 0
        
        # 遍历每个特征
        for feature in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            values = set(row[feature] for row in data)
            
            for threshold in values:
                # 根据阈值分割数据
                left_data, left_labels = [], []
                right_data, right_labels = [], []
                
                for i, row in enumerate(data):
                    if row[feature] < threshold:
                        left_data.append(row)
                        left_labels.append(labels[i])
                    else:
                        right_data.append(row)
                        right_labels.append(labels[i])
                
                # 如果分割后某个子集为空，跳过
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue
                
                # 计算分割后的加权基尼不纯度
                left_gini = self.calculate_gini(left_labels)
                right_gini = self.calculate_gini(right_labels)
                
                # 加权平均
                weighted_gini = (len(left_labels) * left_gini + len(right_labels) * right_gini) / len(labels)
                
                # 更新最佳分割
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, data, labels, depth=0):
        """递归构建决策树"""
        # 停止条件1: 达到最大深度
        if depth >= self.max_depth:
            return self.create_leaf(labels)
        
        # 停止条件2: 所有样本属于同一类
        if len(set(labels)) == 1:
            return {'type': 'leaf', 'value': labels[0]}
        
        # 停止条件3: 样本数太少
        if len(labels) < 2:
            return self.create_leaf(labels)
        
        # 寻找最佳分割点
        best_feature, best_threshold = self.find_best_split(data, labels)
        
        # 如果找不到好的分割点，创建叶节点
        if best_feature is None:
            return self.create_leaf(labels)
        
        # 根据最佳分割点分割数据
        left_data, left_labels = [], []
        right_data, right_labels = [], []
        
        for i, row in enumerate(data):
            if row[best_feature] < best_threshold:
                left_data.append(row)
                left_labels.append(labels[i])
            else:
                right_data.append(row)
                right_labels.append(labels[i])
        
        # 递归构建左右子树
        left_child = self.build_tree(left_data, left_labels, depth + 1)
        right_child = self.build_tree(right_data, right_labels, depth + 1)
        
        # 创建内部节点
        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }
    
    def create_leaf(self, labels):
        """创建叶节点"""
        # 返回多数类
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        return {'type': 'leaf', 'value': most_common}
    
    def fit(self, data, labels):
        """训练决策树"""
        self.root = self.build_tree(data, labels)
    
    def predict_single(self, x, node=None):
        """对单个样本进行预测"""
        if node is None:
            node = self.root
        
        # 如果是叶节点，返回预测值
        if node['type'] == 'leaf':
            return node['value']
        
        # 根据特征值决定走左子树还是右子树
        if x[node['feature']] < node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])
    
    def predict(self, data):
        """对多个样本进行预测"""
        return [self.predict_single(x) for x in data]

def print_tree_simple(node, depth=0, feature_names=['温度', '湿度', '光照']):
    """打印简化版决策树结构"""
    if node is None:
        return
    
    indent = "  " * depth
    
    if node['type'] == 'leaf':
        action = "开启遮阳棚" if node['value'] == 1 else "关闭遮阳棚"
        print indent + "预测: " + action
    else:
        feature_name = feature_names[node['feature']]
        print indent + "如果 " + feature_name + " < " + str(node['threshold']) + ":"
        print_tree_simple(node['left'], depth + 1, feature_names)
        print indent + "否则:"
        print_tree_simple(node['right'], depth + 1, feature_names)

def demo():
    """演示决策树算法"""
    print "=== 决策树算法演示 ==="
    print
    
    # 创建示例数据
    # 特征: [温度, 湿度, 光照]
    # 标签: 0=关闭遮阳棚, 1=开启遮阳棚
    data = [
        [32, 60, 15],  # 高温 -> 开启
        [25, 45, 12],  # 低湿度 -> 开启
        [28, 70, 16],  # 强光照 -> 开启
        [22, 65, 10],  # 正常 -> 关闭
        [35, 80, 14],  # 高温 -> 开启
        [20, 70, 8],   # 正常 -> 关闭
        [30, 50, 13],  # 高温 -> 开启
        [18, 75, 9],   # 正常 -> 关闭
    ]
    
    labels = [1, 1, 1, 0, 1, 0, 1, 0]
    
    print "训练数据:"
    feature_names = ['温度', '湿度', '光照']
    for i, (x, y) in enumerate(zip(data, labels)):
        action = "开启遮阳棚" if y == 1 else "关闭遮阳棚"
        print "样本 %d: 温度=%.1f°C, 湿度=%.1f%%, 光照=%.1f小时 -> %s" % (
            i+1, x[0], x[1], x[2], action
        )
    print
    
    # 训练决策树
    print "训练决策树..."
    tree = SimpleDecisionTree(max_depth=3)
    tree.fit(data, labels)
    
    # 打印决策树结构
    print "决策树结构:"
    print_tree_simple(tree.root, feature_names=feature_names)
    print
    
    # 测试预测
    print "测试预测:"
    test_cases = [
        [33, 55, 14],  # 应该开启
        [23, 60, 11],  # 应该关闭
        [29, 65, 15],  # 应该开启
    ]
    
    for i, test_case in enumerate(test_cases):
        prediction = tree.predict_single(test_case)
        action = "开启遮阳棚" if prediction == 1 else "关闭遮阳棚"
        print "测试 %d: 温度=%.1f°C, 湿度=%.1f%%, 光照=%.1f小时 -> %s" % (
            i+1, test_case[0], test_case[1], test_case[2], action
        )

if __name__ == "__main__":
    demo()
