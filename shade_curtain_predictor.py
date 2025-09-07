#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遮阳帘预测系统 - 基于决策树回归算法
根据光照强度、温度、湿度等环境数据预测是否需要开启遮阳帘
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime


class ShadeCurtainPredictor:
    """遮阳帘预测器类"""
    
    def __init__(self):
        self.model = None
        self.feature_names = ['light_intensity', 'temperature', 'humidity']
        self.model_path = 'models/shade_curtain_model.pkl'
        self.model_info_path = 'models/shade_curtain_model_info.json'
        
    def load_data(self, csv_file):
        """
        加载环境数据
        
        Args:
            csv_file (str): CSV文件路径
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            data = pd.read_csv(csv_file)
            print(f"成功加载数据，共 {len(data)} 条记录")
            print(f"数据列: {list(data.columns)}")
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def preprocess_data(self, data):
        """
        数据预处理和特征工程
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            tuple: (X, y) 特征矩阵和目标变量
        """
        # 创建目标变量：基于光照强度判断是否需要遮阳帘
        # 光照强度越高，越需要遮阳帘（1表示需要，0表示不需要）
        # 使用阈值方法创建目标变量
        light_threshold = 500  # 光照强度阈值
        temp_threshold = 30    # 温度阈值
        
        # 创建目标变量：综合考虑光照强度和温度
        y = np.where(
            (data['light_intensity'] > light_threshold) | 
            (data['temperature'] > temp_threshold), 
            1, 0
        )
        
        # 选择特征
        X = data[self.feature_names].copy()
        
        # 添加时间特征（从时间戳中提取小时）
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            X['hour'] = data['timestamp'].dt.hour
            X['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # 添加交互特征
        X['light_temp_interaction'] = X['light_intensity'] * X['temperature']
        X['temp_humidity_interaction'] = X['temperature'] * X['humidity']
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量分布: {np.bincount(y)}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        训练决策树回归模型
        
        Args:
            X (pd.DataFrame): 特征矩阵
            y (np.array): 目标变量
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        Returns:
            dict: 模型评估结果
        """
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 创建决策树回归器
        self.model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 将回归结果转换为分类结果（阈值0.5）
        y_train_pred_class = (y_train_pred > 0.5).astype(int)
        y_test_pred_class = (y_test_pred > 0.5).astype(int)
        
        # 计算评估指标
        train_accuracy = np.mean(y_train_pred_class == y_train)
        test_accuracy = np.mean(y_test_pred_class == y_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        print("模型训练完成！")
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"训练集MSE: {train_mse:.4f}")
        print(f"测试集MSE: {test_mse:.4f}")
        print(f"训练集R²: {train_r2:.4f}")
        print(f"测试集R²: {test_r2:.4f}")
        
        return results
    
    def save_model(self):
        """保存训练好的模型"""
        if self.model is None:
            print("没有训练好的模型可以保存")
            return False
        
        # 创建模型目录
        os.makedirs('models', exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, self.model_path)
        
        # 保存模型信息
        model_info = {
            'feature_names': list(self.model.feature_names_in_),
            'feature_importance': dict(zip(self.model.feature_names_in_, self.model.feature_importances_)),
            'model_type': 'DecisionTreeRegressor',
            'training_time': datetime.now().isoformat(),
            'n_features': len(self.model.feature_names_in_)
        }
        
        import json
        with open(self.model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {self.model_path}")
        print(f"模型信息已保存到: {self.model_info_path}")
        return True
    
    def load_model(self):
        """加载已保存的模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"模型已从 {self.model_path} 加载")
                return True
            else:
                print(f"模型文件不存在: {self.model_path}")
                return False
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def predict(self, light_intensity, temperature, humidity, hour=None):
        """
        预测是否需要开启遮阳帘
        
        Args:
            light_intensity (float): 光照强度
            temperature (float): 温度
            humidity (float): 湿度
            hour (int): 小时（可选）
            
        Returns:
            dict: 预测结果
        """
        if self.model is None:
            return {"error": "模型未加载，请先训练或加载模型"}
        
        # 准备输入特征
        features = {
            'light_intensity': light_intensity,
            'temperature': temperature,
            'humidity': humidity
        }
        
        # 添加时间特征（如果提供）
        if hour is not None:
            features['hour'] = hour
            features['day_of_week'] = 0  # 默认周一
        
        # 添加交互特征
        features['light_temp_interaction'] = light_intensity * temperature
        features['temp_humidity_interaction'] = temperature * humidity
        
        # 创建特征向量
        feature_vector = []
        for feature_name in self.model.feature_names_in_:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0)  # 默认值
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # 创建DataFrame以保持特征名称
        import pandas as pd
        feature_df = pd.DataFrame(feature_vector, columns=self.model.feature_names_in_)
        
        # 预测
        prediction = self.model.predict(feature_df)[0]
        probability = prediction
        
        # 转换为分类结果
        should_open = prediction > 0.5
        confidence = abs(prediction - 0.5) * 2  # 置信度
        
        result = {
            'should_open_shade': bool(should_open),
            'probability': float(probability),
            'confidence': float(confidence),
            'recommendation': "建议开启遮阳帘" if should_open else "建议关闭遮阳帘",
            'input_features': {
                'light_intensity': light_intensity,
                'temperature': temperature,
                'humidity': humidity,
                'hour': hour
            }
        }
        
        return result


def main():
    """主函数 - 训练模型"""
    predictor = ShadeCurtainPredictor()
    
    # 加载数据
    data_file = 'weekly_environment_data_20250907_160200.csv'
    data = predictor.load_data(data_file)
    
    if data is None:
        return
    
    # 数据预处理
    X, y = predictor.preprocess_data(data)
    
    # 训练模型
    results = predictor.train_model(X, y)
    
    # 保存模型
    predictor.save_model()
    
    # 显示特征重要性
    print("\n特征重要性:")
    for feature, importance in sorted(results['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
