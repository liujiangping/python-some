#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遮阳帘预测系统 - 快速启动版本
简化版本，适合快速测试和演示
"""

import sys
import os
from shade_curtain_predictor import ShadeCurtainPredictor


def quick_predict():
    """快速预测功能"""
    print("🌞 遮阳帘快速预测系统")
    print("=" * 40)
    
    # 初始化预测器
    predictor = ShadeCurtainPredictor()
    
    # 加载模型
    if not predictor.load_model():
        print("❌ 模型未找到，正在训练新模型...")
        
        # 检查数据文件
        data_file = 'weekly_environment_data_20250907_160200.csv'
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return
        
        # 训练模型
        data = predictor.load_data(data_file)
        if data is None:
            return
        
        X, y = predictor.preprocess_data(data)
        predictor.train_model(X, y)
        predictor.save_model()
        print("✅ 模型训练完成")
    
    # 获取用户输入
    print("\n请输入环境数据:")
    try:
        light = float(input("光照强度 (0-1000): "))
        temp = float(input("温度 (°C): "))
        humidity = float(input("湿度 (%): "))
        
        # 进行预测
        result = predictor.predict(light, temp, humidity)
        
        if "error" in result:
            print(f"❌ 预测失败: {result['error']}")
            return
        
        # 显示结果
        print("\n" + "=" * 30)
        print("📊 预测结果")
        print("=" * 30)
        
        status = "🌞 开启" if result['should_open_shade'] else "🌙 关闭"
        print(f"遮阳帘状态: {status}")
        print(f"推荐操作: {result['recommendation']}")
        print(f"预测概率: {result['probability']:.3f}")
        print(f"置信度: {result['confidence']:.3f}")
        
        # 简单解释
        if result['should_open_shade']:
            print("\n💡 建议开启遮阳帘，当前环境可能过于明亮或炎热")
        else:
            print("\n💡 建议关闭遮阳帘，当前环境条件适宜")
    
    except ValueError:
        print("❌ 输入格式错误，请输入有效的数字")
    except KeyboardInterrupt:
        print("\n👋 程序已退出")


def demo_predictions():
    """演示预测功能"""
    print("🎯 遮阳帘预测演示")
    print("=" * 40)
    
    predictor = ShadeCurtainPredictor()
    
    if not predictor.load_model():
        print("❌ 请先运行 quick_predict() 训练模型")
        return
    
    # 演示数据
    demo_data = [
        (800, 35, 60, "强光照高温"),
        (200, 25, 70, "中等光照适宜温度"),
        (50, 20, 80, "低光照低温"),
        (600, 32, 50, "强光照高温低湿"),
        (300, 28, 75, "中等条件")
    ]
    
    print("演示不同环境条件下的预测结果:")
    print("-" * 50)
    
    for light, temp, humidity, desc in demo_data:
        result = predictor.predict(light, temp, humidity)
        
        if "error" not in result:
            status = "🌞开启" if result['should_open_shade'] else "🌙关闭"
            print(f"{desc:12} | 光照:{light:4.0f} 温度:{temp:4.1f}°C 湿度:{humidity:4.1f}% | {status} (概率:{result['probability']:.3f})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_predictions()
    else:
        quick_predict()
