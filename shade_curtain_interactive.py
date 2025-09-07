#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遮阳帘预测系统 - 交互式预测程序
用户可以通过输入环境数据来预测是否需要开启遮阳帘
"""

import sys
import os
from shade_curtain_predictor import ShadeCurtainPredictor


def print_banner():
    """打印程序标题"""
    print("=" * 60)
    print("🌞 遮阳帘智能预测系统 🌞")
    print("基于决策树回归算法的环境数据分析")
    print("=" * 60)


def print_menu():
    """打印主菜单"""
    print("\n请选择操作:")
    print("1. 预测遮阳帘状态")
    print("2. 批量预测")
    print("3. 查看模型信息")
    print("4. 重新训练模型")
    print("5. 退出程序")
    print("-" * 40)


def get_user_input():
    """获取用户输入的环境数据"""
    print("\n请输入当前环境数据:")
    
    try:
        light_intensity = float(input("光照强度 (0-1000): "))
        temperature = float(input("温度 (°C): "))
        humidity = float(input("湿度 (%): "))
        
        # 可选输入时间
        hour_input = input("当前小时 (0-23, 可选，直接回车跳过): ").strip()
        hour = None
        if hour_input:
            hour = int(hour_input)
            if not (0 <= hour <= 23):
                print("警告: 小时数应在0-23之间，将忽略此输入")
                hour = None
        
        return light_intensity, temperature, humidity, hour
    
    except ValueError as e:
        print(f"输入错误: {e}")
        return None


def predict_single(predictor):
    """单次预测"""
    print("\n🔍 单次预测模式")
    user_input = get_user_input()
    
    if user_input is None:
        return
    
    light_intensity, temperature, humidity, hour = user_input
    
    # 进行预测
    result = predictor.predict(light_intensity, temperature, humidity, hour)
    
    if "error" in result:
        print(f"❌ 预测失败: {result['error']}")
        return
    
    # 显示预测结果
    print("\n" + "=" * 50)
    print("📊 预测结果")
    print("=" * 50)
    print(f"输入数据:")
    print(f"  光照强度: {result['input_features']['light_intensity']}")
    print(f"  温度: {result['input_features']['temperature']}°C")
    print(f"  湿度: {result['input_features']['humidity']}%")
    if result['input_features']['hour'] is not None:
        print(f"  时间: {result['input_features']['hour']}:00")
    
    print(f"\n预测结果:")
    status_icon = "🌞" if result['should_open_shade'] else "🌙"
    print(f"  遮阳帘状态: {status_icon} {'开启' if result['should_open_shade'] else '关闭'}")
    print(f"  推荐操作: {result['recommendation']}")
    print(f"  预测概率: {result['probability']:.3f}")
    print(f"  置信度: {result['confidence']:.3f}")
    
    # 给出详细解释
    print(f"\n💡 分析说明:")
    if result['should_open_shade']:
        print("  当前环境条件建议开启遮阳帘，原因可能包括:")
        if light_intensity > 500:
            print(f"    - 光照强度较高 ({light_intensity})")
        if temperature > 30:
            print(f"    - 温度较高 ({temperature}°C)")
        print("    - 开启遮阳帘可以降低室内温度和光照强度")
    else:
        print("  当前环境条件建议关闭遮阳帘，原因可能包括:")
        if light_intensity <= 500:
            print(f"    - 光照强度适中 ({light_intensity})")
        if temperature <= 30:
            print(f"    - 温度适宜 ({temperature}°C)")
        print("    - 保持自然光照有利于植物生长")


def predict_batch(predictor):
    """批量预测"""
    print("\n📋 批量预测模式")
    print("请输入多组环境数据，每行一组，格式: 光照强度,温度,湿度[,小时]")
    print("输入 'done' 结束输入")
    
    predictions = []
    
    while True:
        line = input("\n请输入数据 (或输入 'done' 结束): ").strip()
        
        if line.lower() == 'done':
            break
        
        try:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 3:
                print("❌ 输入格式错误，至少需要光照强度,温度,湿度")
                continue
            
            light_intensity = float(parts[0])
            temperature = float(parts[1])
            humidity = float(parts[2])
            hour = int(parts[3]) if len(parts) > 3 else None
            
            result = predictor.predict(light_intensity, temperature, humidity, hour)
            
            if "error" not in result:
                predictions.append(result)
                status = "开启" if result['should_open_shade'] else "关闭"
                print(f"✅ 预测完成: 遮阳帘{status} (概率: {result['probability']:.3f})")
            else:
                print(f"❌ 预测失败: {result['error']}")
        
        except ValueError:
            print("❌ 输入格式错误，请检查数字格式")
        except Exception as e:
            print(f"❌ 处理错误: {e}")
    
    # 显示批量预测结果汇总
    if predictions:
        print(f"\n📊 批量预测结果汇总 (共{len(predictions)}组)")
        print("-" * 60)
        
        open_count = sum(1 for p in predictions if p['should_open_shade'])
        close_count = len(predictions) - open_count
        
        print(f"建议开启遮阳帘: {open_count} 组")
        print(f"建议关闭遮阳帘: {close_count} 组")
        print(f"开启比例: {open_count/len(predictions)*100:.1f}%")
        
        # 显示详细结果
        print(f"\n详细结果:")
        for i, pred in enumerate(predictions, 1):
            status = "🌞开启" if pred['should_open_shade'] else "🌙关闭"
            print(f"  {i:2d}. {status} - 光照:{pred['input_features']['light_intensity']:6.1f}, "
                  f"温度:{pred['input_features']['temperature']:5.1f}°C, "
                  f"湿度:{pred['input_features']['humidity']:5.1f}% "
                  f"(概率:{pred['probability']:.3f})")


def show_model_info(predictor):
    """显示模型信息"""
    print("\n📋 模型信息")
    print("-" * 40)
    
    if predictor.model is None:
        print("❌ 模型未加载")
        return
    
    # 尝试加载模型信息文件
    try:
        import json
        if os.path.exists(predictor.model_info_path):
            with open(predictor.model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            print(f"模型类型: {model_info.get('model_type', 'Unknown')}")
            print(f"训练时间: {model_info.get('training_time', 'Unknown')}")
            print(f"特征数量: {model_info.get('n_features', 'Unknown')}")
            
            print(f"\n特征重要性:")
            feature_importance = model_info.get('feature_importance', {})
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")
        else:
            print("模型信息文件不存在")
            print(f"模型已加载，特征数量: {len(predictor.model.feature_names_in_)}")
    
    except Exception as e:
        print(f"读取模型信息失败: {e}")


def retrain_model(predictor):
    """重新训练模型"""
    print("\n🔄 重新训练模型")
    print("这将使用最新的数据重新训练模型...")
    
    confirm = input("确认重新训练? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消重新训练")
        return
    
    try:
        # 导入训练函数
        from shade_curtain_predictor import main as train_main
        train_main()
        print("✅ 模型重新训练完成")
        
        # 重新加载模型
        if predictor.load_model():
            print("✅ 新模型已加载")
        else:
            print("❌ 新模型加载失败")
    
    except Exception as e:
        print(f"❌ 重新训练失败: {e}")


def main():
    """主程序"""
    print_banner()
    
    # 初始化预测器
    predictor = ShadeCurtainPredictor()
    
    # 尝试加载已训练的模型
    print("正在加载模型...")
    if not predictor.load_model():
        print("❌ 模型加载失败，请先运行 shade_curtain_predictor.py 训练模型")
        print("或者选择菜单选项4重新训练模型")
    
    # 主循环
    while True:
        print_menu()
        
        try:
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                predict_single(predictor)
            elif choice == '2':
                predict_batch(predictor)
            elif choice == '3':
                show_model_info(predictor)
            elif choice == '4':
                retrain_model(predictor)
            elif choice == '5':
                print("\n👋 感谢使用遮阳帘预测系统！")
                break
            else:
                print("❌ 无效选择，请输入1-5之间的数字")
        
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 程序错误: {e}")
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()
