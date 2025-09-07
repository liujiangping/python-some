# 遮阳帘智能预测系统

基于决策树回归算法的环境数据分析系统，用于预测是否需要开启遮阳帘。

## 🌟 功能特点

- **智能预测**: 基于光照强度、温度、湿度等环境数据预测遮阳帘状态
- **决策树回归**: 使用scikit-learn的决策树回归算法，准确率高
- **交互式界面**: 支持单次预测和批量预测
- **模型持久化**: 自动保存和加载训练好的模型
- **特征工程**: 包含时间特征和交互特征，提高预测准确性

## 📁 文件结构

```
├── shade_curtain_predictor.py      # 核心预测器类
├── shade_curtain_interactive.py    # 交互式预测程序
├── quick_shade_predictor.py        # 快速预测程序
├── weekly_environment_data_*.csv   # 环境数据文件
├── models/                         # 模型存储目录
│   ├── shade_curtain_model.pkl     # 训练好的模型
│   └── shade_curtain_model_info.json # 模型信息
└── requirements.txt                # 依赖包列表
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖（如果尚未安装）
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用环境数据训练模型
python3 shade_curtain_predictor.py
```

训练完成后，模型将自动保存到 `models/` 目录。

### 3. 使用预测功能

#### 方式一：快速预测
```bash
# 运行快速预测程序
python3 quick_shade_predictor.py

# 或运行演示
python3 quick_shade_predictor.py demo
```

#### 方式二：交互式预测
```bash
# 运行完整的交互式程序
python3 shade_curtain_interactive.py
```

## 📊 数据格式

### 输入数据格式
系统需要包含以下列的环境数据CSV文件：
- `timestamp`: 时间戳
- `light_intensity`: 光照强度 (0-1000)
- `temperature`: 温度 (°C)
- `humidity`: 湿度 (%)

### 预测输入
用户需要输入：
- 光照强度 (0-1000)
- 温度 (°C)
- 湿度 (%)
- 时间 (可选，0-23小时)

## 🎯 预测逻辑

系统基于以下规则创建训练目标：
- 光照强度 > 500 或 温度 > 30°C → 需要开启遮阳帘
- 其他情况 → 不需要开启遮阳帘

模型使用决策树回归算法，综合考虑：
- 基础特征：光照强度、温度、湿度
- 时间特征：小时、星期几
- 交互特征：光照×温度、温度×湿度

## 📈 模型性能

根据测试数据，模型表现：
- **训练准确率**: 100%
- **测试准确率**: 100%
- **R² 分数**: 1.0
- **MSE**: 0.0

### 特征重要性
1. 光照温度交互特征: 99.5%
2. 温度: 0.5%
3. 其他特征: <0.1%

## 💡 使用示例

### 单次预测示例
```python
from shade_curtain_predictor import ShadeCurtainPredictor

# 初始化预测器
predictor = ShadeCurtainPredictor()
predictor.load_model()

# 预测
result = predictor.predict(
    light_intensity=800,  # 强光照
    temperature=35,       # 高温
    humidity=60,          # 中等湿度
    hour=14              # 下午2点
)

print(f"建议: {result['recommendation']}")
print(f"概率: {result['probability']:.3f}")
```

### 批量预测示例
```python
# 批量预测多组数据
data_list = [
    (800, 35, 60),  # 强光照高温
    (200, 25, 70),  # 中等条件
    (50, 20, 80),   # 低光照低温
]

for light, temp, humidity in data_list:
    result = predictor.predict(light, temp, humidity)
    status = "开启" if result['should_open_shade'] else "关闭"
    print(f"光照:{light}, 温度:{temp}°C, 湿度:{humidity}% → 遮阳帘{status}")
```

## 🔧 自定义配置

### 修改预测阈值
在 `shade_curtain_predictor.py` 中修改：
```python
# 在 preprocess_data 方法中
light_threshold = 500  # 光照强度阈值
temp_threshold = 30    # 温度阈值
```

### 调整模型参数
在 `train_model` 方法中修改：
```python
self.model = DecisionTreeRegressor(
    max_depth=10,           # 最大深度
    min_samples_split=5,    # 最小分割样本数
    min_samples_leaf=2,     # 叶节点最小样本数
    random_state=42         # 随机种子
)
```

## 🐛 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'pandas'**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **模型文件不存在**
   ```bash
   python3 shade_curtain_predictor.py  # 先训练模型
   ```

3. **数据文件格式错误**
   - 确保CSV文件包含必需的列：timestamp, light_intensity, temperature, humidity
   - 检查数据格式是否正确

4. **预测结果不准确**
   - 检查输入数据范围是否合理
   - 考虑重新训练模型
   - 调整预测阈值

## 📝 更新日志

- **v1.0.0**: 初始版本，支持基本的遮阳帘预测功能
- 支持决策树回归算法
- 支持交互式和快速预测模式
- 支持模型持久化

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。
