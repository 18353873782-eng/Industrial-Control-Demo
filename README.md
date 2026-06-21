# Industrial Vision & PID Control Analysis Platform

本项目是一个基于 Web 的工业自动化仿真平台，旨在展示“视觉感知 + 算法决策 + 闭环执行”的完整控制逻辑。

## 🚀 在线演示
https://sun-control-demo.streamlit.app/

## ✨ 核心功能
1. **工业视觉模块**：
   - 实现 Canny 边缘检测、高斯滤波、二值化分割等核心算法。
   - 模拟工业现场从复杂背景中提取目标零件特征的过程。
2. **电机 PID 控制仿真**：
   - 基于直流电机二阶动力学模型构建闭环系统。
   - 支持 Kp/Ki/Kd 参数实时整定，并量化分析超调量、稳态误差等性能指标。

## 🛠️ 技术栈
- **Language**: Python
- **Libraries**: OpenCV, NumPy, Matplotlib, Streamlit
- **Theory**: PID Control, Classical Control Theory, Digital Image Processing

## 📈 典型应用场景
该平台可用于模拟“智能分拣机器人”的视觉引导系统：通过视觉识别零件偏差，利用 PID 算法驱动执行机构完成精准纠偏。 
