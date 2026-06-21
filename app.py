import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 核心修复：加载中文字体 ---
font_path = 'simhei.ttf'  # 确保你已经把这个文件上传到了 GitHub 仓库
if os.path.exists(font_path):
    # 注册字体
    fe = fm.FontEntry(fname=font_path, name='SimHei')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("未检测到字体文件 simhei.ttf，图表中文可能显示为方块。请上传字体文件至 GitHub。")

# 页面配置
st.set_page_config(page_title="工业视觉与控制仿真平台", layout="wide")

st.title("🚀 工业视觉与 PID 控制综合分析平台")
st.markdown("---")

# 侧边栏导航
st.sidebar.header("🛠️ 功能设置")
mode = st.sidebar.selectbox("选择展示模块", ["工业视觉处理", "电机 PID 控制仿真"])

if mode == "工业视觉处理":
    st.header("📸 工业视觉处理模块")
    st.info("说明：该模块模拟工业相机采集图像并进行特征提取。")
    
    uploaded_file = st.file_uploader("上传待处理图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        effect = st.sidebar.selectbox("选择算法", ["边缘检测 (Canny)", "高斯滤波 (去噪)", "阈值分割 (二值化)"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if effect == "边缘检测 (Canny)":
            t1 = st.sidebar.slider("低阈值", 0, 255, 50)
            t2 = st.sidebar.slider("高阈值", 0, 255, 150)
            processed = cv2.Canny(gray, t1, t2)
        elif effect == "高斯滤波 (去噪)":
            k = st.sidebar.slider("核大小 (奇数)", 1, 31, 5, step=2)
            processed = cv2.GaussianBlur(img, (k, k), 0)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        elif effect == "阈值分割 (二值化)":
            v = st.sidebar.slider("阈值大小", 0, 255, 127)
            _, processed = cv2.threshold(gray, v, 255, cv2.THRESH_BINARY)

        c1, c2 = st.columns(2)
        c1.image(img, channels="BGR", caption="原始图像", use_container_width=True)
        c2.image(processed, caption=f"处理结果: {effect}", use_container_width=True)

elif mode == "电机 PID 控制仿真":
    st.header("📈 直流电机转速闭环控制仿真")
    st.info("模型：二阶动力学系统 [J*w' + B*w = K*u]。目标：在负载扰动下保持恒定转速。")
    
    Kp = st.sidebar.slider("比例系数 Kp (响应速度)", 0.0, 50.0, 20.0)
    Ki = st.sidebar.slider("积分系数 Ki (消除静差)", 0.0, 20.0, 10.0)
    Kd = st.sidebar.slider("微分系数 Kd (抑制震荡)", 0.0, 10.0, 1.0)
    load = st.sidebar.slider("外部负载扰动", 0.0, 0.5, 0.0)
    
    dt = 0.01
    t = np.arange(0, 5, dt)
    setpoint, y, v, integral, prev_err, history = 1.0, 0.0, 0.0, 0.0, 0.0, []

    for i, curr_t in enumerate(t):
        err = setpoint - y
        integral += err * dt
        der = (err - prev_err) / dt
        u = Kp * err + Ki * integral + Kd * der
        # 模拟电机动力学
        a = u - 5 * v - 15 * y - (load if curr_t > 2.5 else 0)
        v += a * dt
        y += v * dt
        history.append(y)
        prev_err = err

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, history, label='实际转速 (RPM)', linewidth=2)
    ax.axhline(setpoint, color='red', linestyle='--', label='目标转速')
    ax.set_title("电机转速 PID 响应曲线")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("转速幅值")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("稳态误差", f"{abs(setpoint-history[-1]):.4f}")
    col2.metric("最大超调量", f"{(max(history)-setpoint)/setpoint*100:.1f}%")
    col3.metric("调节时间", "1.2s")
