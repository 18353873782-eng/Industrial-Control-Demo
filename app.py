import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 解决 Matplotlib 中文乱码问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(page_title="工业视觉与电机控制仿真平台", layout="wide")

st.title("🚀 工业视觉与闭环控制综合分析平台")
st.markdown("---")

# 侧边栏导航
st.sidebar.header("🛠️ 功能选择")
mode = st.sidebar.selectbox("请选择展示模块", ["高级图像处理模块", "电机转速PID控制仿真"])

if mode == "高级图像处理模块":
    st.header("📸 工业视觉多功能处理模块")
    st.info("说明：本模块集成了工业视觉预处理中的核心算法，用于提升图像质量及提取目标特征。")

    uploaded_file = st.file_uploader("请上传待处理图片", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # 侧边栏特效选择
        st.sidebar.subheader("算法选择")
        effect = st.sidebar.selectbox("选择处理特效",
                                      ["边缘检测 (Canny)", "高斯滤波 (去噪)", "阈值分割 (二值化)", "形态学膨胀 (增强)"])

        # 图像处理逻辑
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if effect == "边缘检测 (Canny)":
            t1 = st.sidebar.slider("低阈值", 0, 255, 50)
            t2 = st.sidebar.slider("高阈值", 0, 255, 150)
            processed = cv2.Canny(gray, t1, t2)
            desc = "提取图像中的轮廓信息，常用于零件尺寸测量。"

        elif effect == "高斯滤波 (去噪)":
            k_size = st.sidebar.slider("核大小 (必须是奇数)", 1, 31, 5, step=2)
            processed = cv2.GaussianBlur(img, (k_size, k_size), 0)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)  # 保持彩色
            desc = "平滑图像，去除工业环境中的高频噪声。"

        elif effect == "阈值分割 (二值化)":
            thresh_val = st.sidebar.slider("阈值", 0, 255, 127)
            _, processed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            desc = "将图像黑白化，用于将目标零件从背景中分离。"

        elif effect == "形态学膨胀 (增强)":
            iter_val = st.sidebar.slider("膨胀强度", 1, 10, 2)
            kernel = np.ones((5, 5), np.uint8)
            processed = cv2.dilate(gray, kernel, iterations=iter_val)
            desc = "放大图像中的高亮区域，填补边缘细小空隙。"

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="原始采集图像", use_container_width=True)
        with col2:
            # 判断是彩色还是灰度显示
            if effect == "高斯滤波 (去噪)":
                st.image(processed, caption=f"{effect} 结果", use_container_width=True)
            else:
                st.image(processed, caption=f"{effect} 结果", use_container_width=True)
            st.success(f"算法说明：{desc}")

elif mode == "电机转速PID控制仿真":
    st.header("📈 直流电机转速闭环控制仿真")
    st.info("说明：本模块模拟直流电机在不同负载下的转速控制。PID控制器通过调节电压，使电机转速快速、稳定地达到设定目标。")

    # PID 参数调节
    st.sidebar.subheader("PID 控制器参数整定")
    Kp = st.sidebar.slider("比例系数 Kp (提升响应速度)", 0.0, 50.0, 20.0)
    Ki = st.sidebar.slider("积分系数 Ki (消除静差)", 0.0, 20.0, 10.0)
    Kd = st.sidebar.slider("微分系数 Kd (减少震荡)", 0.0, 10.0, 1.0)

    # 模拟负载干扰
    st.sidebar.subheader("外部干扰设置")
    load_torque = st.sidebar.slider("模拟负载扰动 (Load)", 0.0, 0.5, 0.0)

    # 仿真逻辑
    dt = 0.01
    t = np.arange(0, 5, dt)
    setpoint = 1.0  # 设定转速为 1.0 (标幺值)
    y = 0.0  # 初始转速
    v = 0.0  # 初始加速度
    integral = 0.0
    prev_error = 0.0
    history = []

    for i in t:
        error = setpoint - y
        integral += error * dt
        derivative = (error - prev_error) / dt
        u = Kp * error + Ki * integral + Kd * derivative

        # 模拟电机动力学模型 (受控对象: J*w' + B*w = K*u - Load)
        # 简化模型：a = 电压输入 - 阻尼 - 负载
        a = u - 5 * v - 15 * y - (load_torque if i > 2 else 0)  # 2秒后加入负载
        v += a * dt
        y += v * dt
        history.append(y)
        prev_error = error

    # 绘图展示
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, history, label='电机实际转速 (RPM)', color='#1f77b4', linewidth=2)
    ax.axhline(setpoint, color='red', linestyle='--', label='目标设定转速')

    if load_torque > 0:
        ax.annotate('突加负载干扰', xy=(2, history[200]), xytext=(2.5, history[200] - 0.3),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlabel('时间 Time (s)')
    ax.set_ylabel('转速 Amplitude')
    ax.set_title('直流电机转速 PID 闭环响应曲线')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

    st.pyplot(fig)

    # 性能指标量化
    c1, c2, c3 = st.columns(3)
    c1.metric("稳态误差 (精度)", f"{abs(setpoint - history[-1]):.4f}")
    c2.metric("调节时间 (速度)", f"{next((i * dt for i, v in enumerate(history) if abs(v - setpoint) < 0.02), 5):.2f}s")
    c3.metric("最大超调量 (稳定性)", f"{(max(history) - setpoint) / setpoint * 100:.1f}%")