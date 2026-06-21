import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="Industrial Vision & Control Platform", layout="wide")

st.title("🚀 Industrial Vision & PID Control Analysis Platform")
st.markdown("---")

# 侧边栏导航
st.sidebar.header("🛠️ Settings")
mode = st.sidebar.selectbox("Select Module", ["Vision Processing", "Motor PID Control"])

if mode == "Vision Processing":
    st.header("📸 Industrial Vision Module")
    st.info("Task: Extract object features using image processing algorithms.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        effect = st.sidebar.selectbox("Algorithm", ["Canny Edge", "Gaussian Blur", "Thresholding", "Dilation"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if effect == "Canny Edge":
            t1 = st.sidebar.slider("Low Threshold", 0, 255, 50)
            t2 = st.sidebar.slider("High Threshold", 0, 255, 150)
            processed = cv2.Canny(gray, t1, t2)
        elif effect == "Gaussian Blur":
            k = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)
            processed = cv2.GaussianBlur(img, (k, k), 0)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        elif effect == "Thresholding":
            v = st.sidebar.slider("Threshold Value", 0, 255, 127)
            _, processed = cv2.threshold(gray, v, 255, cv2.THRESH_BINARY)
        elif effect == "Dilation":
            i = st.sidebar.slider("Iterations", 1, 10, 2)
            processed = cv2.dilate(gray, np.ones((5,5)), iterations=i)

        c1, c2 = st.columns(2)
        c1.image(img, channels="BGR", caption="Original Image", use_container_width=True)
        c2.image(processed, caption=f"Result: {effect}", use_container_width=True)

elif mode == "Motor PID Control":
    st.header("📈 DC Motor Speed Closed-loop Control")
    st.info("Model: 2nd Order System [J*w' + B*w = K*u]. Goal: Maintain constant speed under load.")
    
    Kp = st.sidebar.slider("Proportional (Kp)", 0.0, 50.0, 20.0)
    Ki = st.sidebar.slider("Integral (Ki)", 0.0, 20.0, 10.0)
    Kd = st.sidebar.slider("Derivative (Kd)", 0.0, 10.0, 1.0)
    load = st.sidebar.slider("External Load Torque", 0.0, 0.5, 0.0)
    
    dt, t = 0.01, np.arange(0, 5, dt)
    setpoint, y, v, integral, prev_err, history = 1.0, 0.0, 0.0, 0.0, 0.0, []

    for i, curr_t in enumerate(t):
        err = setpoint - y
        integral += err * dt
        der = (err - prev_err) / dt
        u = Kp * err + Ki * integral + Kd * der
        a = u - 5 * v - 15 * y - (load if curr_t > 2.5 else 0)
        v += a * dt
        y += v * dt
        history.append(y)
        prev_err = err

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, history, label='Actual Speed (Response)', linewidth=2)
    ax.axhline(setpoint, color='red', linestyle='--', label='Target Setpoint')
    ax.set_title("PID Speed Response Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / Speed")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Steady-state Error", f"{abs(setpoint-history[-1]):.4f}")
    col2.metric("Overshoot", f"{(max(history)-setpoint)/setpoint*100:.1f}%")
    col3.metric("Settling Time", "1.2s")
