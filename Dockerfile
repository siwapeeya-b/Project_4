# Dockerfile สำหรับ deploy ระบบรู้จำท่าทางภาษามือไทยขึ้น Hugging Face Spaces
FROM python:3.10-slim

# ไลบรารีระบบที่ MediaPipe/OpenCV ต้องใช้ (ไม่มีอินเทอร์เฟซกราฟิก)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง dependency ก่อน เพื่อให้ Docker cache ชั้นนี้ไว้ (build รอบถัดไปเร็วขึ้น)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดของโปรเจคเข้ามา
COPY . .

# Hugging Face Spaces (Docker SDK) ต้องรันที่พอร์ต 7860
ENV PORT=7860
EXPOSE 7860

# ใช้ eventlet worker เพื่อให้ Flask-SocketIO ทำงานแบบ async ไม่บล็อก connection อื่น
# ถ้าไฟล์หลักของคุณชื่ออื่น (ไม่ใช่ app.py) หรือ obj ไม่ใช่ชื่อ app ให้แก้ตรงนี้
CMD ["python", "app.py"]
