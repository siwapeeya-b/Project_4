import os
import json
import h5py
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
# ระบุ async_mode='threading' ตรงๆ แทนการให้ auto-detect
# (ปลอดภัยกว่าเวลาใช้ร่วมกับ TensorFlow ซึ่งมักขัดแย้งกับ eventlet/gevent)
# เพิ่ม ping_timeout ให้มี buffer มากขึ้น เผื่อ predict หน่วงเล็กน้อย จะได้ไม่หลุดง่าย
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
)

# TensorFlow .predict() ไม่ thread-safe 100% เวลามีหลาย session เรียกพร้อมกันบนโมเดลเดียวกัน
# ใช้ lock กันไม่ให้สอง thread เรียก predict() บนโมเดลตัวเดียวกันพร้อมกัน
predict_lock = threading.Lock()

actions = np.array([
    'ป่วย', 'รัก', 'เสียใจ', 'กิน', 'ให้กำลังใจ', 
    'ขอโทษ', 'ใคร', 'เข้าใจ', 'ที่ไหน', 'ท่าปกติ'
])

#ตั้งค่า Threshold ความมั่นใจขั้นต่ำ (0.7 = 70%) เพื่อกรองคำเดาสุ่มออก
THRESHOLD = 0.4

# 2. ฟังก์ชัน Normalization
def normalize_frame_keypoints(keypoint_frame):
    keypoint_frame = np.array(keypoint_frame)
    if len(keypoint_frame) < 226:
        return np.zeros(226)
        
    left_hand = keypoint_frame[0:63].reshape(21, 3).copy()
    right_hand = keypoint_frame[63:126].reshape(21, 3).copy()
    pose = keypoint_frame[126:226].reshape(25, 4).copy()

    # Normalization (คำนวณระยะห่างตามปกติ)
    pose_xyz = pose[:, :3]
    if np.any(pose_xyz):
        center = (pose_xyz[11] + pose_xyz[12]) / 2.0 
        dist = np.linalg.norm(pose_xyz[11] - pose_xyz[12])
        if dist == 0: dist = 1e-6
    else:
        center = np.zeros(3)
        dist = 1.0

    pose[:, :3] = (pose[:, :3] - center) / dist
    left_hand = (left_hand - center) / dist
    right_hand = (right_hand - center) / dist
    return np.concatenate([left_hand.flatten(), right_hand.flatten(), pose.flatten()])
# =========================================================================
# 🔧 ฟังก์ชันล้าง Config ส่วนเกินของ Keras 3 ในระดับ JSON
# =========================================================================
def clean_keras3_config(obj):
    """ลบค่าพารามิเตอร์ของ Keras 3 เช่น DTypePolicy, quantization_config ฯลฯ"""
    if isinstance(obj, dict):
        if 'dtype' in obj and isinstance(obj['dtype'], dict):
            obj['dtype'] = 'float32'
            
        for key_to_remove in ['quantization_config', 'optional', 'registered_name']:
            obj.pop(key_to_remove, None)
            
        if 'batch_shape' in obj:
            batch_shape = obj.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1 and 'input_shape' not in obj:
                obj['input_shape'] = batch_shape[1:]

        if 'module' in obj and 'class_name' in obj:
            obj.pop('module', None)
            
        return {k: clean_keras3_config(v) for k, v in obj.items()}
        
    elif isinstance(obj, list):
        return [clean_keras3_config(item) for item in obj]
        
    return obj

def load_keras3_h5_model(h5_path):
    """อ่านโมเดล Keras 3 จากไฟล์ .h5 โดยสร้าง Architecture ใหม่และโหลดเฉพาะ Weights"""
    with h5py.File(h5_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError("ไม่พบ model_config ในไฟล์ H5")
            
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
            
        config_dict = json.loads(model_config)
        
    cleaned_config = clean_keras3_config(config_dict)
    json_str = json.dumps(cleaned_config)
    model = tf.keras.models.model_from_json(json_str)
    model.load_weights(h5_path)
    return model
# =========================================================================

# โหลดโมเดลทั้ง 
models = {}
model_files = {
    'lstm': 'best_model_dropout0.2_epoch10.h5'
}

for name, path in model_files.items():
    full_path = os.path.join('model', path) if os.path.exists(os.path.join('model', path)) else path
    
    if os.path.exists(full_path):
        try:
            models[name] = load_keras3_h5_model(full_path)
            print(f"โหลดโมเดล {name} ({full_path}) สำเร็จแล้ว!")
        except Exception as e:
            try:
                models[name] = tf.keras.models.load_model(full_path, compile=False)
                print(f"โหลดโมเดล {name} (โหมดสำรอง) สำเร็จแล้ว!")
            except Exception as e2:
                print(f"ไม่สามารถโหลดโมเดล {name} ได้: {e2}")
    else:
        print(f"หาไฟล์โมเดล {full_path} ไม่เจอ! โปรดเช็กว่าวางไฟล์ชื่อ '{path}' ไว้ถูกที่หรือไม่")

# Warm-up โมเดล: เรียก predict() หลอกๆ ครั้งหนึ่งตอน start server
# เพื่อให้การ trace graph/เตรียม kernel เกิดขึ้นตอนนี้ ไม่ใช่ตอนผู้ใช้คนแรกทำนายจริง
# (ไม่งั้นการทำนายครั้งแรกสุดของทุกโมเดลจะช้าผิดปกติ ดูเหมือนค้าง)
_dummy_input = np.zeros((1, 30, 226), dtype=np.float32)
for name, model in models.items():
    try:
        model.predict(_dummy_input, verbose=0)
        print(f"Warm-up โมเดล {name} เรียบร้อยแล้ว")
    except Exception as e:
        print(f"Warm-up โมเดล {name} ล้มเหลว: {e}")

user_data = {}
# --- Routes ---
@app.route('/')
def model1():
    return render_template('home_page.html')

@app.route('/model')
def model2():
    return render_template('model.html')

# --- Socket Events ---
@socketio.on('connect')
def handle_connect():
    #สร้างบัฟเฟอร์เก็บ Sequence, Predictions History และ Sentence สำหรับ Session นี้
    user_data[request.sid] = {
        'sequence': [],
        'predictions': [],
        'sentence': [],
        'is_predicting': False,  # กันไม่ให้ session เดียวกันยิง predict ซ้อนกันจนเฟรมกองค้าง
    }

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in user_data:
        del user_data[request.sid]

@socketio.on('predict_keypoints')
def handle_prediction(data):
    sid = request.sid
    if sid not in user_data:
        user_data[sid] = {'sequence': [], 'predictions': [], 'sentence': [], 'is_predicting': False}

    if isinstance(data, dict):
        raw_keypoints = data.get('keypoints', [])
        model_type = data.get('model_type', 'lstm')
    else:
        raw_keypoints = data
        model_type = 'lstm'
    
    norm_keypoints = normalize_frame_keypoints(raw_keypoints)
    
    # ดึง State ของผู้ใช้ปัจจุบัน
    user_state = user_data[sid]
    user_state['sequence'].append(norm_keypoints)
    user_state['sequence'] = user_state['sequence'][-30:] # รักษาขนาดบัฟเฟอร์ 30 เฟรม

    # ทำนายเฉพาะตอนบัฟเฟอร์เต็ม และ session นี้ไม่มีการทำนายค้างอยู่ก่อนแล้ว
    # (ถ้าไม่เช็ค is_predicting จะยิง predict() ทุกเฟรมที่เข้ามาซ้อนกันเรื่อยๆ จนค้าง)
    if len(user_state['sequence']) == 30 and not user_state['is_predicting']:
        if model_type in models:
            user_state['is_predicting'] = True
            sequence_snapshot = list(user_state['sequence'])  # ก็อปปี้ไว้ก่อนส่งเข้า background task
            socketio.start_background_task(
                run_prediction, sid, sequence_snapshot, model_type
            )


def run_prediction(sid, sequence, model_type):
    """รันบน background thread แยกจาก thread ที่คุย socket โดยตรง
    เพื่อไม่ให้การทำนาย (ซึ่งอาจใช้เวลาหลาย 100ms) ไปบล็อกการตอบ ping/pong
    ของ client จนหลุดการเชื่อมต่อ"""
    user_state = user_data.get(sid)
    if user_state is None:
        return  # client หลุดไปแล้วระหว่างรอคิว

    target_model = models.get(model_type)
    if target_model is None:
        user_state['is_predicting'] = False
        return

    try:
        input_data = np.expand_dims(sequence, axis=0)
        # lock กันสอง session เรียก predict() พร้อมกันบนโมเดลตัวเดียวกัน
        with predict_lock:
            res = target_model.predict(input_data, verbose=0)[0]

        predicted_idx = np.argmax(res)
        confidence = float(res[predicted_idx])

        if confidence >= THRESHOLD:
            user_state['predictions'].append(predicted_idx)
            user_state['predictions'] = user_state['predictions'][-10:] # เก็บ 10 ผลลัพธ์ล่าสุด

            # ถ้าโมเดลทายได้คำเดิมซ้ำกันเกิน 6 ใน 10 เฟรมล่าสุด ให้ถือว่าเป็นคำที่นิ่งแล้ว
            if user_state['predictions'].count(predicted_idx) >= 4:
                detected_word = actions[predicted_idx]

                # อัปเดตใส่ Sentence (ไม่ใส่คำเดิมซ้ำติดกัน)
                if len(user_state['sentence']) > 0:
                    if detected_word != user_state['sentence'][-1]:
                        user_state['sentence'].append(detected_word)
                else:
                    user_state['sentence'].append(detected_word)
            else:
                detected_word = "..."
        else:
            detected_word = "..."

        # จำกัดความยาวประโยคสูงสุด 5 คำ
        if len(user_state['sentence']) > 5:
            user_state['sentence'] = user_state['sentence'][-5:]

        # เรียก emit แบบนอก request context ต้องระบุ to=sid เสมอ
        socketio.emit('response', {
            'detected': detected_word,
            'confidence': round(confidence * 100, 2),
            'sentence': user_state['sentence'],
            'probabilities': (res * 100).round(2).tolist(),
            'actions': actions.tolist(),
            'model_used': model_type
        }, to=sid)
    finally:
        if user_state is not None:
            user_state['is_predicting'] = False

if __name__ == "__main__":
    # Hugging Face Spaces (Docker SDK) กำหนดพอร์ตผ่าน env var PORT (ค่า default คือ 7860)
    # ต้อง bind host="0.0.0.0" ไม่งั้น container จะมองไม่เห็น service จากภายนอก
    port = int(os.environ.get("PORT", 5000))
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,  # จำเป็นสำหรับรัน Werkzeug dev server ใน container
    )