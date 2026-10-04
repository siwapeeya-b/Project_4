import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- 1. ตั้งค่าการเก็บข้อมูล ---
actions = np.array([
    '0',        # 1. ไข้
    '1',        # 2. รัก
    '2',        # 3. เสียใจ
    '3',        # 4. กิน
    '4',        # 5. สู้
    '5',        # 6. ขอโทษ
    '6',        # 7. ใคร
    '7',        # 8. เข้าใจ
    '8',        # 9. ที่ไหน
    '9'         # 10. ปกติ
])

no_sequences = 50 
sequence_length = 30  # ปรับเป็น 30 
data_path = "MP_Data12" 

for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(data_path, action, str(sequence)), exist_ok=True)

# --- 2. ฟังก์ชันดึง Keypoints (226 ค่าต่อเฟรม) ---
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    if results.pose_landmarks:
        upper_pose = results.pose_landmarks.landmark[0:25]
        ps = np.array([[res.x, res.y, res.z, res.visibility] for res in upper_pose]).flatten()
    else:
        ps = np.zeros(25*4)
        
    return np.concatenate([lh, rh, ps])

# --- 3. ตั้งค่า MediaPipe & OpenCV ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ฟังก์ชันสำหรับพักสายตา/พักเมื่อย
def pause_loop(cap):
    print("=== PAUSED: กด 'p' เพื่อเล่นต่อ ===")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # แสดงหน้าจอเตือนว่าพักอยู่
        cv2.rectangle(frame, (100, 180), (540, 300), (0, 0, 0), -1)
        cv2.putText(frame, 'PAUSED (REST TIME)', (140, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, 'Press "p" to Resume Work', (150, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('OpenCV Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            break
        elif key == ord('q'):
            raise KeyboardInterrupt

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            
            # --- หน้าจอรอเริ่มกด 's' เพื่อเริ่มคำศัพท์ใหม่ ---
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'READY TO COLLECT: "{action}"', (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, 'Press "s" Start | "p" Pause | "q" Quit', (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('OpenCV Feed', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    break
                elif key == ord('p'):
                    pause_loop(cap)
                elif key == ord('q'):
                    raise KeyboardInterrupt

            # --- เริ่มลูปเก็บข้อมูลตามจำนวน Sequence ---
            sequence = 0
            while sequence < no_sequences:
                
                # --- Countdown 2 วินาทีก่อนเริ่มอัดคลิป ---
                start_time = time.time()
                redo_flag = False
                
                while time.time() - start_time < 2.0:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    
                    # แสดงเส้น Landmark
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
                    time_left = 2 - int(time.time() - start_time)
                    cv2.putText(frame, f'GET READY: {time_left}...', (200, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f'Action: {action} | Sequence: {sequence+1}/{no_sequences}', (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame, '[p] Pause | [r] Redo Sequence', (15, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('OpenCV Feed', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        pause_loop(cap)
                        break
                    elif key == ord('r'):
                        redo_flag = True
                        break
                    elif key == ord('q'):
                        raise KeyboardInterrupt

                if redo_flag:
                    print(f"ทำรายการอัด Sequence {sequence+1} ใหม่อีกครั้ง...")
                    continue

                # --- บันทึก Sequence (30 เฟรม) ---
                temp_frames = []
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    
                    # วาดเส้น
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
                    # แสดงข้อความบันทึก
                    cv2.putText(frame, 'RECORDING...', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f'Action: {action} | Sequence: {sequence+1} | Frame: {frame_num+1}/{sequence_length}', 
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('OpenCV Feed', frame)

                    # สกัดพิกัดไว้ในแรมชั่วคราวก่อน
                    keypoints = extract_keypoints(results)
                    temp_frames.append(keypoints)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

                # บันทึกลง Disk เมื่อครบ 30 เฟรม
                if len(temp_frames) == sequence_length:
                    for frame_num, keypoints in enumerate(temp_frames):
                        npy_path = os.path.join(data_path, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)
                    sequence += 1 # ขยับไป Sequence ถัดไป

finally:
    print("ปิดกล้องและคืนทรัพยากรเรียบร้อย")
    cap.release()
    cv2.destroyAllWindows()