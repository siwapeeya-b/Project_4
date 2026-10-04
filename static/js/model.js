const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const processingCanvas = document.createElement('canvas');
processingCanvas.width = 640;
processingCanvas.height = 480;
const processingCtx = processingCanvas.getContext('2d');

// 1. เชื่อมต่อ Socket.IO
const socket = io({
    transports: ['websocket'], 
    upgrade: false
});

// ดึงค่า MODEL_TYPE ที่ตั้งไว้ในหน้า HTML (ถ้านึกไม่ออกให้ Default เป็น 'lstm')
const CURRENT_MODEL_TYPE = window.MODEL_TYPE || 'lstm';

// 2. เช็กการเชื่อมต่อ Socket
socket.on('connect', () => {
    console.log("เชื่อมต่อกับ Flask Server สำเร็จ! (Socket ID:", socket.id, ")");
    console.log("ใช้โมเดลประเภท:", CURRENT_MODEL_TYPE);
});

socket.on('connect_error', (err) => {
    console.error("ไม่สามารถเชื่อมต่อ Socket.IO ได้:", err);
});

// [จุดที่แก้ไข 1]: พลิกแกน X (1.0 - l.x) ให้ข้อมูลพิกัดตรงตามความเป็นจริงของโมเดล
function extractKeypoints(results) {
    const lh = results.leftHandLandmarks 
        ? results.leftHandLandmarks.flatMap(l => [l.x, l.y, l.z]) 
        : new Array(63).fill(0);

    const rh = results.rightHandLandmarks 
        ? results.rightHandLandmarks.flatMap(l => [l.x, l.y, l.z]) 
        : new Array(63).fill(0);

    let pose = new Array(100).fill(0);
    if (results.poseLandmarks) {
        pose = results.poseLandmarks.slice(0, 25).flatMap(l => [l.x, l.y, l.z, l.visibility || 0]);
    }

    return [...lh, ...rh, ...pose]; 

}
// 4. ตั้งค่า MediaPipe Holistic
const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

holistic.onResults((results) => {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: 'rgba(255, 255, 255, 0.6)', lineWidth: 1 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FFFFFF', lineWidth: 1, radius: 1 });
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#38BDF8', lineWidth: 1 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#E0F2FE', lineWidth: 1, radius: 1 });
    }
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#34D399', lineWidth: 1 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#ECFDF5', lineWidth: 1, radius: 1 });
    }
    canvasCtx.restore();

    // สกัด keypoints และส่งข้อมูลไปยัง Python Server
    const keypoints = extractKeypoints(results);
    
    socket.emit('predict_keypoints', {
        keypoints: keypoints,
        model_type: CURRENT_MODEL_TYPE
    });
});

// 5. เปิดกล้อง Webcam
const camera = new Camera(videoElement, {
    onFrame: async () => {
        // ต้องพลิกก่อนประมวลผลให้เหมือนตอนเก็บข้อมูลด้วย cv2.flip(frame, 1)
        processingCtx.save();
        processingCtx.translate(processingCanvas.width, 0);
        processingCtx.scale(-1, 1);
        processingCtx.drawImage(videoElement, 0, 0, processingCanvas.width, processingCanvas.height);
        processingCtx.restore();
        await holistic.send({image: processingCanvas});
    },
    width: 640, 
    height: 480
});
camera.start();

// 6. รับผลการทำนายจาก Flask Server และอัปเดต Overlay/UI
socket.on('response', (data) => {
    console.log("ได้รับผลทำนาย:", data);

    const wordElem = document.getElementById('word');
    const confElem = document.getElementById('conf');

    if (wordElem) {
        wordElem.innerText = data.detected;
    }
    if (confElem) {
        confElem.innerText = typeof data.confidence === 'number' ? data.confidence.toFixed(2) : data.confidence;
    }

    //แสดงผล Sentence Bar และ Confidence List
    const sentenceElem = document.getElementById('sentence_bar');
    if (sentenceElem && data.sentence && data.sentence.length > 0) {
        sentenceElem.innerText = data.sentence.join('  ');
    }

    const probListElem = document.getElementById('prob_list');
    if (probListElem && data.probabilities && data.actions) {
        probListElem.innerHTML = '';
        const maxIndex = data.probabilities.indexOf(Math.max(...data.probabilities));

        data.actions.forEach((action, idx) => {
            const prob = data.probabilities[idx];
            if (prob > 2) {
                const item = document.createElement('div');
                item.className = 'prob-item' + (idx === maxIndex ? ' active' : '');
                item.innerText = `${action}`;
                
                if (idx === maxIndex) {
                    item.style.background = `linear-gradient(to right, #007bff ${prob}%, transparent ${prob}%)`;
                }

                probListElem.appendChild(item);
            }
        });
    }
});


