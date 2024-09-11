console.log('🎥 Initializing webcam stream and facial tracking...');

const video_element = document.getElementById('webcam');
const canvas_element = document.getElementById('facial-landmarks');
const webcam_container = document.getElementById('webcam-container');

async function load_face_api_models() {
    console.log('🔧 Loading face-api models...');
    await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models'),
    ]);
    console.log('✅ Face-api models loaded.');
}

function update_canvas_size() {
    console.log('📏 Updating canvas size...');
    const display_size = { width: window.innerWidth, height: window.innerHeight };
    faceapi.matchDimensions(canvas_element, display_size);
    console.log('✅ Canvas size updated.');
}

async function initialize_webcam() {
    try {
        console.log('🔍 Requesting webcam access...');
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        console.log('✅ Webcam access granted.');
        
        console.log('🔗 Connecting stream to video element...');
        video_element.srcObject = stream;
        console.log('✅ Stream connected to video element.');

        video_element.addEventListener('play', () => {
            console.log('▶️ Video playback started.');
            update_canvas_size();
            
            window.addEventListener('resize', update_canvas_size);
            
            setInterval(async () => {
                console.log('🔍 Detecting faces...');
                const detections = await faceapi.detectAllFaces(video_element, new faceapi.TinyFaceDetectorOptions())
                    .withFaceLandmarks();
                console.log(`✅ Detected ${detections.length} face(s).`);
                
                const resized_detections = faceapi.resizeResults(detections, { width: canvas_element.width, height: canvas_element.height });
                canvas_element.getContext('2d').clearRect(0, 0, canvas_element.width, canvas_element.height);
                faceapi.draw.drawFaceLandmarks(canvas_element, resized_detections);
            }, 100);
        });
    } catch (error) {
        console.error('❌ Error accessing webcam:', error);
    }
}

async function start_application() {
    console.log('🚀 Starting application...');
    await load_face_api_models();
    await initialize_webcam();
}

start_application();