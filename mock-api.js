const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage });

// Mock video generation endpoint
app.post('/api/v1/generate/video', upload.single('file'), (req, res) => {
    console.log('Received file:', req.file);
    
    // Simulate processing delay
    setTimeout(() => {
        res.json({
            success: true,
            task_id: 'mock-task-' + Date.now(),
            message: 'Video generation started'
        });
    }, 1000);
});

// Mock status endpoint
app.get('/api/v1/status/:taskId', (req, res) => {
    res.json({
        id: req.params.taskId,
        status: 'completed',
        progress: 100,
        download_url: 'https://example.com/mock-video.mp4',
        error: null
    });
});

// Mock style presets endpoint
app.get('/api/v1/styles/presets', (req, res) => {
    res.json({
        professional: {
            styleType: 'professional',
            intensity: 7,
            culturalInfluence: 'neutral',
            mannerisms: ['formal', 'precise']
        },
        casual: {
            styleType: 'casual',
            intensity: 5,
            culturalInfluence: 'neutral',
            mannerisms: ['relaxed', 'friendly']
        },
        enthusiastic: {
            styleType: 'enthusiastic',
            intensity: 9,
            culturalInfluence: 'neutral',
            mannerisms: ['expressive', 'energetic']
        }
    });
});

app.listen(port, () => {
    console.log(`Mock API server running at http://localhost:${port}`);
    console.log(`- POST /api/v1/generate/video - Upload video`);
    console.log(`- GET  /api/v1/status/:taskId - Check status`);
    console.log(`- GET  /api/v1/styles/presets - Get style presets`);
});
