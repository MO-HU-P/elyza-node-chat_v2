const express = require("express");
const axios = require("axios");
const path = require("path");
const multer = require("multer");
const fs = require('fs').promises;
const fsSync = require('fs');
const crypto = require('crypto');

const app = express();
const PORT = 3000;
const UPLOAD_DIR = 'uploads';

// ファイル操作に関する共通の関数
async function handleFiles(directory, operation = 'clean') {
    try {
        const files = await fs.readdir(directory);
        await Promise.all(files.map(file => {
            const filePath = path.join(directory, file);
            return fs.unlink(filePath)
                .then(() => console.log(`Deleted: ${filePath}`))
                .catch(err => console.error(`Error deleting ${filePath}:`, err));
        }));
        console.log(`${operation} completed successfully`);
    } catch (err) {
        console.error(`Error during ${operation}:`, err);
        if (operation === 'initialize') {
            process.exit(1);
        }
    }
}

// アップロードディレクトリの初期化関数
async function initializeUploadDirectory() {
    try {
        try {
            await fs.access(UPLOAD_DIR);
            await handleFiles(UPLOAD_DIR, 'initialize');
        } catch {
            await fs.mkdir(UPLOAD_DIR);
            console.log('Upload directory created');
        }
    } catch (error) {
        console.error('Error initializing upload directory:', error);
        process.exit(1);
    }
}

// 起動時にディレクトリを初期化
initializeUploadDirectory();

// ファイル名を安全な形式に変換する関数
function sanitizeFileName(originalname) {
    const ext = path.extname(originalname);
    const nameWithoutExt = path.basename(originalname, ext);
    const encodedName = Buffer.from(nameWithoutExt).toString('base64');
    const timestamp = Date.now();
    const randomString = crypto.randomBytes(4).toString('hex');
    return `${timestamp}-${randomString}-${encodedName}${ext}`;
}

// multerの設定
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        if (!fsSync.existsSync(UPLOAD_DIR)) {
            fsSync.mkdirSync(UPLOAD_DIR);
        }
        cb(null, UPLOAD_DIR);
    },
    // ファイル名を元のまま保持するように修正
    filename: (req, file, cb) => cb(null, file.originalname)
});

// 複数ファイルのアップロードに対応
const upload = multer({ storage: storage });


// シグナルハンドリング関数
async function gracefulShutdown(signal) {
    console.log(`Received ${signal}. Starting graceful shutdown...`);
    
    await handleFiles(UPLOAD_DIR, 'cleanup');
    
    if (server) {
        server.close(() => {
            console.log('HTTP server closed');
            process.exit(0);
        });

        setTimeout(() => {
            console.error('Forceful shutdown after timeout');
            process.exit(1);
        }, 10000);
    } else {
        process.exit(0);
    }
}

// シグナルハンドラの設定
['SIGINT', 'SIGTERM', 'SIGUSR2'].forEach(signal => {
    process.on(signal, () => gracefulShutdown(signal));
});

['uncaughtException', 'unhandledRejection'].forEach(event => {
    process.on(event, async (error) => {
        console.error(`${event}:`, error);
        await gracefulShutdown(event);
    });
});

// エクスプレスのミドルウェア設定
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.url}`);
    next();
});

app.use(express.static(path.join(__dirname, "public")));
app.use('/uploads', express.static(UPLOAD_DIR));
app.use(express.json());

// ファイルリスト取得のエンドポイントを追加
app.get("/api/files", async (req, res) => {
    try {
        const files = await fs.readdir(UPLOAD_DIR);
        res.json(files);
    } catch (error) {
        res.status(500).json({ error: "ファイル一覧の取得に失敗しました。" });
    }
});

// ファイル削除のエンドポイントを追加
app.delete("/api/files/:filename", async (req, res) => {
    try {
        await fs.unlink(path.join(UPLOAD_DIR, req.params.filename));
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: "ファイルの削除に失敗しました。" });
    }
});

// ファイルアップロードのエンドポイントを複数ファイル対応に修正
app.post("/api/upload", upload.array('files'), (req, res) => {
    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: "ファイルがアップロードされていません。" });
    }

    res.json(req.files.map(file => ({
        originalname: file.originalname,
        filename: file.filename,
        path: `/uploads/${file.filename}`
    })));
});

app.post("/api/send-message", async (req, res) => {
    console.log("Received message:", req.body.message);

    try {
        const response = await axios.post("http://localhost:8501/api/chat", {
            message: req.body.message,
            attachment: req.body.attachment
        }, {
            headers: { 'Content-Type': 'application/json' }
        });

        console.log("Response from Python backend:", response.data);
        res.json(response.data);

    } catch (error) {
        console.error("Error details:", {
            message: error.message,
            code: error.code,
            response: error.response?.data
        });
        
        res.status(error.code === 'ECONNREFUSED' ? 503 : 500)
           .json({ error: error.code === 'ECONNREFUSED' 
                         ? "Python バックエンドに接続できません。" 
                         : "エラーが発生しました。" });
    }
});

const server = app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});