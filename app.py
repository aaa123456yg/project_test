import os
import librosa
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# --- 設定 ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 確保 'uploads' 資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 輔助函式 ---

def allowed_file(filename):
    """檢查檔案副檔名是否為 .mp3"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_music_segments(file_path):
    """
    使用 Librosa 分析音樂結構
    這是一個啟發式方法(heuristic)，嘗試將歌曲分為 3 個部分。
    """
    try:
        # 載入音檔
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. 計算 "Chroma" 特徵 (描述音高)
        # 這是分析音樂結構的常用特徵
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # 2. 進行結構分段
        # 我們設定 k=3，嘗試找出 3 個主要的音樂段落
        # 這會返回 4 個邊界點 (frames)，例如 [start, b1, b2, end]
        k = 3
        boundaries_frames = librosa.segment.structural_segmentation(chroma, k=k)
        
        # 3. 將 "frames" 轉換為 "time" (秒)
        boundary_times = librosa.frames_to_time(boundaries_frames, sr=sr)
        
        # 4. 建立分段標籤
        segments = []
        labels = ['前奏 (Intro)', '主旋律 (Main)', '結尾 (Outro)']
        
        # 檢查是否有成功分段 (應該要有 k+1 個邊界)
        if len(boundary_times) == k + 1:
            for i in range(k):
                segments.append({
                    'label': labels[i],
                    'start': boundary_times[i],
                    'end': boundary_times[i+1]
                })
            return {'success': True, 'segments': segments}
        else:
            # 如果分段失敗 (例如音樂太短或結構單一)
            # 就回傳整首歌
            duration = librosa.get_duration(y=y, sr=sr)
            segments.append({'label': '整首歌曲 (分段失敗)', 'start': 0.0, 'end': duration})
            return {'success': True, 'segments': segments}

    except Exception as e:
        print(f"Error analyzing file: {e}")
        return {'success': False, 'error': str(e)}

# --- 路由 (Routes) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis_results = None
    filename = None
    difficulty = None
    
    if request.method == 'POST':
        # 檢查是否有 'music_file' 在 request 中
        if 'music_file' not in request.files:
            # 這裡可以加入錯誤訊息
            return redirect(request.url)
        
        file = request.files['music_file']
        
        # 如果使用者沒有選擇檔案，瀏覽器會送出一個空檔案
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # 確保檔案名稱是安全的
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 儲存檔案
            file.save(filepath)
            
            # 取得選擇的難度
            difficulty = request.form.get('difficulty')
            
            # --- 核心：分析音樂 ---
            analysis_results = analyze_music_segments(filepath)
            
            # 為了下一個步驟，我們將結果傳回模板
            return render_template('index.html', 
                                   analysis_results=analysis_results,
                                   filename=filename,
                                   difficulty=difficulty)

    # GET 請求或初始載入
    return render_template('index.html', analysis_results=None)

# --- 執行 App ---
if __name__ == '__main__':
    app.run(debug=True)