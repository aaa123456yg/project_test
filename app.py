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
    使用 Librosa 分析音樂結構 (已更新為新版函式)
    """
    try:
        # 1. 載入音檔 (這一步最耗時)
        y, sr = librosa.load(file_path, sr=None)
        
        # 2. 計算 "Chroma" 特徵 (這一步也耗時)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        k = 3 # 嘗試找出 3 個主要的音樂段落
        
        # 3. 進行結構分段
        segment_labels = librosa.segment.agglomerative(chroma, k=k)
        
        # 4. 找出分段的邊界 (frames)
        boundaries_frames_intermediate = 1 + np.flatnonzero(segment_labels[:-1] != segment_labels[1:])
        
        # 5. (*** 這裡是修正的地方 ***)
        # 加上音樂的開頭 (0) 和結尾 (最後一幀)
        # 舊函式: boundaries_frames = librosa.util.pad_boundaries(...)
        # 新方法:
        start_frame = 0
        end_frame = chroma.shape[1]
        boundaries_frames = np.concatenate(([start_frame], boundaries_frames_intermediate, [end_frame]))

        # 6. 將 "frames" 轉換為 "time" (秒)
        # 找到 CQT 對應的 n_fft
        # (librosa 0.10.0 之後 n_fft 參數已移除，但舊版可能需要)
        try:
            fft_params = librosa.feature.chroma_cqt_options(sr=sr)
            n_fft = fft_params['n_fft']
            boundary_times = librosa.frames_to_time(boundaries_frames, sr=sr, n_fft=n_fft)
        except TypeError:
            # 處理新版 librosa (0.10.0+) 沒有 n_fft 的情況
            boundary_times = librosa.frames_to_time(boundaries_frames, sr=sr)

        # 7. 建立分段標籤
        segments = []
        labels = ['前奏 (Intro)', '主旋律 (Main)', '結尾 (Outro)']
        
        if len(boundary_times) == k + 1:
            for i in range(k):
                segments.append({
                    'label': labels[i],
                    'start': boundary_times[i],
                    'end': boundary_times[i+1]
                })
            return {'success': True, 'segments': segments}
        else:
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
        if 'music_file' not in request.files:
            return redirect(request.url)
        
        file = request.files['music_file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            difficulty = request.form.get('difficulty')
            
            # --- 核心：分析音樂 ---
            analysis_results = analyze_music_segments(filepath)
            
            return render_template('index.html', 
                                   analysis_results=analysis_results,
                                   filename=filename,
                                   difficulty=difficulty)

    return render_template('index.html', analysis_results=None)

# --- 執行 App ---
if __name__ == '__main__':
    app.run(debug=True)