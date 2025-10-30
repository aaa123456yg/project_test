import os
import librosa
import numpy as np 
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Matplotlib 設定 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

# --- Flask 設定 ---
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_very_secret_key_for_flash'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'plots'), exist_ok=True)

# --- 輔助函式 ---

def allowed_file(filename):
    """檢查檔案副檔名是否為 .mp3"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_music_segments(file_path):
    """
    使用 Librosa 分析音樂結構
    """
    try:
        # 1. 載入
        y, sr = librosa.load(file_path, sr=None)
        # 2. 計算特徵
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        k = 3
        
        # 3. 結構分段
        segment_labels = librosa.segment.agglomerative(chroma, k=k)
        
        # 4. 找出邊界 (frames)
        boundaries_frames_intermediate = 1 + np.flatnonzero(segment_labels[:-1] != segment_labels[1:])
        
        # 5. 加上開頭 (0) 和結尾 (最後一幀)
        start_frame = 0
        end_frame = chroma.shape[1]
        boundaries_frames = np.concatenate(([start_frame], boundaries_frames_intermediate, [end_frame]))

        # 6. 將 "frames" 轉換為 "time" (秒)
        # *** 這裡是修正的地方 ***
        # librosa 0.10.0+ 版本已移除 librosa.feature.chroma_cqt_options
        # 並且 frames_to_time 不再需要 n_fft 參數。
        # 我們直接使用最簡單的轉換即可。
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
            # 回傳 segments 和繪圖需要的原始數據 (y, sr)
            return {'success': True, 'segments': segments, 'raw_audio': (y, sr)}
        else:
            duration = librosa.get_duration(y=y, sr=sr)
            segments.append({'label': '整首歌曲 (分段失敗)', 'start': 0.0, 'end': duration})
            return {'success': True, 'segments': segments, 'raw_audio': (y, sr)}

    except Exception as e:
        print(f"Error analyzing file: {e}")
        # *** 變動：將錯誤訊息回傳 ***
        # 這樣 results.html 才能顯示 'No attribute...' 錯誤
        return {'success': False, 'error': str(e)}

def create_plot_visualization(filename_no_ext, raw_audio, segments):
    """
    繪製波形圖並標示分段點 (此函式不變)
    """
    try:
        y, sr = raw_audio
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f'音樂結構分析圖: {filename_no_ext}.mp3')
        plt.xlabel('時間 (秒)')
        plt.ylabel('振幅')
        colors = ['r', 'g', 'b']
        for i, seg in enumerate(segments):
            plt.axvline(x=seg['start'], color=colors[i % len(colors)], linestyle='--', label=f'{seg["label"]} (Start)')
            mid_point = (seg['start'] + seg['end']) / 2
            plt.text(mid_point, 0, seg['label'], ha='center', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.axvline(x=segments[-1]['end'], color='gray', linestyle=':')
        
        # 修正圖例，避免重複標籤
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plot_filename = f"{filename_no_ext}.png"
        plot_save_path = os.path.join(STATIC_FOLDER, 'plots', plot_filename)
        plt.savefig(plot_save_path)
        plt.close()
        plot_url_path = f"plots/{plot_filename}"
        return plot_url_path
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

# --- 路由 (Routes) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'music_file' not in request.files:
            flash('錯誤：請求中沒有檔案欄位。', 'error')
            return redirect(request.url) 
        
        file = request.files['music_file']
        
        if file.filename == '':
            flash('錯誤：您尚未選擇任何檔案。', 'error')
            return redirect(request.url)
        
        if not file or not allowed_file(file.filename):
            flash(f'錯誤：不支援的檔案類型。目前只接受 .mp3 檔案 (您上傳的是: {file.filename})', 'error')
            return redirect(request.url)
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            difficulty = request.form.get('difficulty')
            
            # 核心：分析音樂
            analysis_results = analyze_music_segments(filepath)
            
            plot_url = None
            # *** 變動：檢查 analysis_results['success'] 是否為 True ***
            if analysis_results['success']:
                # 呼叫繪圖函式
                filename_no_ext = os.path.splitext(filename)[0]
                plot_url = create_plot_visualization(
                    filename_no_ext,
                    analysis_results['raw_audio'],
                    analysis_results['segments']
                )
            
            # *** 無論成功或失敗，都渲染 results.html ***
            # (如果失敗，analysis_results 會包含 'error' 訊息)
            return render_template('results.html', 
                                   analysis_results=analysis_results,
                                   filename=filename,
                                   difficulty=difficulty,
                                   plot_url=plot_url
                                  )
        
        except Exception as e:
            print(f"--- 處理 POST 請求時發生未預期錯誤: {e} ---")
            flash(f'處理檔案時發生嚴重錯誤: {e}', 'error')
            return redirect(request.url)

    # --- GET 請求 (初始載入頁面) ---
    return render_template('index.html')

# --- 執行 App ---
if __name__ == '__main__':
    app.run(debug=True)