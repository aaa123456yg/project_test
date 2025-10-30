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
# *** 新增：設定 Matplotlib 中文字型 ***
from matplotlib.font_manager import FontProperties
import platform

def get_chinese_font():
    """
    自動偵測並回傳一個可用的中文字型路徑
    """
    system = platform.system()
    
    # 優先：檢查專案資料夾內是否有字型檔 (例如 'msjheng.ttc')
    if os.path.exists('msjheng.ttc'):
        return FontProperties(fname='msjheng.ttc', size=12)
    if os.path.exists('msjhl.ttc'):
        return FontProperties(fname='msjhl.ttc', size=12)
        
    # 其次：根據作業系統猜測路徑
    try:
        if system == 'Windows':
            # Windows: 微軟正黑體
            return FontProperties(fname=r'c:\windows\fonts\msjhl.ttc', size=12)
        elif system == 'Darwin':
            # macOS: 蘋方體
            return FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=12)
        elif system == 'Linux':
            # Linux: Noto Sans (需安裝)
            return FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf', size=12)
    except IOError:
        pass # 找不到字型

    # 備案：如果都找不到，回傳預設值 (仍會亂碼，但在終端機會提示)
    print("[警告] 找不到可用的中文字型檔。圖表標題將顯示為方框。")
    print("       請下載 'msjheng.ttc' (微軟正黑體) 並放到 'project' 資料夾中。")
    return FontProperties(size=12)

# 取得可用的中文字型設定
CHINESE_FONT = get_chinese_font()


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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_nearest_beat(beat_times, target_time):
    """
    從節拍時間點中，找到最接近 target_time 的那個點
    """
    idx = (np.abs(beat_times - target_time)).argmin()
    return beat_times[idx]

def analyze_music_segments(file_path):
    """
    *** 函式重構：使用新的 1/5, 3/5 比例分割邏輯 ***
    """
    try:
        # 1. 載入
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. 計算節拍 (Beats)
        # 我們在節拍點上切割，會比在重拍點(Onsets)上切割更規律
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # 如果找不到節拍 (例如純人聲或無節奏音樂)，就只能使用絕對時間
        if len(beat_times) < 5: # 至少要有幾個節拍點才具參考價值
            print("[警告] 節拍偵測效果不佳，將使用絕對時間比例分割。")
            intro_end_time = duration * (1/5)
            main_end_time = duration * (4/5) # 1/5 + 3/5
        else:
            # 3. 計算目標時間點
            target_intro_end = duration * (1/5)
            target_main_end = duration * (4/5) # 1/5 (Intro) + 3/5 (Main)
            
            # 4. 找到最接近目標時間點的「節拍點」
            intro_end_time = find_nearest_beat(beat_times, target_intro_end)
            main_end_time = find_nearest_beat(beat_times, target_main_end)

            # 5. 確保分割點是合理的 (例如前奏至少 1 秒)
            if intro_end_time < 1.0:
                intro_end_time = target_intro_end
            if main_end_time <= intro_end_time:
                main_end_time = target_main_end
        
        # 6. 建立分段標籤
        segments = [
            {'label': '前奏 (熱身)', 'start': 0.0, 'end': intro_end_time},
            {'label': '主旋律', 'start': intro_end_time, 'end': main_end_time},
            {'label': '結尾 (緩和)', 'start': main_end_time, 'end': duration}
        ]
            
        return {'success': True, 'segments': segments, 'raw_audio': (y, sr)}

    except Exception as e:
        print(f"Error analyzing file: {e}")
        return {'success': False, 'error': str(e)}

def create_plot_visualization(filename_no_ext, raw_audio, segments):
    """
    繪製波形圖並標示分段點
    """
    try:
        y, sr = raw_audio
        
        plt.figure(figsize=(12, 4))
        
        # 1. 繪製波形圖
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        
        # *** 變動：使用中文字型 ***
        plt.title(f'音樂結構分析圖: {filename_no_ext}.mp3', fontproperties=CHINESE_FONT)
        plt.xlabel('時間 (秒)', fontproperties=CHINESE_FONT)
        plt.ylabel('振幅', fontproperties=CHINESE_FONT)
        
        # 2. 繪製分段的垂直線
        colors = ['r', 'g', 'b']
        labels_seen = set() # 用於處理圖例

        for i, seg in enumerate(segments):
            label_text = seg["label"]
            color = colors[i % len(colors)]
            
            # 處理圖例 (legend)
            legend_label = f'{label_text} (Start)'
            if legend_label not in labels_seen:
                plt.axvline(x=seg['start'], color=color, linestyle='--', label=legend_label)
                labels_seen.add(legend_label)
            else:
                 plt.axvline(x=seg['start'], color=color, linestyle='--')

            # 3. 在每個分段中間加上文字標籤
            mid_point = (seg['start'] + seg['end']) / 2
            # *** 變動：使用中文字型 ***
            plt.text(mid_point, 0, label_text, ha='center', fontsize=12, 
                     fontproperties=CHINESE_FONT,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 處理最後一條線 (整首歌的結尾)
        plt.axvline(x=segments[-1]['end'], color='gray', linestyle=':')
        
        # 4. 產生圖例 (Legend)
        # *** 變動：使用中文字型 ***
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop=CHINESE_FONT)
        
        plt.tight_layout()
        
        # 5. 儲存圖片
        plot_filename = f"{filename_no_ext}.png"
        plot_save_path = os.path.join(STATIC_FOLDER, 'plots', plot_filename)
        
        plt.savefig(plot_save_path)
        plt.close() # 釋放記憶體
        
        # 6. 回傳圖片的路徑
        plot_url_path = f"plots/{plot_filename}"
        return plot_url_path
    
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

# --- 路由 (Routes) ---
# ( app.route('/') 區塊完全不需要修改 )
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
            
            # 核心：分析音樂 (現在會使用新的分割邏輯)
            analysis_results = analyze_music_segments(filepath)
            
            plot_url = None
            if analysis_results['success']:
                # 呼叫繪圖函式 (現在會使用中文字型)
                filename_no_ext = os.path.splitext(filename)[0]
                plot_url = create_plot_visualization(
                    filename_no_ext,
                    analysis_results['raw_audio'],
                    analysis_results['segments']
                )
            
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