import os
import librosa
import numpy as np 
# *** 新增：導入 json (讀取資料庫) 和 random (隨機挑選) ***
import json
import random
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Matplotlib 設定 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.font_manager import FontProperties
import platform

# --- Flask 設定 ---
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'mp3'}
DATABASE_FILE = 'standing_exercise.json' # <-- 指定資料庫檔案

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_very_secret_key_for_flash'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'plots'), exist_ok=True)

# --- (中文字型 get_chinese_font 函式 ... 保持不變) ---
def get_chinese_font():
    system = platform.system()
    if os.path.exists('msjheng.ttc'):
        return FontProperties(fname='msjheng.ttc', size=12)
    if os.path.exists('msjhl.ttc'):
        return FontProperties(fname='msjhl.ttc', size=12)
    try:
        if system == 'Windows':
            return FontProperties(fname=r'c:\windows\fonts\msjhl.ttc', size=12)
        elif system == 'Darwin':
            return FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=12)
        elif system == 'Linux':
            return FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf', size=12)
    except IOError:
        pass
    print("[警告] 找不到可用的中文字型檔。")
    return FontProperties(size=12)

CHINESE_FONT = get_chinese_font()

# --- *** 新增：讀取 JSON 資料庫 *** ---
def load_exercise_database(filename=DATABASE_FILE):
    """在程式啟動時載入 JSON 資料庫"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            database = json.load(f)
            print(f"--- 成功載入資料庫: {filename} ---")
            return database
    except FileNotFoundError:
        print(f"--- [嚴重錯誤] 找不到資料庫檔案: {filename} ---")
        print("--- 請確認 'standing_exercise.json' 檔案在 'app.py' 旁邊 ---")
        return None
    except json.JSONDecodeError:
        print(f"--- [嚴重錯誤] 資料庫檔案 {filename} 格式錯誤 ---")
        return None

# 在 Flask 啟動時，就先把資料庫載入記憶體
EXERCISE_DATABASE = load_exercise_database()

# --- *** 新增：配對邏輯函式 *** ---
def get_matched_exercises(segments, difficulty, database):
    """根據音樂分段和難度，從資料庫配對 GIF"""
    
    # 1. 建立「翻譯字典」
    difficulty_map = {
        '初級': 'low',
        '中級': 'medium',
        '高級': 'high'
    }
    # 我們的分段標籤和 JSON key 的對應
    segment_map = {
        '前奏 (熱身)': 'warmup',
        '主旋律': 'core',
        '結尾 (緩和)': 'cooldown'
    }

    # 取得使用者選擇的難度 (e.g., 'low')
    difficulty_key = difficulty_map.get(difficulty)
    if not difficulty_key:
        print(f"警告：未知的難度 '{difficulty}'，將使用 'low' 作為預設值")
        difficulty_key = 'low'
        
    matched_playlist = []

    # 2. 檢查資料庫是否成功載入
    if not database:
        return {'success': False, 'error': '資料庫未成功載入，請檢查伺服器日誌。'}

    # 3. 遍歷音樂的
    for seg in segments:
        segment_label = seg['label']
        segment_duration = seg['end'] - seg['start']
        
        # 取得對應的 JSON key (e.g., 'warmup')
        category_key = segment_map.get(segment_label)
        
        if not category_key:
            print(f"警告：找不到 {segment_label} 的配對")
            continue
            
        try:
            # 4. 撈取該分類可用的所有動作 (e.g., database['warmup']['low'])
            available_actions = database[category_key][difficulty_key]
            
            # 5. 隨機挑選一個動作作為代表
            # (未來：我們會根據 duration 挑選多個動作)
            if available_actions:
                selected_action = random.choice(available_actions)
                
                matched_playlist.append({
                    'segment_label': segment_label,
                    'duration': segment_duration,
                    'exercise': selected_action # 包含 name, gif_url, audio_text
                })
            else:
                print(f"警告：在 {category_key} / {difficulty_key} 中找不到任何動作")
                
        except KeyError:
            print(f"錯誤：在 JSON 中找不到路徑 {category_key} -> {difficulty_key}")
            return {'success': False, 'error': f'資料庫結構錯誤，找不到 {category_key}/{difficulty_key}'}
            
    return {'success': True, 'playlist': matched_playlist}


# --- ( allowed_file, analyze_music_segments, create_plot_visualization 函式保持不變 ) ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_nearest_beat(beat_times, target_time):
    idx = (np.abs(beat_times - target_time)).argmin()
    return beat_times[idx]

def analyze_music_segments(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 5: 
            intro_end_time = duration * (1/5)
            main_end_time = duration * (4/5)
        else:
            target_intro_end = duration * (1/5)
            target_main_end = duration * (4/5)
            intro_end_time = find_nearest_beat(beat_times, target_intro_end)
            main_end_time = find_nearest_beat(beat_times, target_main_end)
            if intro_end_time < 1.0: intro_end_time = target_intro_end
            if main_end_time <= intro_end_time: main_end_time = target_main_end
        
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
    try:
        y, sr = raw_audio
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f'音樂結構分析圖: {filename_no_ext}.mp3', fontproperties=CHINESE_FONT)
        plt.xlabel('時間 (秒)', fontproperties=CHINESE_FONT)
        plt.ylabel('振幅', fontproperties=CHINESE_FONT)
        colors = ['r', 'g', 'b']
        labels_seen = set()
        for i, seg in enumerate(segments):
            label_text = seg["label"]
            color = colors[i % len(colors)]
            legend_label = f'{label_text} (Start)'
            if legend_label not in labels_seen:
                plt.axvline(x=seg['start'], color=color, linestyle='--', label=legend_label)
                labels_seen.add(legend_label)
            else:
                 plt.axvline(x=seg['start'], color=color, linestyle='--')
            mid_point = (seg['start'] + seg['end']) / 2
            plt.text(mid_point, 0, label_text, ha='center', fontsize=12, 
                     fontproperties=CHINESE_FONT,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.axvline(x=segments[-1]['end'], color='gray', linestyle=':')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop=CHINESE_FONT)
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
        # ... (檔案上傳檢查 ... 保持不變) ...
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
            
            # 1. 分析音樂
            analysis_results = analyze_music_segments(filepath)
            
            plot_url = None
            matched_playlist_results = None # <-- 新增
            
            if analysis_results['success']:
                # 2. 繪製圖表
                filename_no_ext = os.path.splitext(filename)[0]
                plot_url = create_plot_visualization(
                    filename_no_ext,
                    analysis_results['raw_audio'],
                    analysis_results['segments']
                )
                
                # 3. *** 新增：呼叫配對函式 ***
                matched_playlist_results = get_matched_exercises(
                    analysis_results['segments'],
                    difficulty,
                    EXERCISE_DATABASE # 使用我們預先載入的資料庫
                )

            return render_template('results.html', 
                                   analysis_results=analysis_results,
                                   filename=filename,
                                   difficulty=difficulty,
                                   plot_url=plot_url,
                                   matched_playlist_results=matched_playlist_results # <-- 傳給 HTML
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