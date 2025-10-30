import os
import librosa
import numpy as np 
import json
import random
import requests
import shutil   
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- 影片剪輯套件 ---
import moviepy.editor as mp 

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
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'outputs') 
DATABASE_FILE = 'standing_exercise.json'
ALLOWED_EXTENSIONS = {'mp3'} # <-- 確保這一行存在

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_very_secret_key_for_flash'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'plots'), exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# --- (資料庫載入 load_exercise_database 函式 ... 保持不變) ---
def load_exercise_database(filename=DATABASE_FILE):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            database = json.load(f)
            print(f"--- 成功載入資料庫: {filename} ---")
            return database
    except FileNotFoundError:
        print(f"--- [嚴重錯誤] 找不到資料庫檔案: {filename} ---")
        return None
    except json.JSONDecodeError:
        print(f"--- [嚴重錯誤] 資料庫檔案 {filename} 格式錯誤 ---")
        return None
EXERCISE_DATABASE = load_exercise_database()

# --- ( allowed_file, analyze_music_segments, get_matched_exercises ... 保持不變) ---
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

def get_matched_exercises(segments, difficulty, database):
    difficulty_map = {'初級': 'low', '中級': 'medium', '高級': 'high'}
    segment_map = {'前奏 (熱身)': 'warmup', '主旋律': 'core', '結尾 (緩和)': 'cooldown'}
    difficulty_key = difficulty_map.get(difficulty, 'low')
    matched_playlist = []
    if not database:
        return {'success': False, 'error': '資料庫未成功載入'}
    for seg in segments:
        segment_label = seg['label']
        segment_duration = seg['end'] - seg['start']
        category_key = segment_map.get(segment_label)
        if not category_key: continue
        try:
            available_actions = database[category_key][difficulty_key]
            if available_actions:
                selected_action = random.choice(available_actions)
                matched_playlist.append({
                    'segment_label': segment_label,
                    'duration': segment_duration,
                    'exercise': selected_action
                })
        except KeyError:
            return {'success': False, 'error': f'資料庫結構錯誤，找不到 {category_key}/{difficulty_key}'}
    return {'success': True, 'playlist': matched_playlist}

# --- ( create_plot_visualization 函式 ... 保持不變) ---
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

# --- *** 影片合成函式 (已修正) *** ---
def create_workout_video(music_filepath, playlist, output_filename):
    """
    合成影片的核心函式 (已修正 WinError 32)
    """
    video_clips = []
    temp_files = []
    temp_gif_clips = [] # <-- 新增：用來追蹤原始的 GIF clip 物件
    
    # <-- 新增：在 try 區塊外宣告變數，以便 finally 區塊能存取
    audio_clip = None
    final_video_no_audio = None
    final_video = None

    try:
        print("--- 開始合成影片 ---")
        
        # 1. 載入音訊
        audio_clip = mp.AudioFileClip(music_filepath)
        
        # 2. 遍歷播放清單，下載並建立影片片段
        for i, item in enumerate(playlist):
            gif_url = item['exercise']['gif_url']
            duration = item['duration']
            
            print(f"處理中: {item['segment_label']} ( {duration:.2f}s ) ...")
            
            # --- 下載 GIF ---
            temp_gif_path = f"temp_gif_{i}.gif"
            try:
                response = requests.get(gif_url, stream=True)
                response.raise_for_status()
                with open(temp_gif_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                temp_files.append(temp_gif_path)
            except Exception as e:
                print(f"警告：下載 GIF 失敗 {gif_url}。錯誤: {e}。將跳過此片段。")
                continue # 跳過這個 GIF

            # --- 載入並循環 GIF ---
            gif = mp.VideoFileClip(temp_gif_path)
            temp_gif_clips.append(gif) # <-- 新增：追蹤此 clip
            
            gif_resized = gif.resize(width=640) 
            looped_gif = gif_resized.fx(mp.vfx.loop, duration=duration)
            
            video_clips.append(looped_gif)

        if not video_clips:
            print("錯誤：沒有任何影片片段可合成。")
            return None

        # 3. 拼接所有影片片段
        print("拼接影片片段...")
        final_video_no_audio = mp.concatenate_videoclips(video_clips)
        
        # 4. 加上完整的音訊
        print("加入音訊...")
        final_video = final_video_no_audio.set_audio(audio_clip)
        
        # 5. 寫入 MP4 檔案
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
        print(f"開始渲染 MP4: {output_filepath} ... (這會花很長間)")
        
        final_video.write_videofile(
            output_filepath, 
            codec='libx264', 
            audio_codec='aac', 
            threads=4,
            logger='bar'
        )
        
        print("--- 影片合成完畢 ---")
        return output_filename
        
    except Exception as e:
        print(f"[嚴重錯誤] 影片合成失敗: {e}")
        return None
    finally:
        # --- (*** 這是關鍵的修正區塊 ***) ---
        print("--- 開始清理暫存檔案 ---")
        
        # 1. 關閉所有 moviepy 物件 (釋放檔案鎖定)
        if final_video:
            final_video.close()
        if final_video_no_audio:
            final_video_no_audio.close()
        if audio_clip:
            audio_clip.close()
            
        for clip in video_clips:
            clip.close()
        for clip in temp_gif_clips: # 關閉原始的 GIF clip
            clip.close()
                
        # 2. 清理暫存的 GIF 檔案
        print("--- 正在刪除 .gif 暫存檔 ---")
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"已刪除: {f}")
                except Exception as e:
                    # 即使關閉了，Windows 有時仍需一點時間，故加入例外處理
                    print(f"刪除 {f} 失敗: {e} (檔案可能仍被鎖定，但可手動刪除)")
            else:
                print(f"找不到檔案 {f} (可能未成功下載)")

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
        
        # (修正：這裡的檢查邏輯反了)
        if not file or not allowed_file(file.filename):
            flash(f'錯誤：不支援的檔案類型。目前只接受 .mp3 檔案', 'error')
            return redirect(request.url)
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            difficulty = request.form.get('difficulty')
            
            analysis_results = analyze_music_segments(filepath)
            
            plot_url = None
            matched_playlist_results = None 
            
            if analysis_results['success']:
                filename_no_ext = os.path.splitext(filename)[0]
                plot_url = create_plot_visualization(
                    filename_no_ext,
                    analysis_results['raw_audio'],
                    analysis_results['segments']
                )
                matched_playlist_results = get_matched_exercises(
                    analysis_results['segments'],
                    difficulty,
                    EXERCISE_DATABASE
                )

            return render_template('results.html', 
                                   analysis_results=analysis_results,
                                   filename=filename,
                                   difficulty=difficulty,
                                   plot_url=plot_url,
                                   matched_playlist_results=matched_playlist_results 
                                  )
        
        except Exception as e:
            print(f"--- 處理 POST 請求時發生未預期錯誤: {e} ---")
            flash(f'處理檔案時發生嚴重錯誤: {e}', 'error')
            return redirect(request.url)

    # --- GET 請求 (初始載入頁面) ---
    return render_template('index.html')


# --- (影片合成路由 /generate_video ... 保持不變) ---
@app.route('/generate_video')
def generate_video():
    music_filename = request.args.get('filename')
    difficulty = request.args.get('difficulty')
    
    if not music_filename or not difficulty:
        flash('錯誤：缺少音樂檔案名稱或難度。', 'error')
        return redirect(url_for('index'))
        
    music_filepath = os.path.join(app.config['UPLOAD_FOLDER'], music_filename)
    
    if not os.path.exists(music_filepath):
        flash('錯誤：找不到音樂檔案，請重新上傳。', 'error')
        return redirect(url_for('index'))

    analysis_results = analyze_music_segments(music_filepath)
    if not analysis_results['success']:
        flash(f'錯誤：重新分析音樂失敗: {analysis_results.error}', 'error')
        return redirect(url_for('index'))
        
    matched_playlist_results = get_matched_exercises(
        analysis_results['segments'],
        difficulty,
        EXERCISE_DATABASE
    )
    if not matched_playlist_results['success']:
        flash(f'錯誤：重新配對動作失敗: {matched_playlist_results.error}', 'error')
        return redirect(url_for('index'))
        
    music_filename_no_ext = os.path.splitext(music_filename)[0]
    output_filename = f"workout_{music_filename_no_ext}.mp4"
    
    video_file = create_workout_video(
        music_filepath,
        matched_playlist_results['playlist'],
        output_filename
    )
    
    if video_file:
        return redirect(url_for('play_video', videofile=video_file))
    else:
        flash('錯誤：影片合成失敗，請檢查伺服器日誌。', 'error')
        return redirect(url_for('index'))


# --- (影片播放路由 /play ... 保持不變) ---
@app.route('/play')
def play_video():
    video_filename = request.args.get('videofile')
    if not video_filename:
        flash('錯誤：沒有指定影片檔案。', 'error')
        return redirect(url_for('index'))
        
    video_url = url_for('static', filename=f'outputs/{video_filename}')
    
    return render_template('play.html', video_url=video_url, video_filename=video_filename)


# --- 執行 App ---
if __name__ == '__main__':
    app.run(debug=True)