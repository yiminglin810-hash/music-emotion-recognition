"""
批量特征提取脚本

功能：
1. 遍历所有DEAM音频文件
2. 提取133维特征
3. 保存到CSV文件
4. 显示进度条

作者: Week 2 Day 3
日期: 2025-11-08
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.features.traditional import extract_all_features


def extract_features_for_dataset(
    audio_dir,
    annotation_file_1,
    annotation_file_2,
    output_file,
    max_songs=None
):
    """
    批量提取特征
    
    Parameters:
    -----------
    audio_dir : Path
        音频文件目录
    annotation_file_1 : Path
        标注文件1 (1-2000)
    annotation_file_2 : Path
        标注文件2 (2000-2058)
    output_file : Path
        输出CSV文件路径
    max_songs : int, optional
        最大处理歌曲数（用于测试）
    """
    
    print("=" * 70)
    print("🎵 DEAM Dataset - Batch Feature Extraction")
    print("=" * 70)
    
    # 1. 加载标注数据
    print("\n📋 Step 1: Loading annotations...")
    df1 = pd.read_csv(annotation_file_1)
    df2 = pd.read_csv(annotation_file_2)
    df_annotations = pd.concat([df1, df2], ignore_index=True)
    df_annotations.columns = df_annotations.columns.str.strip()
    df_annotations = df_annotations.set_index('song_id')
    
    print(f"   ✅ Loaded {len(df_annotations)} song annotations")
    
    # 2. 获取所有音频文件
    print("\n🎼 Step 2: Scanning audio files...")
    audio_files = list(audio_dir.glob("*.mp3"))
    print(f"   ✅ Found {len(audio_files)} audio files")
    
    # 3. 筛选有标注的音频文件
    valid_songs = []
    for audio_file in audio_files:
        song_id = int(audio_file.stem)
        if song_id in df_annotations.index:
            valid_songs.append({
                'song_id': song_id,
                'audio_path': audio_file,
                'valence': df_annotations.loc[song_id, 'valence_mean'],
                'arousal': df_annotations.loc[song_id, 'arousal_mean']
            })
    
    print(f"   ✅ {len(valid_songs)} songs have both audio and annotations")
    
    # 限制处理数量（用于测试）
    if max_songs:
        valid_songs = valid_songs[:max_songs]
        print(f"   ⚠️  Limited to {max_songs} songs for testing")
    
    # 4. 批量提取特征
    print(f"\n🎵 Step 3: Extracting features from {len(valid_songs)} songs...")
    print("   This may take a while... ☕\n")
    
    results = []
    failed_songs = []
    
    for song_info in tqdm(valid_songs, desc="Extracting", unit="song"):
        try:
            # 提取特征
            features = extract_all_features(str(song_info['audio_path']))
            
            # 添加元数据
            features['song_id'] = song_info['song_id']
            features['valence'] = song_info['valence']
            features['arousal'] = song_info['arousal']
            
            results.append(features)
            
        except Exception as e:
            failed_songs.append({
                'song_id': song_info['song_id'],
                'error': str(e)
            })
            print(f"\n   ❌ Failed: Song {song_info['song_id']} - {e}")
    
    # 5. 转换为DataFrame
    print(f"\n💾 Step 4: Saving results...")
    df_features = pd.DataFrame(results)
    
    # 重新排列列顺序：song_id, valence, arousal放在前面
    cols = ['song_id', 'valence', 'arousal'] + [col for col in df_features.columns if col not in ['song_id', 'valence', 'arousal']]
    df_features = df_features[cols]
    
    # 保存到CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved to: {output_file}")
    print(f"   📊 Shape: {df_features.shape}")
    
    # 6. 输出统计信息
    print("\n" + "=" * 70)
    print("📊 Extraction Summary")
    print("=" * 70)
    print(f"Total songs processed: {len(valid_songs)}")
    print(f"Successfully extracted: {len(results)}")
    print(f"Failed: {len(failed_songs)}")
    print(f"Success rate: {len(results)/len(valid_songs)*100:.1f}%")
    
    print(f"\nFeatures extracted:")
    print(f"  • Total features: {len(df_features.columns) - 3} dimensions")
    print(f"  • MFCC features: 60 (mean, std, delta)")
    print(f"  • Chroma features: 48 (STFT, CQT)")
    print(f"  • Spectral features: 20")
    print(f"  • Rhythm features: 5")
    
    print(f"\nOutput file:")
    print(f"  📁 {output_file}")
    print(f"  📏 Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    if failed_songs:
        print(f"\n⚠️  Failed songs ({len(failed_songs)}):")
        for fail in failed_songs[:10]:  # 只显示前10个
            print(f"   - Song {fail['song_id']}: {fail['error']}")
    
    print("\n" + "=" * 70)
    print("🎉 Feature extraction complete!")
    print("=" * 70)
    
    return df_features, failed_songs


def main():
    """主函数"""
    
    # 设置路径
    BASE_DIR = Path(__file__).parent.parent
    AUDIO_DIR = BASE_DIR / "data" / "DEAM" / "DEAM_audio" / "MEMD_audio"
    ANNOTATION_DIR = BASE_DIR / "data" / "DEAM" / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level"
    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    
    ANNOTATION_FILE_1 = ANNOTATION_DIR / "static_annotations_averaged_songs_1_2000.csv"
    ANNOTATION_FILE_2 = ANNOTATION_DIR / "static_annotations_averaged_songs_2000_2058.csv"
    OUTPUT_FILE = OUTPUT_DIR / "deam_features_all.csv"
    
    # 验证路径
    if not AUDIO_DIR.exists():
        print(f"❌ Error: Audio directory not found: {AUDIO_DIR}")
        return
    
    if not ANNOTATION_FILE_1.exists() or not ANNOTATION_FILE_2.exists():
        print(f"❌ Error: Annotation files not found")
        return
    
    # 提取特征
    # max_songs=10 用于测试，删除此参数可处理所有歌曲
    df_features, failed_songs = extract_features_for_dataset(
        audio_dir=AUDIO_DIR,
        annotation_file_1=ANNOTATION_FILE_1,
        annotation_file_2=ANNOTATION_FILE_2,
        output_file=OUTPUT_FILE,
        max_songs=None  # None = 处理所有歌曲，设置数字可用于测试
    )
    
    print(f"\n✅ All done! Features saved to:")
    print(f"   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

