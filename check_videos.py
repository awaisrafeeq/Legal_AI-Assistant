"""
Check Video and Audio Files - What happened to them?
"""
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse, unquote
from collections import defaultdict

load_dotenv()

VIDEO_EXTS = ['mp4', 'avi', 'mov', 'wmv', 'mkv', 'flv', 'webm', 'm4v']
AUDIO_EXTS = ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg', 'wma']

def split_sas_url(sas_url):
    p = urlparse(sas_url)
    return f"{p.scheme}://{p.netloc}", unquote(p.path.strip("/").split("/")[0]), p.query

def main():
    sas_url = os.environ.get("CONTAINER_SAS_URL")
    account_url, container_name, sas_token = split_sas_url(sas_url)
    
    client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container = client.get_container_client(container_name)
    
    print("="*70)
    print("VIDEO & AUDIO FILES ANALYSIS")
    print("="*70)
    
    # Find all video/audio files
    video_files = []
    audio_files = []
    processed_video = []
    processed_audio = []
    
    print("\nScanning all blobs...")
    
    for blob in container.list_blobs():
        name = blob.name.lower()
        
        if name.startswith('extracted-text/'):
            # Check if it's a video/audio processed file
            orig = name.replace('extracted-text/', '').replace('.txt', '').lower()
            ext = orig.rsplit('.', 1)[-1] if '.' in orig else ''
            if ext in VIDEO_EXTS:
                processed_video.append((name, blob.size))
            elif ext in AUDIO_EXTS:
                processed_audio.append((name, blob.size))
        elif not name.startswith('.'):
            ext = name.rsplit('.', 1)[-1] if '.' in name else ''
            if ext in VIDEO_EXTS:
                video_files.append(name)
            elif ext in AUDIO_EXTS:
                audio_files.append(name)
    
    # Report
    print("\n" + "="*70)
    print("VIDEO FILES")
    print("="*70)
    print(f"Total video files found:     {len(video_files)}")
    print(f"Video files with OCR output: {len(processed_video)}")
    
    if video_files:
        print(f"\nOriginal video files in container:")
        for v in video_files[:5]:
            print(f"  📹 {v}")
        if len(video_files) > 5:
            print(f"  ... and {len(video_files)-5} more")
    
    if processed_video:
        print(f"\nProcessed video outputs (with .txt extension):")
        for v, size in processed_video[:5]:
            print(f"  📝 {v.replace('extracted-text/', '')} ({size} bytes)")
    
    # Show actual content of video OCR
    if processed_video:
        print(f"\n{'='*70}")
        print("VIDEO OCR CONTENT SAMPLE")
        print(f"{'='*70}")
        sample = processed_video[0][0]
        print(f"\nFile: {sample}")
        try:
            content = container.get_blob_client(sample).download_blob().readall()
            text = content.decode('utf-8', errors='ignore')
            print("-" * 60)
            print(text)
            print("-" * 60)
        except Exception as e:
            print(f"Error reading: {e}")
    
    # AUDIO
    print(f"\n{'='*70}")
    print("AUDIO FILES")
    print(f"{'='*70}")
    print(f"Total audio files found:     {len(audio_files)}")
    print(f"Audio files with OCR output: {len(processed_audio)}")
    
    if audio_files:
        print(f"\nOriginal audio files in container:")
        for a in audio_files[:5]:
            print(f"  🎵 {a}")
        if len(audio_files) > 5:
            print(f"  ... and {len(audio_files)-5} more")
    
    # Check content for audio
    if processed_audio:
        print(f"\nAudio OCR content sample:")
        sample = processed_audio[0][0]
        try:
            content = container.get_blob_client(sample).download_blob().readall()
            text = content.decode('utf-8', errors='ignore')
            print("-" * 60)
            print(text[:500])
            print("-" * 60)
        except Exception as e:
            print(f"Error reading: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if not video_files and not audio_files:
        print("No video or audio files found in the container!")
    elif len(processed_video) == 0 and len(processed_audio) == 0:
        print(f"⚠️  WARNING: Found {len(video_files)} video files but NONE were processed!")
        print(f"⚠️  WARNING: Found {len(audio_files)} audio files but NONE were processed!")
        print("\nReason: Audio/Video files need Azure Speech Services for transcription.")
        print("Current script only saves metadata, not actual transcription.")
    else:
        print(f"✅ Video files processed: {len(processed_video)}/{len(video_files)}")
        print(f"✅ Audio files processed: {len(processed_audio)}/{len(audio_files)}")

if __name__ == "__main__":
    main()
