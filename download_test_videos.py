import subprocess
import sys
from pathlib import Path

VIDEOS = [
    "https://www.youtube.com/watch?v=FsGPxhidwGg",
    "https://www.youtube.com/watch?v=dP1UQ59d0Vw",
]
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "test_videos"


def ensure_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        print("yt-dlp غير مثبت. سيتم تثبيته الآن...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=False)
        return result.returncode == 0


def download_videos():
    if not ensure_yt_dlp():
        print("فشل تثبيت yt-dlp. الرجاء تثبيته يدويًا ثم أعد التشغيل.")
        return

    from yt_dlp import YoutubeDL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(OUTPUT_DIR / "%(title)s.%(ext)s"),
        "format": "mp4[ext=mp4]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        for url in VIDEOS:
            print(f"Downloading: {url}")
            try:
                ydl.download([url])
            except Exception as exc:
                print(f"فشل تنزيل الفيديو {url}: {exc}")

    print(f"التحميل اكتمل. الفيديوهات موجودة في: {OUTPUT_DIR}")


if __name__ == "__main__":
    download_videos()
