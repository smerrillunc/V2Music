import subprocess
import time
import random
import argparse
import os

# Default proxy list (empty by default so the script runs without proxies).
proxies = []  # Example: ["socks5://localhost:1080", "socks5://localhost:1081"]

def download_video(url, proxy):
    """
    Download a YouTube video using yt-dlp via a command line call.
    
    This command-line version downloads the best video and audio streams,
    merges them into an MP4 file, and forces re-encoding of the audio to AAC
    if needed.
    """
    # Base command options. Options correspond to:
    #   --sleep-interval 5
    #   --max-sleep-interval 10
    #   --limit-rate 1M
    #   --user-agent "Mozilla/5.0 ..."
    #   --referer "https://www.youtube.com/"
    #   --cookiefile cookies.txt
    #   --outtmpl "data/%(id)s.mp4"
    #   --format "bestvideo+bestaudio/best"
    #   --merge-output-format mp4
    cmd = [
        "yt-dlp",
        "--limit-rate", "1M",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "--referer", "https://www.youtube.com/",
        "--cookies", "cookies.txt",
        "--download-sections", "*0-30",
        "-f", "bv[ext=mp4],ba",
        "-o", "test_data/%(width&video|audio)s/%(id)s.%(ext)s"
    ]
    # If a proxy is provided, include it.
    if proxy:
        cmd.extend(["--proxy", proxy])
    
    # Append the URL to download.
    cmd.append(url)

    print(f"Downloading: {url} using proxy: {proxy if proxy else 'no proxy'}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error downloading {url} using proxy {proxy} (return code: {result.returncode}).")

def main():
    """
    Main function that:
      1. Parses command-line arguments (--start and --end) for slicing video IDs.
      2. Reads YouTube video IDs from 'V2M-20k.txt' (one ID per line).
      3. Constructs the full YouTube URL for each video.
      4. Uses rotated proxy values if provided.
      5. Uses a single command-line call (via subprocess) for each download.
      6. Waits a random extra delay between downloads.
    """
    parser = argparse.ArgumentParser(
        description="Download YouTube videos into MP4 using the yt-dlp command-line interface via Python."
    )
    parser.add_argument("--start", type=int, default=0, help="Start index for video IDs (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index for video IDs (exclusive)")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs("data", exist_ok=True)

    try:
        with open("V2M-20k.txt", "r") as file:
            video_ids = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print("Error: 'V2M-20k.txt' not found. Please ensure the file exists in the same directory as this script.")
        return

    # Slice the list of video IDs if indices are specified
    video_ids = video_ids[args.start:args.end]
    print(f"Processing {len(video_ids)} video(s) from index {args.start} to {args.end if args.end is not None else 'end'}.")

    proxy_count = len(proxies)
    for idx, vid in enumerate(video_ids):
        video_url = f"https://www.youtube.com/watch?v={vid}"
        print(video_url)
        # Rotate proxies if any; otherwise, no proxy.
        proxy = proxies[idx % proxy_count] if proxy_count > 0 else ""
        download_video(video_url, proxy)
        
        # Wait a random delay between downloads.
        extra_delay = random.uniform(5, 10)
        print(f"Waiting for an extra {extra_delay:.1f} seconds before downloading Video {idx +1}...\n")
        time.sleep(extra_delay)

if __name__ == "__main__":
    main()