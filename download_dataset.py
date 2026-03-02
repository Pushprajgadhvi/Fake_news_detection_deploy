import re
import os
import sys
import zipfile
import requests

WELFAKE_URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1"
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DEST_PATH   = os.path.join(BASE_DIR, "WELFake_Dataset.csv")

def download_welfake():
    if os.path.exists(DEST_PATH):
        size = os.path.getsize(DEST_PATH)
        if size > 1_000_000:          # > 1 MB → file looks valid
            print(f"WELFake already present ({size//1_048_576} MB). Skipping download.")
            return True

    print("Downloading WELFake dataset (~245 MB) from Zenodo …")
    print("This may take a few minutes depending on your connection speed.")

    try:
        with requests.get(WELFAKE_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(DEST_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1_048_576):   # 1 MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:.1f}%  ({downloaded//1_048_576} MB / {total//1_048_576} MB)", end="", flush=True)
            print()
        print(f"Download complete → {DEST_PATH}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(DEST_PATH):
            os.remove(DEST_PATH)
        return False


if __name__ == "__main__":
    ok = download_welfake()
    if ok:
        print("\nDataset ready. You can now run:  python app.py")
        print("(The server will auto-detect WELFake_Dataset.csv and use it for training.)")
    else:
        print("\nFailed. The app will fall back to True.csv + Fake.csv.")
        sys.exit(1)
