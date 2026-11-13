"""Download Metrica Sports' public sample data (3 games) into external/metrica/.

https://github.com/metrica-sports/sample-data -- anonymised tracking (25 fps) and event data from
the same provider as the challenge, published openly for research and education. These are
OTHER games, so their labels may be used for training and selection (RULES.md).

Resumes and retries: built on a connection that drops and crawls.
    python tools/fetch_metrica_sample.py
"""
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(HERE), "external", "metrica")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"

# path -> size in bytes (from the GitHub tree API), so completeness is checkable
FILES = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv": 139446,
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": 32782073,
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": 32782961,
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv": 153897,
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": 29627711,
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": 31890338,
    "Sample_Game_3/Sample_Game_3_events.json": 1709486,
    "Sample_Game_3/Sample_Game_3_metadata.xml": 81806,
    "Sample_Game_3/Sample_Game_3_tracking.txt": 53443115,
}


def fetch(rel, expected, attempts=200):
    path = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(attempts):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == expected:
            return
        if have > expected:
            have = 0
        headers = {"User-Agent": "mtah-possession research download"}
        if have:
            headers["Range"] = "bytes={}-".format(have)
        try:
            request = urllib.request.Request(BASE + rel, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0
                with open(path, "ab" if have else "wb") as out:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
                        out.flush()
        except Exception as error:
            print("  {} retry {} ({})".format(os.path.basename(rel), attempt + 1,
                                              type(error).__name__), flush=True)
            time.sleep(min(5 * (attempt + 1), 30))
    raise SystemExit("gave up on " + rel)


if __name__ == "__main__":
    # small files first, then the tracking, smallest game first
    for rel, size in sorted(FILES.items(), key=lambda kv: kv[1]):
        fetch(rel, size)
        print("done  {}".format(rel), flush=True)
