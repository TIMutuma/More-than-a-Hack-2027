"""Download the IDSSE open dataset (DFL, 7 Bundesliga matches, CC-BY 4.0) into external/idsse/.

Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event data in
elite soccer." Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

Only match information + raw observed positions are needed (~2.6 GB). Downloads resume where
they stopped and retry, because the connection this was built on drops often.

    python tools/fetch_idsse.py
"""
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(HERE), "external", "idsse")
URL = "https://ndownloader.figshare.com/files/{}"

# match_id -> ((matchinformation id, bytes), (raw observed positions id, bytes))
FILES = {
    "J03WPY": ((51643487, 11922), (51643526, 371688499)),
    "J03WN1": ((51643472, 11722), (51643517, 391722803)),
    "J03WMX": ((51643475, 11930), (51643514, 418524833)),
    "J03WOH": ((51643478, 11745), (51643520, 348945715)),
    "J03WQQ": ((51643484, 11932), (51643529, 356219423)),
    "J03WOY": ((51643481, 11952), (51643523, 362567859)),
    "J03WR9": ((51643490, 11823), (51643532, 372934140)),
}


def fetch(file_id, path, expected, attempts=50):
    for attempt in range(attempts):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == expected:
            return
        if have > expected:
            have = 0  # corrupt leftover: start over
        request = urllib.request.Request(URL.format(file_id), headers={
            "User-Agent": "mtah-possession research download",
            **({"Range": "bytes={}-".format(have)} if have else {})})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:
                    have = 0  # server ignored the range: start over
                with open(path, "ab" if have else "wb") as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
        except Exception as error:  # network drop: resume
            print("  {} retry {} ({})".format(os.path.basename(path), attempt + 1,
                                              type(error).__name__), flush=True)
            time.sleep(min(5 * (attempt + 1), 60))
    raise SystemExit("gave up on {}".format(path))


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    wanted = sys.argv[1:] or sorted(FILES)
    for match in wanted:
        (meta_id, meta_size), (pos_id, pos_size) = FILES[match]
        fetch(meta_id, os.path.join(DEST, match + "_meta.xml"), meta_size)
        fetch(pos_id, os.path.join(DEST, match + "_positions.xml"), pos_size)
        print("{} done".format(match), flush=True)
