"""Stage 0: what channels does the automatic ATD carry?

Prints ONLY counts and yes/no answers -- never a data value -- so the output is safe to share.
Usage: python tools/probe_format.py <data_dir>
"""
import collections
import os
import sys
import xml.etree.ElementTree as ET

data_dir = sys.argv[1]
KEYWORDS = ("own", "poss", "state", "status", "alive", "live")

for source in ("tactical", "broadcast"):
    stem = "BAR-ATH_" + source
    meta = os.path.join(data_dir, source, stem + "_atd.xml")
    raw = os.path.join(data_dir, source, stem + "_atd.txt")
    print("[{}]".format(source))

    root = ET.parse(meta).getroot()
    tags = collections.Counter(el.tag.split("}")[-1] for el in root.iter())
    channel_like = sum(n for t, n in tags.items() if "channel" in t.lower())
    names = [(el.get("id") or "") + " " + (el.findtext("Name") or "")
             for el in root.iter() if "channel" in el.tag.lower()]
    hit = any(k in n.lower() for n in names for k in KEYWORDS)
    print("  metadata: {} channel-like elements; any named like ball state/ownership: {}".format(
        channel_like, hit))

    parts_count = collections.Counter()
    player_fields = collections.Counter()
    ball_fields = collections.Counter()
    with open(raw) as handle:
        for i, line in enumerate(handle):
            parts = line.rstrip("\n").split(":")
            parts_count[len(parts)] += 1
            if len(parts) >= 3:
                player_fields[len(parts[1].split(";")[0].split(","))] += 1
                ball_fields[len(parts[2].split(","))] += 1
            if i >= 20000:
                break
    print("  ':'-separated parts per line   {}".format(dict(parts_count)))
    print("  fields per player group        {}".format(dict(player_fields)))
    print("  fields per ball group          {}".format(dict(ball_fields)))
