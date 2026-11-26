"""Build the offline report, evidence guide and recording gallery from their Markdown."""
import argparse
import base64
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

import markdown

DOC = Path(__file__).resolve().parents[1] / "docs"
PAGES = ("research_report", "video_gallery", "research_evidence")
STYLE = """
:root{color-scheme:light;--ink:#162e39;--sea:#176962;--line:#cddad7}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding:1.2rem}
body{margin:0;background:#fafbf8;color:var(--ink);font:17px/1.65 system-ui,sans-serif}
a{color:#075a60;text-underline-offset:.17em}a:focus-visible,summary:focus-visible{outline:3px solid #ba5618}
header{background:#183a43;color:white;padding:.7rem max(4vw,1rem);display:flex;gap:2rem;align-items:center;flex-wrap:wrap}
header a{color:white}header strong{letter-spacing:.08em}aside{font-size:14px}aside summary{font-weight:650;cursor:pointer}
.frame{max-width:1530px;margin:auto;display:grid;grid-template-columns:minmax(180px,235px) minmax(0,1000px);gap:clamp(2rem,4vw,5rem);padding:2.5rem 4vw 5rem}
aside>details{position:sticky;top:1rem;max-height:92vh;overflow:auto}aside ul{list-style:none;padding-left:0}
aside li{margin:.42rem 0}aside ul ul{padding-left:.8rem;border-left:1px solid var(--line)}
aside a{text-decoration:none}main{min-width:0}main>p,main>ul,main>ol{max-width:88ch}
h1,h2,h3{line-height:1.2;text-wrap:balance}h1,h2{font-family:Georgia,serif}h1{font-size:clamp(2.3rem,4.2vw,4.1rem);letter-spacing:-.045em;margin:0 0 1rem}
h2{font-size:clamp(1.7rem,2.6vw,2.4rem);margin:3rem 0 1rem;border-top:2px solid var(--sea);padding-top:.85rem}
h3{margin:2rem 0 .7rem;font-size:1.32rem}img,video{max-width:100%;height:auto}
figure{margin:1.5rem 0;background:white;border:1px solid var(--line);padding:clamp(.3rem,1vw,1rem)}
figure img{width:100%;display:block}figcaption{font-size:.88rem;line-height:1.5;border-top:1px solid var(--line);padding:.75rem .3rem .1rem}
.table{overflow:auto;margin:1.35rem 0}table{border-collapse:collapse;width:100%;font-size:.87rem;line-height:1.4;background:white}
th,td{text-align:left;padding:.62rem .68rem;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#e4efec;color:#173e43}
tbody tr:nth-child(even){background:#f3f6f4}pre{overflow:auto;background:#edf1ee;padding:.9rem}code{font-size:.92em;overflow-wrap:anywhere}
.skip{position:absolute;left:-100vw}.skip:focus{position:static}video{display:block;background:#172c35;width:100%;max-height:65vh}
.player{padding:.55rem;margin:.5rem 0;background:white;border:1px solid var(--line)}.player summary{cursor:pointer}
@media(max-width:880px){.frame{display:block;padding:1rem 4vw 3rem}aside>details{position:static;max-height:13rem;margin-bottom:1rem}header{gap:.8rem}}
@media print{body{font-size:10pt;background:white}header,aside,.skip,.player{display:none}.frame{display:block;padding:0}
h2,h3{break-after:avoid}figure,tr{break-inside:avoid}.table{overflow:visible}th,td{padding:.23rem}a{color:inherit}}
"""


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        self.urls.extend(attrs[k] for k in ("href", "src") if k in attrs)


def build(name):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    source = (DOC / (name + ".md")).read_text()
    title = escape(source.splitlines()[0][2:])
    body = md.convert(source)
    for page in PAGES:
        body = body.replace(f'href="{page}.md', f'href="{page}.html')

    def figure(match):
        image, caption = match.groups()
        svg = re.search(r'src="([^"]+)"', image)[1]
        data = base64.b64encode((DOC / svg).read_bytes()).decode()
        image = image.replace(svg, "data:image/svg+xml;base64," + data)
        pdf = escape(str(Path(svg).with_suffix(".pdf")), quote=True)
        return f'<figure>{image}<figcaption>{caption} <a href="{pdf}">Vector PDF</a></figcaption></figure>'

    body = re.sub(r'<p>(<img [^>]+>)</p>\s*<p>(<em>Figure [^\n]+?)</p>', figure, body)
    body = body.replace("<table>", '<div class="table" tabindex="0"><table>').replace("</table>", "</table></div>")
    if name == "video_gallery":
        clips = [("Selected fixed push, force-conditioned, 4101", "report_media/push_4101_B.mp4"),
                 ("Strict release versus retained contact, common cone 3109", "report_media/relic_cone_3109_311.mp4"),
                 ("Heavy cone, matched ordinary target succeeds", "../artifacts/final/3410/311-blind-Cz/dashboard.mp4")]
        players = "".join(f'<details class="player"><summary>Play {label}</summary>'
                          f'<video controls preload="none" aria-label="{label}" src="{src}"></video></details>'
                          for label, src in clips)
        body = body.replace("<h2 ", players + "<h2 ", 1)
    nav = '<header><strong>UniFP-MPC</strong><a href="research_report.html">Research report</a>' \
          '<a href="video_gallery.html">Recordings</a><a href="research_evidence.html">Evidence and reproduction</a></header>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f'<style>{STYLE}</style></head><body><a class="skip" href="#main">Skip to content</a>{nav}'
            f'<div class="frame"><aside><details open><summary>Contents</summary>{md.toc}</details></aside>'
            f'<main id="main">{body}</main></div></body></html>')


def verify_media(paths, ffmpeg):
    rows = []
    for path in sorted(paths):
        # Decode every frame. Metadata read from the input stream, before the null output.
        run = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-xerror", "-i",
                              str(path), "-map", "0:v:0", "-f", "null", "-"], check=True,
                             capture_output=True, text=True)
        stream = next(line for line in run.stderr.splitlines() if "Video:" in line)
        width, height = re.search(r"\b(\d{2,5})x(\d{2,5})\b", stream).groups()
        duration = re.search(r"Duration: ([\d:.]+)", run.stderr)[1]
        rows.append([path.relative_to(DOC.parent), width, height,
                     re.search(r"([\d.]+) fps", stream)[1], duration,
                     hashlib.sha256(path.read_bytes()).hexdigest()])
    with (DOC / "report_data/verified_media.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["repository_path", "width", "height", "fps", "duration_hms", "sha256"])
        writer.writerows(rows)
    print(f"Decoded {len(rows)} complete videos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-media", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    parsed = {}
    for page in PAGES:
        html = build(page)
        path = DOC / (page + ".html")
        path.write_text(html)
        parsed[path] = Links()
        parsed[path].feed(html)
    local = [(path, urlsplit(url)) for path, links in parsed.items() for url in links.urls
             if not urlsplit(url).scheme]
    if args.verify_media:
        verify_media({(DOC / unquote(url.path)).resolve() for _, url in local if url.path.endswith(".mp4")}, args.ffmpeg)
    for page, url in local:
        target = (DOC / unquote(url.path)).resolve() if url.path else page
        if not target.exists():
            raise FileNotFoundError(f"{page.name}: {url.path}")
        if url.fragment and target in parsed and url.fragment not in parsed[target].ids:
            raise ValueError(f"Missing section {url.fragment} in {target.name}")
    print(f"Verified {len(local)} local link and navigation targets across {len(parsed)} pages")


if __name__ == "__main__":
    main()
