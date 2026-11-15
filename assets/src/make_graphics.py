#!/usr/bin/env python3
"""Generate Clamor's README diagrams (light + dark) and the social preview banner.

Usage: python3 assets/src/make_graphics.py            # writes into assets/
Then render the banner PNG with:
    inkscape assets/social-preview.svg -o assets/png/social-preview.png -w 1280 -h 640

Palette comes from the icon: coral #fd5453 -> #fc8b61.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

THEMES = {
    "light": dict(text="#2b1d1a", muted="#7a6460", card="#fff8f5", stroke="#f2c9bc", line="#e2715f", chipText="#ffffff",
                  seg=["#fd5453", "#fc7a5c", "#fca07c"], segText=["#ffffff", "#ffffff", "#5a2318"], panel="#fffbf9"),
    "dark": dict(text="#f6ece9", muted="#bba7a2", card="#2a1f1d", stroke="#5b3d35", line="#ff7f6a", chipText="#ffffff",
                 seg=["#fd5453", "#fc7a5c", "#fca07c"], segText=["#ffffff", "#ffffff", "#5a2318"], panel="#211816"),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, fill="#000", weight=400, anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def arrow_marker(mid, color):
    return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')


# Device glyphs, drawn in a 40x32 box at (x, y).
def glyph(kind, x, y, c):
    s = f'stroke="{c}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"'
    g = {
        "desktop": f'<rect x="0" y="0" width="40" height="26" rx="3" {s}/><path d="M20 26v5M12 31h16" {s}/>',
        "laptop": f'<rect x="5" y="2" width="30" height="21" rx="2" {s}/><path d="M1 27h38l-3 4H4z" {s}/>',
        "mini": f'<rect x="2" y="9" width="36" height="18" rx="5" {s}/><circle cx="10" cy="18" r="1.6" fill="{c}"/><path d="M17 18h14" {s}/>',
        "phone": f'<rect x="11" y="0" width="18" height="32" rx="4" {s}/><path d="M17.5 27.5h5" {s}/>',
        "tower": f'<rect x="10" y="0" width="20" height="32" rx="3" {s}/><path d="M15 7h10M15 12h10" {s}/><circle cx="20" cy="24" r="2" fill="{c}"/>',
    }[kind]
    return f'<g transform="translate({x},{y})">{g}</g>'


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>{body}</svg>\n')


def split_model(t):
    """One big model, cut into layer ranges, each on a different machine, joined in a ring."""
    W, H = 960, 470
    b = [f"<defs>{arrow_marker('a', t['line'])}</defs>"]
    # The model bar.
    b.append(text(40, 34, "One 70B model · 80 layers · about 40 GB", 15, t["text"], 600))
    b.append(text(920, 34, "too big for any one of these machines", 14, t["muted"], 400, "end"))
    x0, x1, y, bh = 40, 920, 48, 40
    layers = [(1, 40), (41, 68), (69, 80)]
    spans = []
    x = x0
    for i, (a, z) in enumerate(layers):
        w = (z - a + 1) / 80 * (x1 - x0)
        r = 10
        # Round only the outer corners of the whole bar.
        if i == 0:
            d = f"M{x + r},{y} H{x + w} V{y + bh} H{x + r} Q{x},{y + bh} {x},{y + bh - r} V{y + r} Q{x},{y} {x + r},{y} Z"
        elif i == len(layers) - 1:
            d = f"M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + bh - r} Q{x + w},{y + bh} {x + w - r},{y + bh} H{x} Z"
        else:
            d = f"M{x},{y} H{x + w} V{y + bh} H{x} Z"
        b.append(f'<path d="{d}" fill="{t["seg"][i]}"/>')
        b.append(text(x + w / 2, y + 25, f"layers {a}–{z}", 14, t["segText"][i], 600, "middle"))
        spans.append(x + w / 2)
        x += w

    # Machines.
    machines = [("desktop", "Gaming PC", "32 GB RAM", "20 GB of the model"),
                ("laptop", "Laptop", "16 GB RAM", "14 GB of the model"),
                ("mini", "Mini PC", "8 GB RAM", "6 GB of the model")]
    cw, ch, cy = 236, 150, 170
    centers = [40 + cw / 2, 480, 920 - cw / 2]
    for i, ((kind, name, ram, share), cx) in enumerate(zip(machines, centers)):
        # Dashed lead from the bar segment to its machine.
        b.append(f'<path d="M{spans[i]},{y + bh + 4} C{spans[i]},{y + bh + 40} {cx},{cy - 40} {cx},{cy - 6}" stroke="{t["seg"][i]}" stroke-width="2" stroke-dasharray="4 5" fill="none"/>')
        x = cx - cw / 2
        b.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="{t["card"]}" stroke="{t["stroke"]}" stroke-width="1.5"/>')
        b.append(glyph(kind, cx - 20, cy + 22, t["text"]))
        b.append(text(cx, cy + 80, name, 17, t["text"], 650, "middle"))
        b.append(text(cx, cy + 102, ram, 14, t["muted"], 400, "middle"))
        b.append(f'<rect x="{cx - 78}" y="{cy + 114}" width="156" height="24" rx="12" fill="{t["seg"][i]}"/>')
        b.append(text(cx, cy + 131, share, 12.5, t["segText"][i], 600, "middle"))

    # Ring: forward hops between cards, and the next word back to the first machine.
    mid = cy + ch / 2
    for i in range(2):
        xa, xb = centers[i] + cw / 2 + 6, centers[i + 1] - cw / 2 - 8
        b.append(f'<path d="M{xa},{mid} H{xb}" stroke="{t["line"]}" stroke-width="2.5" fill="none" marker-end="url(#a)"/>')
    by = cy + ch + 42
    b.append(f'<path d="M{centers[2]},{cy + ch + 4} V{by - 14} Q{centers[2]},{by} {centers[2] - 14},{by} H{centers[0] + 14} Q{centers[0]},{by} {centers[0]},{by - 14} V{cy + ch + 10}" '
             f'stroke="{t["line"]}" stroke-width="2.5" fill="none" marker-end="url(#a)"/>')
    b.append(f'<rect x="{480 - 88}" y="{by - 14}" width="176" height="28" rx="14" fill="{t["line"]}"/>')
    b.append(text(480, by + 5, "next word goes back", 13.5, t["chipText"], 600, "middle"))
    b.append(text(480, by + 50, "Each word makes one trip around the ring: every machine runs its share of the layers and passes the result on.", 14, t["muted"], 400, "middle"))
    return svg(W, H, "".join(b), "One big model split across three machines",
               "A 70-billion-parameter model of about 40 GB is cut into three layer ranges. A gaming PC runs layers 1 to 40, "
               "a laptop runs 41 to 68 and a mini PC runs 69 to 80. Each passes its result to the next, and the last sends the "
               "next word back to the first.")


def two_lanes(t):
    """The two kinds of work: one big model as a ring, and many small jobs spread out."""
    W, H = 960, 400
    b = [f"<defs>{arrow_marker('a', t['line'])}{arrow_marker('m', t['muted'])}</defs>"]
    pw, ph, py = 450, 360, 20
    for px, title, sub in [(20, "One big model", "Too big for any single machine? Split it."), (490, "Lots of small jobs", "Spread them out. Every device helps.")]:
        b.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="20" fill="{t["panel"]}" stroke="{t["stroke"]}" stroke-width="1.5"/>')
        b.append(text(px + 28, py + 42, title, 20, t["text"], 700))
        b.append(text(px + 28, py + 66, sub, 14.5, t["muted"]))

    # Left: ring of four machines.
    cx, cy, r = 245, 245, 92
    import math
    nodes = [("desktop", -90), ("laptop", 0), ("tower", 90), ("mini", 180)]
    pts = []
    for kind, ang in nodes:
        a = math.radians(ang)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a), kind))
    for i in range(4):
        x1_, y1_, _ = pts[i]
        x2_, y2_, _ = pts[(i + 1) % 4]
        a1 = math.radians(nodes[i][1] + 24)
        a2 = math.radians(nodes[(i + 1) % 4][1] - 26)
        sx, sy = cx + r * math.cos(a1), cy + r * math.sin(a1)
        ex, ey = cx + r * math.cos(a2), cy + r * math.sin(a2)
        b.append(f'<path d="M{sx:.1f},{sy:.1f} A{r},{r} 0 0 1 {ex:.1f},{ey:.1f}" stroke="{t["line"]}" stroke-width="2.5" fill="none" marker-end="url(#a)"/>')
    for i, (x, y, kind) in enumerate(pts):
        b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="30" fill="{t["card"]}" stroke="{t["seg"][min(i, 2)]}" stroke-width="2.5"/>')
        b.append(glyph(kind, x - 20, y - 16, t["text"]))
    b.append(text(cx, cy - 4, "one chat", 14, t["text"], 650, "middle"))
    b.append(text(cx, cy + 15, "around the ring", 12.5, t["muted"], 400, "middle"))

    # Right: a queue of jobs fanning out to devices, including a phone.
    jobs = ["Summarize", "Embed", "Transcribe", "Read text (OCR)", "Tool call"]
    jx, jy0, jh = 520, 110, 46
    for i, j in enumerate(jobs):
        y = jy0 + i * jh
        b.append(f'<rect x="{jx}" y="{y}" width="150" height="34" rx="17" fill="{t["seg"][i % 3]}"/>')
        b.append(text(jx + 75, y + 22, j, 13.5, t["segText"][i % 3], 600, "middle"))
    devs = [("desktop", "PC"), ("laptop", "Laptop"), ("mini", "Mini PC"), ("phone", "Phone")]
    dx, dy0, dh = 830, 112, 62
    for i, (kind, name) in enumerate(devs):
        y = dy0 + i * dh
        b.append(f'<circle cx="{dx}" cy="{y + 16}" r="28" fill="{t["card"]}" stroke="{t["stroke"]}" stroke-width="1.5"/>')
        b.append(glyph(kind, dx - 20, y, t["text"]))
        b.append(text(dx + 38, y + 21, name, 13, t["muted"]))
    # Monotone job -> device mapping: no crossing lines, and every device gets work.
    links = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)]
    for j, d in links:
        sy = jy0 + j * jh + 17
        ey = dy0 + d * dh + 16
        b.append(f'<path d="M{jx + 156},{sy} C{jx + 220},{sy} {dx - 100},{ey} {dx - 36},{ey}" stroke="{t["line"]}" stroke-width="2" fill="none" opacity="0.85" marker-end="url(#a)"/>')
    return svg(W, H, "".join(b), "Two kinds of work",
               "Left: one big model split across four machines that pass each word around a ring. "
               "Right: small jobs such as summarize, embed, transcribe, OCR and tool calls are sent to whichever device is free, including a phone.")


def banner(mark_svg):
    """1280x640 social preview: coral gradient, the crab ring, wordmark and tagline."""
    inner = re.search(r"<svg[^>]*>(.*)</svg>", mark_svg, re.S).group(1)
    vx, vy, vw, _ = (float(v) for v in re.search(r'viewBox="([^"]*)"', mark_svg).group(1).split())
    W, H = 1280, 640
    b = [
        '<defs><linearGradient id="bg" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#fd5453"/><stop offset="1" stop-color="#fc8b61"/></linearGradient></defs>',
        f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
        # Mark: its viewBox scaled to 380 px, centred at (290, 320).
        f'<g transform="translate(100,130) scale({380 / vw}) translate({-vx},{-vy})">{inner}</g>',
        text(548, 300, "Clamor", 124, "#ffffff", 800, "start", 'letter-spacing="-2"'),
        text(552, 366, "Pool the idle computers on your network", 32, "#fff3ef", 500),
        text(552, 410, "to run AI none of them could run alone.", 32, "#fff3ef", 500),
    ]
    return svg(W, H, "".join(b), "Clamor", "Clamor: pool the idle computers on your network to run AI none of them could run alone.")


def main():
    for name, t in THEMES.items():
        open(os.path.join(OUT, f"diagram-split-{name}.svg"), "w").write(split_model(t))
        open(os.path.join(OUT, f"diagram-lanes-{name}.svg"), "w").write(two_lanes(t))
    mark = open(os.path.join(OUT, "clamor-mark-white.svg")).read()
    open(os.path.join(OUT, "social-preview.svg"), "w").write(banner(mark))
    print("wrote diagrams and social-preview.svg to", OUT)


if __name__ == "__main__":
    main()
