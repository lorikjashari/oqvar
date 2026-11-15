"""Contact sheet of AutoCAD library hatch patterns (ezdxf) with names, to check stoa_qa.hatch_synth.PAT_CLASS by eye."""
import random, sys
from PIL import Image, ImageDraw
from ezdxf.tools import pattern
from stoa_qa import hatch_synth as hs

names = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(pattern.ISO_PATTERN)
hs._LIB = dict(pattern.ISO_PATTERN)
S, cols = 150, 8
W = Image.new("L", (cols * (S + 6), ((len(names) + cols - 1) // cols) * (S + 20)), 255)
d = ImageDraw.Draw(W)
rng = random.Random(0)
for i, n in enumerate(names):
    sp = hs._pattern_spacing(n)
    im = hs.render_pattern(n, S * 2, 14 * 2 / sp, 0, 2, 0, rng).resize((S, S), Image.LANCZOS)
    x, y = (i % cols) * (S + 6), (i // cols) * (S + 20)
    W.paste(im, (x, y + 14)); d.rectangle([x, y + 14, x + S, y + 14 + S], outline=0)
    d.text((x + 2, y), f"{n} -> {hs.PAT_CLASS.get(n)}", fill=0)
W.save(sys.argv[1])
