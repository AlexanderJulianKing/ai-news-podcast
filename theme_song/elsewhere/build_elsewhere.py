"""Build the assets for 'Elsewhere', the roundup's music bed (run on a Mac with GarageBand/Logic's instruments).

Renders the score (score.py) once: a one-bar shaker head, BODY_BARS of body (everything but the shaker, and the
shaker on its own track, so the Pi can ease the bed in without touching the shaker), and the ending, each with the
room reverb included, into elsewhere_assets.npz next to this file. The Pi (newscaster/audio/roundup_bed.py) fits
them to each day's roundup. The head is pre-scaled to the shaker's level inside the bed, so the shaker never
changes level at the join. The file is Apple sample content: copy it to the Pi and never commit it.

    python3 theme_song/elsewhere/build_elsewhere.py
    scp theme_song/elsewhere/elsewhere_assets.npz alex@raspberrypi...:/home/alex/ai-news-podcast/theme_song/elsewhere/

Needs numpy, soundfile and librosa (Mac only).
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import sampler  # noqa: E402
import score  # noqa: E402
from newscaster.audio.off_the_wire import _reverb  # noqa: E402
from newscaster.audio.roundup_bed import _band_dip  # noqa: E402

BODY_BARS = 150          # 6.4 minutes: longer than any roundup so far (357 s)
REVERB = dict(wet=0.14, seconds=2.2)


def balanced(stems, balance):
    """Each stem at its balance relative to the others, measured where it plays."""
    out = {}
    for k, s in stems.items():
        L = sampler.lufs(s)
        if np.isfinite(L):
            out[k] = s * 10 ** ((balance.get(k, -20.0) - L) / 20)
    return out


def main(path=os.path.join(HERE, 'elsewhere_assets.npz')):
    rng = np.random.default_rng(21)
    work = tempfile.mkdtemp(prefix='elsewhere_')
    Ph, (Pb, chords), Pe = score.head(), score.body(BODY_BARS), score.ending()
    head = sum(balanced(score.render_parts(Ph, score.HEAD_BARS * score.BAR + 3, work, rng, 'head'), score.BALANCE_HEAD).values())
    body = balanced(score.render_parts(Pb, BODY_BARS * score.BAR + 3, work, rng, 'body'), score.BALANCE_BODY)
    end = sum(balanced(score.render_parts(Pe, 2 * score.BAR + 4, work, rng, 'end'), score.BALANCE_END).values())
    shaker = body.pop('shaker')
    rest = sum(body.values())
    n = max(rest.shape[1], shaker.shape[1])
    rest = np.pad(rest, ((0, 0), (0, n - rest.shape[1])))
    shaker = np.pad(shaker, ((0, 0), (0, n - shaker.shape[1])))
    rest, shaker, head, end = (_reverb(x, seed=13, **REVERB) for x in (rest, shaker, head, end))
    # the head at the shaker's level inside the bed (as heard: speech band dipped), measured over the first two bars
    two = int(2 * score.BAR * sampler.SR)
    target = sampler.lufs(_band_dip(shaker[:, :two].astype(np.float32)))
    head *= 10 ** ((target - sampler.lufs(head[:, :int(score.HEAD_BARS * score.BAR * sampler.SR)])) / 20)
    arrays = dict(head=head, body=rest, body_shaker=shaker, ending=end)
    peak = max(float(np.abs(a).max()) for a in arrays.values())
    scale = peak / 32767.0
    ints = {k: np.round(a / scale).astype('<i2') for k, a in arrays.items()}
    np.savez_compressed(path, scale=np.array(scale), bpm=np.array(score.BPM), head_bars=np.array(score.HEAD_BARS),
                        body_bars=np.array(BODY_BARS), **ints)
    print(f"Elsewhere assets: head {head.shape[1] / sampler.SR:.1f}s, body {rest.shape[1] / sampler.SR:.1f}s "
          f"({BODY_BARS} bars), ending {end.shape[1] / sampler.SR:.1f}s -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == '__main__':
    main()
