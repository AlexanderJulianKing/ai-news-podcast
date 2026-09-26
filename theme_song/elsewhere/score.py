"""'Elsewhere' - the bed under the news roundup. A minor/Dorian, 94 bpm. Claude, 2026-09-26.

v3 spoke jazz (walking bass, 9th and 6/9 chords, comping, bebop lines) and Alex found it too casual. This version
works the way The Daily's music is described: a pulse that keeps climbing ("asking a question"), plain chords that
never quite resolve, long low notes, and complexity from layers rather than licks. Every note is new; nothing is
taken from The Daily's theme.

Layers:
  pulse   a warm synth (GarageBand's Liquid Synth Keys) in unbroken eighths. It cycles a short cell of chord tones
          whose length (3, 4 or 5 eighths) does not fit the 4/4 bar, so its accents drift across the bar lines;
          within a section it climbs a chord tone every two bars, then starts lower in the next.
  piano   a second, sparser figure in a different cycle (hocket offbeats, dotted-quarter dyads, or a 5-cycle),
          so the two pulses interlock and shift against each other.
  motif   a three-note question on the piano, E-A-B over A minor (up a fourth, then a step to the 9th, held,
          unanswered), developed section by section: stated, extended, inverted, slowed, echoed by the piano.
  low     long low synth notes (GarageBand's New Carnival Synth) and a soft low piano octave at section starts.
  pad     the same warm synth, sustained, in some sections. (v4 used GarageBand's sampler strings: they sounded fake.)
  shaker  the head is one bar of shaker alone (Alex's call); underneath it stays straight and quiet.
Harmony is plain triads and sus chords over slow changes, with no V-i cadence anywhere.
"""
import numpy as np

import piano
import sampler as ck
import scales as melody
import shaker

BPM = 94
BEAT = 60.0 / BPM
BAR = 4 * BEAT
HEAD_BARS = 1

CHORDS = {   # bass pitch class, chord tones, colour tones allowed on long notes
    'Am':      (9, [9, 0, 4], [11, 2]),
    'Asus2':   (9, [9, 11, 4], [2]),
    'Fmaj7':   (5, [5, 9, 0, 4], [7]),
    'F':       (5, [5, 9, 0], [7, 4]),
    'C':       (0, [0, 4, 7], [2, 11]),
    'Csus2':   (0, [0, 2, 7], [4]),
    'G':       (7, [7, 11, 2], [9]),
    'Gsus4':   (7, [7, 0, 2], [9]),
    'Em':      (4, [4, 7, 11], [6, 2]),
    'Esus4':   (4, [4, 9, 11], [2]),
    'D':       (2, [2, 6, 9], [4]),
    'Dsus2':   (2, [2, 4, 9], [7]),
    'F/A':     (9, [5, 9, 0], [4, 7]),
    'G/A':     (9, [7, 11, 2], [9, 4]),
    'Dsus2/A': (9, [2, 4, 9], [7]),
    'Esus4/A': (9, [4, 9, 11], [2]),
}
BLOCKS = {
    'open':  ['Am', 'Am', 'Fmaj7', 'Fmaj7', 'Csus2', 'Csus2', 'Gsus4', 'G'],
    'pedal': ['Am', 'F/A', 'G/A', 'Am', 'Am', 'F/A', 'Dsus2/A', 'Esus4/A'],
    'rise':  ['F', 'G', 'Am', 'Am', 'F', 'G', 'C', 'Esus4'],
    'lift':  ['Dsus2', 'D', 'Am', 'Am', 'C', 'C', 'Gsus4', 'G'],
    'still': ['Am', 'Am', 'Am', 'Am', 'Fmaj7', 'Fmaj7', 'Fmaj7', 'Fmaj7'],
    'turn':  ['C', 'Em', 'F', 'Gsus4', 'Am', 'Em', 'Fmaj7', 'Esus4'],
}
# (block, pulse cell as ladder steps, piano figure, motif treatment, pad, shaker, climb)
SECTIONS = [
    ('open',  [0, 2, 4],       'none',   'none',    False, True,  False),
    ('pedal', [0, 2, 4, 5],    'hocket', 'state',   False, True,  True),
    ('rise',  [0, 1, 3],       'dyad',   'none',    True,  True,  True),
    ('still', [0, 2, 3, 5, 4], 'hocket', 'augment', True,  True,  False),
    ('lift',  [0, 2, 4, 6],    'arp5',   'extend',  False, True,  True),
    ('open',  [0, 3, 1, 4],    'dyad',   'invert',  True,  True,  False),
    ('turn',  [0, 2, 4],       'hocket', 'echo',    True,  False, True),
    ('pedal', [0, 1, 2, 4, 3], 'none',   'state',   False, True,  True),
    ('rise',  [0, 2, 4, 5],    'arp5',   'extend',  True,  True,  False),
    ('still', [0, 2, 4],       'hocket', 'invert',  True,  True,  True),
    ('lift',  [0, 3, 5],       'dyad',   'augment', False, True,  False),
    ('turn',  [0, 2, 3, 5],    'hocket', 'state',   True,  True,  True),
    ('open',  [0, 2, 4, 6, 5], 'arp5',   'echo',    True,  True,  False),
    ('pedal', [0, 2, 4],       'dyad',   'extend',  True,  True,  True),
    ('rise',  [0, 1, 3, 4],    'hocket', 'invert',  False, True,  False),
    ('still', [0, 2, 4, 5],    'arp5',   'state',   True,  True,  True),
]
MOTIF = [(0.0, 2.0, 76), (2.0, 1.0, 81), (3.0, 3.0, 83)]            # E5 A5 B5 over A minor: a question


def _ladder(chord, base, span=26, colour=True):
    root, tones, col = CHORDS[chord]
    pcs = set(tones) | (set(col[:1]) if colour else set())
    return [p for p in range(base, base + span) if p % 12 in pcs]


def _root_note(chord, lo):
    pc = CHORDS[chord][0]
    return lo + (pc - lo) % 12


def _fit_motif(notes, chord):
    """Move the A-minor motif onto `chord` by scale steps, landing every note on a chord or colour tone."""
    src_root, dst_root = 9, CHORDS[chord][0]
    shift = melody._to_steps(60 + dst_root)[0] - melody._to_steps(60 + src_root)[0]
    shift = (shift + 3) % 7 - 3
    ok = set(CHORDS[chord][1]) | set(CHORDS[chord][2])
    out = []
    for b, d, m in notes:
        m = melody.shift_diatonic(m, shift)
        if m % 12 not in ok:
            m = min((m + k for k in range(-2, 3) if (m + k) % 12 in ok), key=lambda n: (abs(n - m), n))
        out.append((b, d, m))
    return out


def _motif_for(kind, chords8):
    """(bar offset, [(beat, dur, pitch)]) statements for one section; beats relative to the statement's bar."""
    ext = MOTIF + [(6.0, 1.0, 84), (7.0, 3.0, 86)]                            # ... C6, D6: the question goes further
    inv = [(0.0, 2.0, 76), (2.0, 1.0, 71), (3.0, 3.0, 69)]                   # E5 B4 A4: an answer, not a resolution
    aug = [(b * 2, d * 2, m) for b, d, m in MOTIF]
    if kind == 'none':
        return []
    if kind == 'state':
        return [(0, _fit_motif(MOTIF, chords8[0])), (4, _fit_motif(MOTIF, chords8[4]))]
    if kind == 'extend':
        return [(0, _fit_motif(ext[:3], chords8[0]) + _fit_motif(ext[3:], chords8[1])), (4, _fit_motif(MOTIF, chords8[4]))]
    if kind == 'invert':
        return [(0, _fit_motif(MOTIF, chords8[0])), (4, _fit_motif(inv, chords8[4]))]
    if kind == 'augment':
        return [(0, _fit_motif(aug[:2], chords8[0]) + _fit_motif(aug[2:], chords8[1]))]
    if kind == 'echo':
        return [(0, _fit_motif(MOTIF, chords8[0])), (4, _fit_motif(MOTIF, chords8[4]))]
    return []


def shaker_bar(rng, t0, soft=1.0):
    """Straight, quiet sixteenths: a firmer stroke on each beat, the hand going forward and back."""
    notes = []
    for k in range(16):
        base = 52 if k % 4 == 0 else 38 if k % 2 == 0 else 32
        stroke = (60 if k % 4 == 0 else 61) if k % 2 == 0 else 62
        notes.append((t0 + k / 4, 0.2, stroke, int(np.clip((base + rng.integers(-4, 5)) * soft, 15, 110))))
    return notes


def head():
    """One bar of shaker alone (Alex: two bars was too many), easing in over its first beat."""
    P = {k: [] for k in ('synth', 'piano', 'low', 'pad', 'motif', 'shaker')}
    rng = np.random.default_rng(3)
    P['shaker'] = [(b, d, m, int(v * min(1.0, 0.6 + 0.4 * b))) for b, d, m, v in shaker_bar(rng, 0.0)]
    return P


def body(n_bars, seed=17):
    rng = np.random.default_rng(seed)
    P = {k: [] for k in ('synth', 'piano', 'low', 'pad', 'motif', 'shaker')}
    plan = []
    for rep in range(3):
        for si, sec in enumerate(SECTIONS):
            for i, c in enumerate(BLOCKS[sec[0]]):
                plan.append((sec, rep * len(SECTIONS) + si, i, c))
    plan = plan[:n_bars]
    chords = [p[3] for p in plan]
    e_global = 0
    motifs = {}
    for b, (sec, sec_id, i, chord) in enumerate(plan):
        block, cell, pfig, mkind, pad, shk, climb = sec
        t0 = 4 * b
        # the pulse: unbroken eighths cycling the cell across the bar lines, climbing every two bars
        lift = (i // 2) if climb else 0
        ladder = _ladder(chord, 45)
        L = len(cell)
        for k in range(8):
            pos = (e_global + k) % L
            step = cell[pos] + lift
            m = ladder[min(step, len(ladder) - 1)]
            v = 60 if pos == 0 else 44
            P['synth'].append((t0 + k / 2, 0.55, m, int(v + rng.integers(-3, 4))))
        # the piano's figure, in its own cycle
        pl = _ladder(chord, 60, colour=False)
        if pfig == 'hocket':        # offbeats, a 3-cycle over the upper chord tones
            for k in range(1, 8, 2):
                m = pl[((e_global + k) // 2) % 3 + (1 if lift >= 2 else 0)]
                P['piano'].append((t0 + k / 2, 0.45, m, int(44 + rng.integers(-3, 4))))
        elif pfig == 'dyad':        # dotted quarters: three against the bar's four
            for k in range(0, 8):
                if (e_global + k) % 3 == 0:
                    P['piano'] += [(t0 + k / 2, 1.3, m, int(42 + rng.integers(-3, 4))) for m in pl[1:3]]
        elif pfig == 'arp5':        # a rising five-eighth figure on the even eighths
            for k in range(0, 8, 2):
                m = pl[((e_global + k) // 2) % 5 % len(pl)]
                P['piano'].append((t0 + k / 2, 0.9, m, int(44 + rng.integers(-3, 4))))
        e_global += 8
        # long low notes when the harmony moves (or at least every other bar), a soft low piano octave at section starts
        prev = chords[b - 1] if b else None
        if i == 0 or chord != prev or i % 2 == 0:
            dur = 4.0 * (2 if (b + 1 < len(chords) and chords[b + 1] == chord and i % 2 == 0) else 1) - 0.1
            P['low'].append((t0, dur, _root_note(chord, 33), 56))
        if i == 0:
            r = _root_note(chord, 33)
            P['piano'] += [(t0, 3.5, r, 40), (t0, 3.5, r + 12, 36)]
        if pad and i % 2 == 0:
            pd = [p for p in _ladder(chord, 55, colour=False)][:3]
            P['pad'] += [(t0, 7.8, m, 36) for m in pd]
        if shk:
            P['shaker'] += shaker_bar(rng, t0)
        # the motif (strings), planned per section
        if i == 0:
            sec_chords = BLOCKS[block]
            for off, notes in _motif_for(mkind, sec_chords):
                for bb, d, m in notes:
                    P['motif'].append((t0 + 4 * off + bb, d, m, int(62 + rng.integers(-3, 4))))
                if mkind == 'echo':    # the piano answers the tail an octave down, a bar later
                    for bb, d, m in notes[1:]:
                        P['piano'].append((t0 + 4 * off + 4 + bb, min(d, 1.5), m - 12, 44))
    return P, chords


def ending():
    """After the last word: the pulse climbs one more bar under the motif's question over Fmaj7, then an open Asus2
    (A E B) is left ringing with the 9th on top: it ends on the question. Beats from the ending downbeat."""
    P = {k: [] for k in ('synth', 'piano', 'low', 'pad', 'motif', 'shaker')}
    ladder = _ladder('Fmaj7', 41)
    for k in range(8):
        P['synth'].append((k / 2, 0.55, ladder[min(k, len(ladder) - 1)], 40 + 2 * k))
    P['synth'].append((4.0, 1.5, 45, 54))
    P['motif'] += [(0.0, 1.5, 76, 58), (1.5, 1.0, 81, 62), (2.5, 3.7, 83, 66)]
    P['low'] += [(0.0, 3.9, 41, 50), (4.0, 2.6, 33, 58)]
    P['pad'] += [(0.0, 3.9, m, 36) for m in (57, 60, 64)] + [(4.0, 2.4, m, 44) for m in (45, 52, 59, 64)]
    P['piano'] += [(4.0, 2.6, m, 46) for m in (33, 45, 52, 59, 64)]
    rng = np.random.default_rng(5)
    P['shaker'] += [(b, d, m, int(v * (1 - b / 4))) for b, d, m, v in shaker_bar(rng, 0.0) if b < 3.0]
    return P


# ------------------------------------------------------------------ rendering and export

INSTR = dict(synth='synth', low='carnival', pad='carnival')     # no strings: GarageBand's sampler strings sounded fake
BALANCE_HEAD = dict(shaker=0.0)
BALANCE_BODY = dict(synth=-3.0, motif=-4.0, low=-5.0, piano=-8.0, pad=-11.0, shaker=-7.0)   # shaker audible: it carries the head
BALANCE_END = dict(motif=0.0, pad=-4.0, low=-4.0, piano=-4.0, synth=-6.0, shaker=-14.0)


def render_parts(P, total_s, workdir, rng, name):
    stems = {}
    for part, notes in P.items():
        if not notes:
            continue
        jitter = dict(synth=3, piano=5, low=0, pad=0, motif=0, shaker=4)[part]
        sec = ck.humanize([(b * BEAT, d * BEAT, m, v) for b, d, m, v in notes], rng, t_ms=jitter, v=2)
        if part in ('piano', 'motif'):
            stems[part] = piano.render(sec, total_s)
        elif part == 'shaker':
            stems[part] = shaker.render(sec, total_s)
        else:
            stems[part] = ck.render_part(f'{name}_{part}', INSTR[part], sec, total_s, workdir)
    n = max(s.shape[1] for s in stems.values())
    return {k: np.pad(s, ((0, 0), (0, n - s.shape[1]))) for k, s in stems.items()}


def write_midi(parts, path):
    import mido
    tpb = 480
    mf = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM)))
    mf.tracks.append(meta)
    programs = dict(synth=81, piano=0, low=38, pad=89, motif=0, shaker=0)
    for ch, name in enumerate(['synth', 'piano', 'motif', 'low', 'pad', 'shaker']):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage('track_name', name=name))
        chan = 9 if name == 'shaker' else ch
        tr.append(mido.Message('program_change', program=programs[name], channel=chan))
        ev = []
        for off, P in parts:
            for b, d, m, v in P.get(name, []):
                ev.append((int(round((b + off) * tpb)), 1, m, v))
                ev.append((int(round((b + off + d) * tpb)), 0, m, 0))
        ev.sort(key=lambda e: (e[0], e[1]))
        now = 0
        for t, on, m, v in ev:
            tr.append(mido.Message('note_on' if on else 'note_off', note=m, velocity=v, channel=chan, time=t - now))
            now = t
        mf.tracks.append(tr)
    mf.save(path)
