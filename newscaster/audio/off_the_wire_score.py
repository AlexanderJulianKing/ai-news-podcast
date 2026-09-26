"""'Off the Wire': the notes of the news-edition intro (composed by Claude, 2026-09-25).

A one-second logo (three taps on B into an E minor hit), then a groove under the host's reading that never
repeats a bar back to back: 8-bar phrases (Em9 A C D Em9 A C B7) that grow busier as the reading goes on,
with short fills in the host's sentence breaks. Then a one-bar turnaround, a breath just after the last word,
and a one-bar button: the taps over F sliding down to the final E minor chord.

Everything here is data plus numpy, so it runs on the Pi. Pitches are MIDI numbers (60 = middle C).
Bar events are (pos16, dur16, [pitches], velocity, articulation), articulation '>' accent, '.' staccato.
The day's variation comes from a random generator seeded with the date, so a date always gives the same intro.
"""

import numpy as np

BPM = 100
BEAT = 60.0 / BPM            # 0.6 s
BAR = 4 * BEAT               # 2.4 s
T0 = BEAT                    # the pickup taps take one beat; the logo hit lands at T0
BUTTON_RING = 1.8            # extra ring on the button's final chord, seconds

_PC = {'c': 0, 'cs': 1, 'd': 2, 'ds': 3, 'e': 4, 'f': 5, 'fs': 6, 'g': 7, 'gs': 8, 'a': 9, 'as': 10, 'b': 11}


def P(name):
    """'fs4' -> 66."""
    return _PC[name[:-1]] + 12 * (int(name[-1]) + 1)


def _ev(pos, dur, notes, vel, art=''):
    return (pos, dur, [P(n) for n in notes], vel, art)


def _stabs(positions, chord, vel):
    return [_ev(p, 1, chord, vel + (4 if p % 4 == 2 else 0), '.') for p in positions]


# --- fixed parts -------------------------------------------------------------------------------------------
_EM9 = ['fs4', 'g4', 'b4', 'd5']
PICKUP = {'RH': [_ev(13, 1, ['b4', 'b5'], 92, '>'), _ev(14, 1, ['b4', 'b5'], 76), _ev(15, 1, ['b4', 'b5'], 86)], 'LH': []}
LOGO = {'RH': [_ev(0, 3, ['g4', 'b4', 'e5', 'b5'], 100, '>')] + _stabs([7, 10, 14], _EM9, 60),
        'LH': [_ev(0, 2, ['e1', 'e2'], 104, '>'), _ev(3, 1, ['e3'], 66, '.'), _ev(6, 1, ['d3'], 70, '.'), _ev(7, 2, ['e3'], 80),
               _ev(10, 1, ['g2'], 72, '.'), _ev(11, 1, ['a2'], 74, '.'), _ev(12, 2, ['b2'], 84), _ev(14, 1, ['d3'], 68, '.'),
               _ev(15, 1, ['b2'], 64, '.')]}
# B7sus -> B7 under the host's last words; beat 4 is left empty: the breath
TURN = {'RH': [_ev(0, 2, ['a4', 'b4', 'e5'], 72, '.'), _ev(3, 1, ['a4', 'b4', 'e5'], 64, '.'), _ev(6, 2, ['a4', 'b4', 'e5'], 76, '.'),
               _ev(8, 2, ['a4', 'ds5', 'fs5'], 84, '>'), _ev(10, 2, ['a4', 'c5', 'ds5'], 90, '>')],
        'LH': [_ev(0, 2, ['b1'], 86, '>'), _ev(3, 1, ['b2'], 66, '.'), _ev(6, 1, ['a2'], 72, '.'), _ev(7, 1, ['b2'], 76),
               _ev(8, 2, ['cs3'], 84), _ev(10, 2, ['ds3'], 92, '>')]}
# the taps over F (the flat-II), sliding down a half step to the final E minor chord; one high tap to close
BUTTON = {'RH': [_ev(0, 1, ['b4', 'b5'], 104, '>'), _ev(1, 1, ['b4', 'b5'], 86), _ev(2, 2, ['b4', 'b5'], 98, '.'),
                 _ev(4, 2, ['a4', 'c5', 'e5'], 96, '>'), _ev(8, 5, ['g4', 'b4', 'e5', 'b5'], 108, '>'), _ev(14, 1, ['b6'], 66, '.')],
          'LH': [_ev(0, 4, ['f1', 'f2'], 104, '>'), _ev(4, 2, ['c3'], 84), _ev(8, 8, ['e1', 'e2'], 112, '>')]}

# --- the groove --------------------------------------------------------------------------------------------
PHRASE = ['Em', 'A', 'C', 'D', 'Em', 'A', 'C', 'B7']
LHM = {'Em': dict(R='e2', O='e3', F='b2', X='d3', p=('g2', 'a2')), 'A': dict(R='a1', O='a2', F='e2', X='fs2', p=('cs3', 'b2')),
       'C': dict(R='c2', O='c3', F='g2', X='b2', p=('e2', 'a2')), 'D': dict(R='d2', O='d3', F='a2', X='b2', p=('fs2', 'e2')),
       'B7': dict(R='b1', O='b2', F='fs2', X='a2', p=('ds3', 'cs3'))}
ROOT = {'Em': 'e2', 'A': 'a1', 'C': 'c2', 'D': 'd2', 'B7': 'b1'}
VOIC = {'Em': [['fs4', 'g4', 'b4', 'd5'], ['d4', 'fs4', 'g4', 'b4'], ['g4', 'b4', 'd5', 'fs5'], ['a4', 'd5', 'g5'], ['b4', 'd5', 'fs5', 'a5']],
        'A': [['a4', 'b4', 'cs5', 'e5'], ['e4', 'a4', 'b4', 'cs5'], ['cs5', 'e5', 'fs5', 'a5'], ['e4', 'fs4', 'a4', 'cs5'], ['cs5', 'fs5', 'b5']],
        'C': [['fs4', 'g4', 'b4', 'e5'], ['e4', 'fs4', 'g4', 'b4'], ['b4', 'e5', 'fs5', 'g5'], ['b4', 'e5', 'a5'], ['d5', 'fs5', 'g5', 'b5']],
        'D': [['fs4', 'a4', 'b4', 'e5'], ['e4', 'fs4', 'a4', 'b4'], ['a4', 'b4', 'e5', 'fs5'], ['e4', 'a4', 'b4', 'd5'], ['cs5', 'e5', 'fs5', 'a5']],
        'B7': [['a4', 'b4', 'e5'], ['a4', 'ds5', 'fs5'], ['a4', 'c5', 'ds5'], ['a4', 'ds5', 'g5'], ['fs4', 'a4', 'ds5']]}
SCALE = {'Em': ['e4', 'g4', 'a4', 'b4', 'd5', 'e5', 'g5'], 'A': ['a4', 'b4', 'cs5', 'e5', 'fs5', 'a5'],
         'C': ['e4', 'g4', 'a4', 'b4', 'e5', 'fs5', 'g5'], 'D': ['fs4', 'a4', 'b4', 'd5', 'e5', 'fs5', 'a5'],
         'B7': ['b4', 'ds5', 'fs5', 'a5', 'b5']}
# left-hand patterns: (pos, dur, role, velocity, articulation); roles R root, O octave, F fifth, X colour, p0/p1 passing, lead
LH_T = {
    'drive': [(0, 2, 'R', 92, '>'), (3, 1, 'O', 66, '.'), (6, 1, 'X', 70, '.'), (7, 2, 'O', 80, ''), (10, 1, 'p0', 72, '.'),
              (11, 1, 'p1', 74, '.'), (12, 2, 'F', 84, ''), (14, 1, 'X', 68, '.'), (15, 1, 'lead', 64, '.')],
    'pops': [(0, 2, 'R', 92, '>'), (2, 1, 'O', 70, '.'), (4, 1, 'R', 66, '.'), (6, 2, 'O', 82, ''), (9, 1, 'F', 70, '.'),
             (10, 2, 'O', 80, ''), (13, 1, 'X', 68, '.'), (14, 2, 'lead', 74, '')],
    'sparse': [(0, 3, 'R', 90, '>'), (3, 1, 'O', 64, '.'), (7, 1, 'F', 70, '.'), (8, 2, 'R', 80, ''), (11, 1, 'O', 66, '.'),
               (12, 1, 'F', 70, '.'), (14, 2, 'lead', 74, '')],
    'walk': [(0, 2, 'R', 90, '>'), (2, 2, 'F', 76, ''), (4, 2, 'O', 78, ''), (6, 1, 'X', 66, '.'), (7, 1, 'O', 70, '.'),
             (8, 2, 'F', 78, ''), (10, 2, 'p0', 74, ''), (12, 2, 'O', 80, ''), (14, 2, 'lead', 76, '')],
    'stop': [(0, 2, 'R', 100, '>'), (6, 2, 'O', 96, '>'), (14, 2, 'lead', 80, '')],
    'bounce': [(0, 1, 'R', 92, '>'), (1, 1, 'O', 64, '.'), (3, 2, 'F', 78, ''), (6, 1, 'R', 70, '.'), (7, 1, 'O', 74, '.'),
               (8, 2, 'R', 82, ''), (11, 1, 'X', 66, '.'), (12, 1, 'O', 70, '.'), (13, 1, 'F', 68, '.'), (14, 2, 'lead', 76, '')],
    'halftime': [(0, 4, 'R', 92, '>'), (6, 2, 'O', 78, ''), (8, 2, 'F', 76, ''), (12, 2, 'R', 80, ''), (14, 2, 'lead', 74, '')],
    'sixteen': [(0, 1, 'R', 92, '>'), (1, 1, 'R', 60, '.'), (2, 1, 'O', 72, '.'), (4, 1, 'F', 70, '.'), (6, 1, 'O', 76, '.'),
                (7, 1, 'R', 62, '.'), (8, 1, 'O', 78, '.'), (10, 1, 'p0', 72, '.'), (11, 1, 'p1', 74, '.'), (12, 2, 'F', 82, ''),
                (14, 1, 'O', 70, '.'), (15, 1, 'lead', 66, '.')],
    'push4': [(0, 2, 'R', 92, '>'), (3, 1, 'O', 66, '.'), (5, 1, 'F', 72, '.'), (7, 2, 'O', 82, ''), (10, 2, 'R', 80, ''),
              (13, 1, 'X', 68, '.'), (14, 2, 'lead', 76, '')],
}
# right-hand stab rhythms (16th positions)
RH_T = {'light': [2, 7, 10, 14], 'push': [2, 6, 10, 13], 'lazy': [3, 6, 11], 'busy': [2, 5, 8, 12, 14], 'sparse': [6, 14],
        'stop': [0, 6], 'offbeat': [2, 6, 10, 14], 'anticip': [3, 7, 11, 15], 'double': [2, 3, 10, 11], 'charleston': [0, 6],
        'late': [5, 13], 'syncopa': [1, 6, 9, 14]}
# per phrase: the patterns to draw from, the voicings, the chance of a crushed grace note, a velocity offset
ARC = [dict(lh=['sparse', 'drive', 'halftime'], rh=['sparse', 'lazy', 'late', 'charleston'], v=[0, 1, 3], crush=0.0, vel=-6),
       dict(lh=['drive', 'pops', 'push4', 'bounce'], rh=['light', 'push', 'offbeat', 'anticip'], v=[0, 1, 2, 3], crush=0.35, vel=-2),
       dict(lh=['walk', 'drive', 'bounce', 'sixteen'], rh=['busy', 'light', 'double', 'syncopa'], v=[0, 2, 3, 4], crush=0.3, vel=0),
       dict(lh=['pops', 'walk', 'sixteen', 'push4'], rh=['push', 'busy', 'anticip', 'double'], v=[1, 2, 3, 4], crush=0.4, vel=2,
            first='stop')]
WALKUP = ['b1', 'cs2', 'd2', 'ds2']       # chromatic walk up into E at the end of a phrase


def _lead_to(next_chord, cur):
    """A bass approach note into the next bar's root: a step below or above, whichever sits nearest."""
    tgt = P(ROOT[next_chord])
    return min([tgt - 1, tgt + 2, tgt - 2], key=lambda m: abs(m - P(LHM[cur]['F'])))


def _arc(phrase):
    return ARC[phrase] if phrase < len(ARC) else ARC[1 + (phrase - 1) % (len(ARC) - 1)]


def groove(n, seed):
    """Bars 1..n-1 of the groove (bar 0 is the logo bar). Each bar: chord, LH/RH events, the patterns used, phrase."""
    rng = np.random.default_rng(seed)
    out, last, prev = [], {}, None
    for i in range(1, n):
        chord = PHRASE[i % 8]
        phrase = i // 8
        arc = _arc(phrase)
        nxt = PHRASE[(i + 1) % 8] if i + 1 < n else 'B7'
        for _ in range(20):          # this chord's bar must differ from its last appearance, and from the bar before
            lt = 'stop' if (arc.get('first') == 'stop' and i % 8 == 0) else str(rng.choice(arc['lh']))
            rt = 'stop' if lt == 'stop' else str(rng.choice(arc['rh']))
            vi = int(rng.choice(arc['v']))
            if (lt, rt, vi) != last.get(chord) and (lt, rt) != prev:
                break
        last[chord], prev = (lt, rt, vi), (lt, rt)
        m = LHM[chord]
        LH = []
        for (p, d, role, v, a) in LH_T[lt]:
            pitch = _lead_to(nxt, chord) if role == 'lead' else P(m[role] if role in m else m['p'][int(role[1])])
            LH.append((p, d, [pitch], v + arc['vel'] + int(rng.integers(-3, 4)), a))
        RH = []
        for j, p in enumerate(RH_T[rt]):
            vo = [P(x) for x in (VOIC['B7'][1 + j % 2] if chord == 'B7' and p >= 8 else VOIC[chord][vi])]
            d = 2 if rt in ('sparse', 'stop') else 1
            vel = 60 + arc['vel'] + (6 if p % 4 == 2 else 0) + (14 if rt == 'stop' else 0) + int(rng.integers(-3, 4))
            RH.append((p, d, vo, vel, '.', bool(rng.random() < arc['crush'])))
        if chord == 'B7' and lt != 'stop' and rng.random() < 0.6:
            LH = [e for e in LH if e[0] < 12] + [(12 + j, 1, [P(w)], 70 + 4 * j, '.') for j, w in enumerate(WALKUP)]
        if phrase >= 1 and rt != 'stop' and max(p for p, *_ in RH) < 13 and rng.random() < 0.2:
            top = sorted(P(x) for x in VOIC[chord][vi])[-1]
            below = max([q for q in (P(x) for x in SCALE[chord]) if q < top], default=top)
            RH = RH + [(14, 1, [below], 58, '.', False), (15, 1, [top], 64, '.', False)]
        out.append(dict(chord=chord, LH=LH, RH=RH, tags=(lt, rt, vi), phrase=phrase))
    return out


def lick(chord, room16, rng, avoid=None):
    """A short answer built from the motif's gestures, fitted to the chord and the room in a pause (in 16ths).
    Returns (kind, [(pos16, dur16, pitch or [pitches], velocity)])."""
    sc = [P(x) for x in SCALE[chord]]
    vo = sorted(P(x) for x in VOIC[chord][0])
    kinds = ['taps', 'pops', 'slide', 'octave'] + (['run'] if room16 >= 4 else []) + (['answer'] if room16 >= 6 else [])
    kinds = [k for k in kinds if k != avoid] or kinds
    kind = str(rng.choice(kinds))
    if kind == 'taps':
        t = sc[min(3, len(sc) - 1)]
        up = min(sc, key=lambda q: abs(q - (t + 5)))
        return kind, [(0, 1, t, 76), (1, 1, t, 62), (2, 1, t, 70)] + ([(3, 2, up, 82)] if room16 >= 5 else [])
    if kind == 'pops':
        return kind, [(0, 1, vo[-3:-1], 72), (1, 2, vo[-2:], 80)]
    if kind == 'slide':
        lo = [q for q in sc if q < vo[-1]][-2:]
        return kind, ([(0, 1, lo[0], 62), (1, 1, lo[1], 68), (2, 2, vo[-1], 80)] if len(lo) == 2 else [(0, 2, vo[-1], 80)])
    if kind == 'octave':
        return kind, [(0, 1, vo[1], 72), (2, 2, vo[1] + 12, 80)]
    if kind == 'answer':
        return kind, [(0, 3, sc[-2], 80), (3, 1, sc[-3], 70), (4, 2, sc[-4], 76)]
    k = min(6, room16, len(sc))
    s0 = int(rng.integers(0, len(sc) - k + 1))
    return kind, [(j, 1, sc[s0 + j], 64 + 3 * j) for j in range(k)]


def plan(d1, d2):
    """Timing for the day. d1/d2: lengths of the host's two lines (s).
    Returns (n, s1, pause, v_end): groove bars (logo bar included), when the host starts, the pause between the
    lines, and when the host stops. The breath (the turnaround's empty beat 4) lands 0.5-0.9 s after the last word."""
    s1_min = T0 + 0.45                         # the host comes in just after the logo hit
    v_min = s1_min + d1 + 0.4 + d2
    n = max(1, int(np.ceil((v_min + 0.5 - T0 - 3 * BEAT) / BAR - 1e-9)))
    slack = T0 + n * BAR + 3 * BEAT - v_min - 0.5
    pause = 0.4 + min(slack, 0.6)
    slack -= pause - 0.4
    s1 = s1_min + min(slack, 1.4)
    return n, s1, pause, s1 + d1 + pause + d2


def arrange(n, s1, v_end, breaks, seed):
    """All notes of the day's intro as (start_s, end_s, pitch, velocity), plus the fills placed and the groove bars.
    breaks: [(t0, t1)] sentence breaks on the intro timeline, where the right hand may answer the host."""
    rng = np.random.default_rng(seed + 1)
    bars = groove(n, seed)
    notes, fills, used, last_kind = [], [], set(), None

    def add(t_bar, p16, d16, ms, v, a, hand, crush=False, ring=0.0):
        t0 = t_bar + p16 / 4 * BEAT
        dur = d16 / 4 * BEAT * (0.5 if a == '.' else 0.92) + ring
        dt = rng.normal(0, 0.004) + (0.009 if hand == 'LH' and p16 % 2 else 0) + (0.006 if hand == 'RH' and p16 % 2 == 0 else 0)
        top = max(ms)
        for m in ms:
            vel = v + (8 if a == '>' else 0) + ((9 if m == top else -8) if hand == 'RH' and len(ms) > 1 else 0)
            s = max(0.0, t0 + dt)
            notes.append((s, s + dur, int(m), int(np.clip(vel + rng.normal(0, 2.5), 18, 124))))
        if crush:                              # a quick crush into the top note from a half step below
            s = max(0.0, t0 + dt - 0.045)
            notes.append((s, s + 0.04, int(top - 1), int(np.clip(v - 6, 18, 110))))

    for hand in ('RH', 'LH'):
        for e in PICKUP[hand]:
            add(T0 - BAR, *e, hand)
        for e in LOGO[hand]:
            add(T0, *e, hand)
    for k, bar in enumerate(bars, start=1):
        t_bar = T0 + k * BAR
        ans = None
        for bi, (a0, a1) in enumerate(breaks):
            if bi in used or not (s1 + 0.5 < a0 and a1 < v_end - 0.3):
                continue
            s16 = max(0, int(np.ceil((a0 + 0.06 - t_bar) / (BEAT / 4))))
            e16 = min(16, int(np.floor((a1 + 0.25 - t_bar) / (BEAT / 4))))   # a fill may finish softly under the next syllable
            if e16 - s16 >= 3:
                ans = (s16, e16)
                used.add(bi)
                break
        for e in bar['LH']:
            add(t_bar, *e, 'LH')
        for (p16, d16, ms, v, a, crush) in bar['RH']:
            if ans and ans[0] - 1 <= p16 < ans[1]:
                continue
            add(t_bar, p16, d16, ms, v, a, 'RH', crush)
        if ans:
            kind, fig = lick(bar['chord'], ans[1] - ans[0], rng, avoid=last_kind)
            last_kind = kind
            fills.append((round(t_bar + ans[0] / 4 * BEAT, 2), bar['chord'], kind))
            for (q, d, m, v) in fig:
                if ans[0] + q < 16:
                    add(t_bar, ans[0] + q, d, m if isinstance(m, list) else [m], v - 8, '' if d > 1 else '.', 'RH')
    for hand in ('RH', 'LH'):
        for e in TURN[hand]:
            add(T0 + n * BAR, *e, hand)
    t_button = T0 + (n + 1) * BAR
    for hand in ('RH', 'LH'):
        for (p16, d16, ms, v, a) in BUTTON[hand]:
            add(t_button, p16, d16, ms, v, a, hand, ring=BUTTON_RING if p16 >= 8 else 0.0)
    return notes, fills, bars


def end_time(n):
    """When the music has finished: the button's final chord (beat 3) plus its ring."""
    return T0 + (n + 1) * BAR + 2 * BEAT + 2.0


def all_pitches():
    """Every pitch the intro can play (for building the sample bank)."""
    ps = set()
    for part in (PICKUP, LOGO, TURN, BUTTON):
        for hand in part.values():
            for (_, _, ms, _, _) in hand:
                ps.update(ms)
    for chord, m in LHM.items():
        ps.update(P(x) for x in (m['R'], m['O'], m['F'], m['X']) + m['p'])
        for nxt in ROOT:
            ps.add(_lead_to(nxt, chord))
        for vo in VOIC[chord]:
            ps.update(P(x) for x in vo)
            ps.add(max(P(x) for x in vo) - 1)          # crush notes
        ps.update(P(x) for x in SCALE[chord])
        vo0 = sorted(P(x) for x in VOIC[chord][0])
        ps.add(vo0[1] + 12)                            # the 'octave' fill
    ps.update(P(w) for w in WALKUP)
    return sorted(ps)


def long_pitches():
    """Pitches that must ring for seconds (the button's final chord)."""
    return sorted({m for hand in BUTTON.values() for (p, _, ms, _, _) in hand if p >= 8 for m in ms})
