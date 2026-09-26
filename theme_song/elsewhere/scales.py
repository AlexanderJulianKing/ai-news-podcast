"""A Dorian scale steps: move a note by scale degrees (used to fit the motif to each chord)."""

SCALE = [9, 11, 0, 2, 4, 6, 7]          # A Dorian: A B C D E F# G

def _degree(pc):
    """Scale index of a pitch class (F natural counts as the F# slot, lowered)."""
    if pc in SCALE:
        return SCALE.index(pc), 0
    if pc == 5:
        return SCALE.index(6), -1
    lower = max((s for s in SCALE if s < pc), default=None)
    return SCALE.index(lower if lower is not None else SCALE[-1]), 1


def _to_steps(m):
    """Absolute scale position counted from A (so C-G belong to the octave that starts on the A below them)."""
    idx, alt = _degree(m % 12)
    base = m - alt
    return ((base - 9) // 12) * 7 + idx, alt


def _from_steps(steps, alt):
    octave, idx = divmod(steps, 7)
    return 9 + octave * 12 + (SCALE[idx] - 9) % 12 + alt


def shift_diatonic(m, steps):
    s, alt = _to_steps(m)
    return _from_steps(s + steps, alt)


def invert_diatonic(m, axis=76):
    s, alt = _to_steps(m)
    a, _ = _to_steps(axis)
    return _from_steps(2 * a - s, -alt)


