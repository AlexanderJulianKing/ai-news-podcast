# Elsewhere

The music bed under the news roundup. It was composed with Alex on 2026-09-26: A minor/Dorian at 94 bpm, with a climbing synth pulse, a piano counter-figure, a three-note piano motif (E-A-B, a question), long low synth notes, and a one-bar shaker head. The design notes, every draft, and the demos live outside the repo in `~/Music/GarageBand/Newscaster theme/roundup/Elsewhere/`.

- `score.py` holds the notes: head, body (sections that keep changing) and ending.
- `build_elsewhere.py` renders the score once, on a Mac with GarageBand/Logic's instruments, into `elsewhere_assets.npz`. The file is about 128 MB of Apple sample content, so it is never committed. Copy it to the Pi:

      python3 theme_song/elsewhere/build_elsewhere.py
      scp theme_song/elsewhere/elsewhere_assets.npz alex@raspberrypi.tail470879.ts.net:/home/alex/ai-news-podcast/theme_song/elsewhere/

- `newscaster/audio/roundup_bed.py` fits the assets to each day's roundup on the Pi. The music stretches to the reading, the ending starts 0.6 s after the last word, and the roundup is then followed by the outro. It is switched on by `ROUNDUP_BED_ENABLED` in `newscaster/config.py`. If anything fails, the show uses the plain roundup.
- `sampler.py`, `render_part.swift`, `piano.py` and `shaker.py` render the instruments. Apple's sampler plays the synths; the Steinway and the shaker are decoded directly, because Apple's sampler can't load them.
