# Off the Wire (intro music)

The show's intro music since 2026-09-26, composed by Claude on 2026-09-25. The notes live in
`newscaster/audio/off_the_wire_score.py` and the renderer in `newscaster/audio/off_the_wire.py`.

**The shape:**

- **The logo.** Three taps on B into an E minor hit, about a second long. The host starts right after it.
- **Under the reading.** A funk-piano groove at a steady level, about 9 dB under the voice, cycling Em9–A–C–D. It changes every day, since the generator is seeded by the date. It grows busier as the reading goes on, and it answers the host's sentence breaks with short fills.
- **The ending.** A turnaround under the last words, a breath just after them, then a one-bar button. The music ends about 4 s after the host.

**The piano** is GarageBand's Steinway Grand Piano. `build_steinway_bank.py` renders the notes the intro needs into `steinway_bank_48k.npz`. Run it on a Mac with GarageBand installed, then copy the file to the Pi. It is Apple sample content: do not commit it (it is in `.gitignore`).

**The switch.** `OFF_THE_WIRE_INTRO_ENABLED` in `newscaster/config.py`. If the bank is missing or anything fails, the classic theme is used.

The full concert piece, its score and the design history are in `theme_song/sketches/2026-09-25 Off the Wire/` on the Mac. That folder is not in the repo.
