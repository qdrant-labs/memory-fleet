"""MPS memory-leak bisect (2026-07-01, this laptop, torch 2.12.1, macOS 25.4).

The live-webcam soak failed its memory check: RSS grew linearly ~80 MB/min.
Bisect (live webcam, current-RSS via ps, NOT ru_maxrss — that high-water mark
let early allocation spikes mask the creep in footage soaks):

  camera-only            +0 MB   (cv2/AVFoundation clean)
  camera + track()     +204 MB / 2.4 min
  predict, no tracker  +191 MB / 2.5 min   <- tracker exonerated; STracks flat
  torch.mps allocator   flat 145 MB        <- creep is OUTSIDE torch accounting
  + empty_cache()/30f   +70 MB / 2 min     (no fix)
  + sync+empty+gc/30f  worse               (no fix)
  objc.autorelease_pool around predict:
      wrapped           +18 MB, flattening  <- THE FIX
      bare (same proc) +131 MB, climbing

Cause: torch-MPS autoreleases Metal objects per inference; a pure Python loop
never drains the main autorelease pool. Fix shipped in perception/detector.py:
every model call runs inside objc.autorelease_pool() (pyobjc-core, darwin-only
dep). Rerun this by pointing scripts/soak.py at a live camera.
"""

print(__doc__)
