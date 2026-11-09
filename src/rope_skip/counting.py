from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional
import numpy as np

@dataclass
class CountState:
    count: int = 0
    phase: str = "ground"
    last_jump_frame: int = -10_000
    baseline: Optional[float] = None
    smooth_y: Optional[float] = None
    history: deque = None

class JumpCounter:
    """Count vertical jump cycles from a stable body anchor trajectory."""
    def __init__(self, threshold_px=22, min_interval_frames=5, cooldown_frames=8, smoothing=.65):
        self.threshold=float(threshold_px)
        self.min_interval=int(min_interval_frames)
        self.cooldown=int(cooldown_frames)
        self.alpha=float(smoothing)
        self.state=CountState(history=deque(maxlen=90))
        self.frame=0

    def update(self, anchor_xy):
        self.frame+=1
        if anchor_xy is None: return self.state.count,False
        y=float(anchor_xy[1])
        s=self.state
        s.smooth_y=y if s.smooth_y is None else self.alpha*y+(1-self.alpha)*s.smooth_y
        s.history.append(s.smooth_y)
        if len(s.history)<5: return s.count,False
        # Image y decreases as the athlete rises. A rolling lower envelope estimates floor stance.
        recent=list(s.history)
        floor_y=float(np.percentile(recent,85))
        lift=floor_y-s.smooth_y
        jumped=False
        if s.phase=="ground" and lift>=self.threshold and self.frame-s.last_jump_frame>=self.min_interval:
            s.phase="air"
        elif s.phase=="air" and lift<=self.threshold*.35 and self.frame-s.last_jump_frame>=self.cooldown:
            s.count+=1; s.last_jump_frame=self.frame; s.phase="ground"; jumped=True
        return s.count,jumped
