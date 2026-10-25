"""从连续压力记录量化限压持续时间；所有 N 使用同一 20 MPa 阈值。"""
import numpy as np


def pressure_exposure(trace, threshold=20.):
    """Linear crossing interpolation on the dense trace, no curve shifts.

    Duplicate interval boundaries contribute zero duration. The first/last
    crossing timestamps describe the full exposure envelope; disjoint late
    peaks remain visible in the total duration rather than being hidden.
    """
    trace=np.asarray(trace,dtype=float)
    intervals=[]
    for left,right,pl,pr in zip(trace[:-1,0],trace[1:,0],trace[:-1,2],trace[1:,2]):
        if right<=left or max(pl,pr)<threshold:continue
        if min(pl,pr)>=threshold:
            intervals.append((left,right))
        else:
            crossing=left+(threshold-pl)*(right-left)/(pr-pl)
            intervals.append((crossing,right) if pr>=threshold else (left,crossing))
    return dict(p2_above_20mpa_ms=float(sum(b-a for a,b in intervals)*1000),
                p2_first_20mpa_s=None if not intervals else float(intervals[0][0]),
                p2_last_20mpa_s=None if not intervals else float(intervals[-1][1]))


def pressure_dwell(trace,threshold=20.):
    return pressure_exposure(trace,threshold)['p2_above_20mpa_ms']
