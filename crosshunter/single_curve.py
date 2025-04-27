import numpy as np
from scipy.interpolate import interp1d

def segment_curve(curve):
    """
    Segments a curve into monotonic pieces.
    
    Args:
        curve (np.ndarray): An (N, 2) array of (x, y) points.
        
    Returns:
        list: A list of curve segments as (M, 2) arrays.
    """
    x = curve[:, 0]
    dx = np.diff(x)
    split_indices = np.where(np.diff(np.sign(dx)) != 0)[0] + 1
    segments = np.split(curve, split_indices)
    return segments

def find_intersections(curve_1, curve_2, err=1e-2):
    """
    Finds intersection points between two curves.
    
    Args:
        curve_1 (np.ndarray): First curve (N, 2).
        curve_2 (np.ndarray): Second curve (M, 2).
        err (float): Precision threshold for bisection.
        
    Returns:
        list of tuples: Intersection points (x, y).
    """
    curve_1_segments = segment_curve(curve_1)
    curve_2_segments = segment_curve(curve_2)
    intersections = []

    for seg_1 in curve_1_segments:
        x_1 = seg_1[:, 0]
        y_1 = seg_1[:, 1]
        seg_x_min, seg_x_max = min(x_1[0], x_1[-1]), max(x_1[0], x_1[-1])
        interp_1 = interp1d(x_1, y_1, kind='linear', bounds_error=False, fill_value=np.nan)

        for seg_2 in curve_2_segments:
            x_2 = seg_2[:, 0]
            y_2 = seg_2[:, 1]
            seg_x_min_2, seg_x_max_2 = min(x_2[0], x_2[-1]), max(x_2[0], x_2[-1])

            if seg_x_min_2 > seg_x_max or seg_x_max_2 < seg_x_min:
                continue

            interp_2 = interp1d(x_2, y_2, kind='linear', bounds_error=False, fill_value=np.nan)
            x_vals = np.linspace(max(seg_x_min, seg_x_min_2), min(seg_x_max, seg_x_max_2), 100)
            y_diff = interp_1(x_vals) - interp_2(x_vals)

            sign_change_indices = np.where(np.diff(np.sign(y_diff)) != 0)[0]

            for idx in sign_change_indices:
                left_x, right_x = x_vals[idx], x_vals[idx + 1]
                flag = y_diff[idx] > 0

                try:
                    while right_x - left_x > err:
                        mid_x = (left_x + right_x) / 2
                        mid_y = interp_1(mid_x) - interp_2(mid_x)
                        if np.abs(mid_y) < err:
                            intersections.append((mid_x, interp_1(mid_x)))
                            break
                        elif mid_y > 0:
                            if flag:
                                left_x = mid_x
                            else:
                                right_x = mid_x
                        else:
                            if flag:
                                right_x = mid_x
                            else:
                                left_x = mid_x
                except Exception:
                    continue

    return intersections
