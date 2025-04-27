import numpy as np
from scipy.interpolate import interp1d

def prepare_curve_segments(curve):
    """
    Prepares a curve for fast intersection checking by segmenting and interpolating.
    
    Args:
        curve (np.ndarray): An (N, 2) array of (x, y) points.
        
    Returns:
        list: List of (segment, interpolator) tuples.
    """
    x = curve[:, 0]
    dx = np.diff(x)
    split_indices = np.where(np.diff(np.sign(dx)) != 0)[0] + 1
    segments = np.split(curve, split_indices)

    segments_with_interp = []
    for seg in segments:
        x_seg = seg[:, 0]
        y_seg = seg[:, 1]
        interp_func = interp1d(x_seg, y_seg, kind='linear', bounds_error=False, fill_value=np.nan)
        segments_with_interp.append((seg, interp_func))
    
    return segments_with_interp

def find_intersections_with_prepared_curve(prepared_curve_1, curve_2, err=1e-2):
    """
    Finds intersections between a prepared curve and a new curve.
    
    Args:
        prepared_curve_1 (list): Output from prepare_curve_segments.
        curve_2 (np.ndarray): New curve (N, 2).
        err (float): Precision for bisection.
        
    Returns:
        list of tuples: Intersection points (x, y).
    """
    intersections = []

    x = curve_2[:, 0]
    dx = np.diff(x)
    split_indices = np.where(np.diff(np.sign(dx)) != 0)[0] + 1
    curve_2_segments = np.split(curve_2, split_indices)

    for seg_2 in curve_2_segments:
        x_2 = seg_2[:, 0]
        y_2 = seg_2[:, 1]
        interp_2 = interp1d(x_2, y_2, kind='linear', bounds_error=False, fill_value=np.nan)

        for seg_1, interp_1 in prepared_curve_1:
            x_1 = seg_1[:, 0]
            seg_x_min, seg_x_max = min(x_1[0], x_1[-1]), max(x_1[0], x_1[-1])
            common_x_min = max(seg_x_min, np.min(x_2))
            common_x_max = min(seg_x_max, np.max(x_2))

            if common_x_min >= common_x_max:
                continue

            x_vals = np.linspace(common_x_min, common_x_max, 100)
            y1_vals = interp_1(x_vals)
            y2_vals = interp_2(x_vals)
            y_diff = y1_vals - y2_vals

            if np.any(np.isnan(y_diff)):
                continue

            sign_change_indices = np.where(np.diff(np.sign(y_diff)) != 0)[0]

            for idx in sign_change_indices:
                left_x, right_x = x_vals[idx], x_vals[idx + 1]
                flag = y_diff[idx] > 0

                try:
                    for _ in range(50):
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

def find_intersections_one_to_many(curve_1, curves_2, err=1e-2):
    """
    Finds intersections between a fixed curve and multiple target curves efficiently.
    
    Args:
        curve_1 (np.ndarray): Fixed curve (N, 2).
        curves_2 (list of np.ndarray): List of target curves.
        err (float): Bisection error threshold.
        
    Returns:
        list of list of tuples: List of intersection points per curve.
    """
    prepared_curve_1 = prepare_curve_segments(curve_1)
    results = []
    for curve_2 in curves_2:
        intersections = find_intersections_with_prepared_curve(prepared_curve_1, curve_2, err=err)
        results.append(intersections)
    return results
