import numpy as np
import matplotlib.pyplot as plt
from curve_intersections import find_intersections

def generate_example_curves():
    """
    生成两条示例曲线。
    """
    x = np.linspace(0, 10, 100)
    curve_1 = np.stack([x, np.sin(x)], axis=1)
    curve_2 = np.stack([x, 0.5 * np.cos(x)], axis=1)
    return curve_1, curve_2

def plot_curves_and_intersections(curve_1, curve_2, intersections):
    """
    绘制曲线和交点。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(curve_1[:, 0], curve_1[:, 1], label="Curve 1 (sin)")
    plt.plot(curve_2[:, 0], curve_2[:, 1], label="Curve 2 (cos)")
    
    if intersections:
        intersections = np.array(intersections)
        plt.scatter(intersections[:, 0], intersections[:, 1], color='red', label="Intersections", zorder=5)
    
    plt.legend()
    plt.title("Single Curve Intersection Example")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def main():
    curve_1, curve_2 = generate_example_curves()
    intersections = find_intersections(curve_1, curve_2)
    print(f"Found {len(intersections)} intersections.")
    for idx, point in enumerate(intersections):
        print(f"Intersection {idx+1}: x = {point[0]:.4f}, y = {point[1]:.4f}")
    plot_curves_and_intersections(curve_1, curve_2, intersections)

if __name__ == "__main__":
    main()
