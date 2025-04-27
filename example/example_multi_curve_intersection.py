import numpy as np
import matplotlib.pyplot as plt
from multi-curve import find_intersections_one_to_many

def generate_example_curves():
    """
    生成一条基准曲线和多条待求交的曲线。
    """
    x = np.linspace(0, 10, 100)
    curve_1 = np.stack([x, np.sin(x)], axis=1)

    # 多条曲线：cos(x)、0.5*cos(x)、-0.5*cos(x)
    curves_2 = []
    for scale in [1.0, 0.5, -0.5]:
        curves_2.append(np.stack([x, scale * np.cos(x)], axis=1))
    
    return curve_1, curves_2

def plot_curves_and_intersections(curve_1, curves_2, all_intersections):
    """
    绘制基准曲线、其他曲线和所有交点。
    """
    plt.figure(figsize=(10, 8))
    plt.plot(curve_1[:, 0], curve_1[:, 1], label="Base Curve (sin)", linewidth=2)

    colors = ['blue', 'green', 'orange']
    for idx, curve_2 in enumerate(curves_2):
        plt.plot(curve_2[:, 0], curve_2[:, 1], label=f"Curve 2-{idx+1}", linestyle="--", color=colors[idx])

        if all_intersections[idx]:
            intersections = np.array(all_intersections[idx])
            plt.scatter(intersections[:, 0], intersections[:, 1], color=colors[idx], s=50, edgecolors='k', label=f"Intersections {idx+1}")

    plt.legend()
    plt.title("Batch Curve Intersection Example")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def main():
    curve_1, curves_2 = generate_example_curves()
    all_intersections = batch_intersections(curve_1, curves_2)
    
    for i, intersections in enumerate(all_intersections):
        print(f"Curve 2-{i+1}: Found {len(intersections)} intersections.")
        for idx, point in enumerate(intersections):
            print(f"  Intersection {idx+1}: x = {point[0]:.4f}, y = {point[1]:.4f}")
    
    plot_curves_and_intersections(curve_1, curves_2, all_intersections)

if __name__ == "__main__":
    main()
