import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_Hough(lines, rho_resolution, theta_resolution, rho_max, threed=True):
    if threed:
        ax = plt.axes(projection="3d", proj_type='ortho')
    else:
        ax = plt.axes()
    rho_n = int(round(rho_max / rho_resolution)) + 1
    theta_n = int(round(np.pi / theta_resolution / 2)) + 1
    print(rho_n, theta_n, type(theta_n))
    A = np.zeros((rho_n, theta_n), dtype=np.uint32)
    for ((rho, theta, votes),) in lines:
        rho_ndx = int(round((rho_max + rho) / rho_resolution / 2))
        rho_ndx, theta_ndx = int(round((rho_max + rho) / rho_resolution / 2)), int(round(theta / theta_resolution / 2))
        A[rho_ndx, theta_ndx] = votes
    A = (A / A.max() * 255).astype(np.uint8)
    if threed:
        X = np.arange(-rho_max, rho_max + rho_resolution * 2, rho_resolution * 2)
        Y = np.arange(0, np.pi / 2 + theta_resolution, theta_resolution)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, A.T)
    else:
        plt.imshow(A.T)


def plot_line_in_img(line, img):
    if isinstance(line[0], np.ndarray):  # a line as represented by cv2.HoughLines
        rho, theta = line[0][:2]
    else:  # a line as represented by selected_lines
        rho, theta = line
    r = 3000
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1 = int(x0 + r * b)
    x2 = int(x0 - r * b)
    y1 = int(y0 - r * a)
    y2 = int(y0 + r * a)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def find_intercepts(lines):
    intercepts = []
    for l1, l2 in itertools.combinations(lines, 2):
        # intercept point: rho, theta = line
        if np.abs(l1[1]) > 0 and np.abs(l2[1]) > 0 and l1[1] != l2[1]:
            x = (l1[0] / np.sin(l1[1]) - l2[0] / np.sin(l2[1])) / (1 / np.tan(l1[1]) - 1 / np.tan(l2[1]))
            y = (l1[0] - x * np.cos(l1[1])) / np.sin(l1[1])
            intercepts.append(np.array([x, y]))
    return intercepts


def find_cluster(points, dmax):
    maxPoints = 0
    center = np.array([0., 0.])
    for p in points:
        nPoints = 0
        sum_q = np.array([0., 0.])
        for q in points:
            if np.linalg.norm(p - q) <= dmax:
                nPoints += 1
                sum_q += q
        if nPoints > maxPoints:
            maxPoints = nPoints
            center = sum_q / nPoints
    return center
