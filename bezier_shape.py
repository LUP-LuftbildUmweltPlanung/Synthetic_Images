import numpy as np
import warnings
from scipy.special import binom
from utils import fill_contours


def bernstein(n, k, t):
    return binom(n, k) * t ** k * (1. - t) ** (n - k)


def bezier(points, num=200):
    n = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(n):
        curve += np.outer(bernstein(n - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, detail, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", detail)  # adjusts how many points each segment contains --> prevents holes
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points()
        self.curve = bezier(self.p, self.numpoints)

    def calc_intermediate_points(self):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])


def get_curve(points, detail, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], detail, **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, detail, rad=0.2, edgy=0.0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    angle = (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang1 = angle
    ang2 = np.roll(angle, 1)
    angle = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    angle = np.append(angle, [angle[0]])
    a = np.append(a, np.atleast_2d(angle).T, axis=1)
    s, c = get_curve(a, detail, r=rad, method="var")
    x_coord, y_coord = c.T
    return x_coord, y_coord, a


def get_random_points(n=5, scale=0.8, min_dist=None, rec=0):
    """ create n random points in the unit square, which are *min_dist*
    apart, then scale them."""
    min_dist = min_dist or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= min_dist) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, min_dist=min_dist, rec=rec + 1)


def array_from_coordinates(x_coord, y_coord, size=100):
    bezier_array = np.zeros((size, size))
    x_coord = np.interp(x_coord, (x_coord.min(), x_coord.max()), (0, 1))
    y_coord = np.interp(y_coord, (y_coord.min(), y_coord.max()), (0, 1))

    x_coord = np.rint(x_coord * (size - 1))
    y_coord = np.rint(y_coord * (size - 1))
    for x_coord, y_coord in zip(x_coord, y_coord):
        bezier_array[int(x_coord), int(y_coord)] = 1.0

    return bezier_array


def random_shape(size, shape_type, radius=0.4):
    """Returns an array containing a random, natural looking shape based on random points connected by bezier-curves.

                Keyword arguments (same as 'place_tree'):
                size -- height and width of the returned square array
                shape_type -- defines shape-complexity (use: 'close', 'single_tree', 'cluster' with rising complexity)
                radius -- effects radius of curve (between 0 and 1, default 0.4)
    """
    if shape_type == 'cluster':
        points = 7
    elif shape_type == 'single_tree':
        points = 5
    elif shape_type == 'close':
        points = 3
    else:
        warnings.warn("Warning: Shape type not recognised. Type 'close' will be used. "
                      "Please use either 'close' for smooth shapes, single_tree for less smooth, "
                      "or 'cluster' for complicated shapes.")
        points = 3
    array = get_random_points(n=points, scale=1)
    x_coord, y_coord, _ = get_bezier_curve(array, detail=size*10, rad=radius, edgy=0.0)

    bezier_img = array_from_coordinates(x_coord, y_coord, size=size)

    return fill_contours(bezier_img)
