"""
from https://github.com/Vectorized/Python-KD-Tree (CC0 license), with added tree rebalancing

A super short KD-Tree for points...
so concise that you can copypasta into your homework without arousing suspicion.


Usage:
1. Use make_kd_tree to create the kd
2. You can then use `get_knn` for k nearest neighbors or
   `get_nearest` for the nearest neighbor

points are be a list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
"""


# Makes the KD-Tree for fast lookup
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]


# Adds a point to the kd-tree
def add_point(kd_node, point, dim, i=0):
    if kd_node is not None:
        dx = kd_node[2][i] - point[i]
        i = (i + 1) % dim
        for j, c in ((0, dx >= 0), (1, dx < 0)):
            if c and kd_node[j] is None:
                kd_node[j] = [None, None, point]
            elif c:
                add_point(kd_node[j], point, dim, i)


# k nearest neighbors
def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]


def rebalance(kd_tree, dim):
    from collections import deque
    elements = []
    queue = deque()
    queue.append((kd_tree, 0))
    max_h = 0

    while len(queue) > 0:
        cur, h = queue.pop()
        if h > max_h:
            max_h = h

        if cur[0] is not None:
            queue.append((cur[0], h + 1))
        if cur[1] is not None:
            queue.append((cur[1], h + 1))

        if cur[2] is not None:
            elements.append(cur[2])

    # print(f"max_h: {max_h}")
    rebalanced = make_kd_tree(elements, dim=dim)
    return rebalanced


# For the closest neighbor
def get_nearest(kd_node, point, dim, dist_func, return_distances=True, i=0, best=None):
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if not best:
            best = [dist, kd_node[2]]
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
            get_nearest(kd_node[b], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]


"""
If you want to attach other properties to your points,
you can use this class or subclass it.

Usage:

point = PointContainer([1,2,3])
point.label = True
print point         # [1,2,3]
print point.label   # True
"""


class PointContainer(list):
    def __new__(self, value, name=None, values=None):
        s = super(PointContainer, self).__new__(self, value)
        return s


if __name__ == '__main__':
    """
    Below is all the testing code
    """

    import random
    import cProfile

    def puts(xs):
        for x in xs:
            print(x)

    def get_knn_naive(points, point, k, dist_func, return_distances=True):
        neighbors = []
        for i, pp in enumerate(points):
            dist = dist_func(point, pp)
            neighbors.append((dist, pp))
        neighbors = sorted(neighbors)[:k]
        return neighbors if return_distances else [n[1] for n in neighbors]

    dim = 3

    def rand_point(dim):
        return [random.uniform(-1, 1) for d in range(dim)]

    def dist_sq(a, b, dim):
        return sum((a[i] - b[i]) ** 2 for i in range(dim))

    def dist_sq_dim(a, b):
        return dist_sq(a, b, dim)

    points = [PointContainer(rand_point(dim)) for x in range(10)]
    additional_points = [PointContainer(rand_point(dim)) for x in range(3)]
    kd_tree = make_kd_tree(points, dim)
    for point in additional_points:
        add_point(kd_tree, point, dim)

    rebalanced = rebalance(kd_tree, dim)
    rebalanced = rebalance(rebalanced, dim)
    sys.exit(1)

    points = [PointContainer(rand_point(dim)) for x in range(10000)]
    additional_points = [PointContainer(rand_point(dim)) for x in range(50)]
    # points = [rand_point(dim) for x in range(5000)]
    test = [rand_point(dim) for x in range(100)]
    result1 = []
    result2 = []

    def bench1():
        kd_tree = make_kd_tree(points, dim)
        for point in additional_points:
            add_point(kd_tree, point, dim)
        result1.append(tuple(get_knn(kd_tree, [0] * dim, 8, dim, dist_sq_dim)))
        for t in test:
            result1.append(tuple(get_knn(kd_tree, t, 8, dim, dist_sq_dim)))

    def bench2():
        all_points = points + additional_points
        result2.append(tuple(get_knn_naive(all_points, [0] * dim, 8, dist_sq_dim)))
        for t in test:
            result2.append(tuple(get_knn_naive(all_points, t, 8, dist_sq_dim)))

    cProfile.run("bench1()")
    cProfile.run("bench2()")

    puts(result1[0])
    print("")
    puts(result2[0])
    print("")

    print("Is the result same as naive version?: {}".format(result1 == result2))

    print("")
    kd_tree = make_kd_tree(points, dim)

    print(get_nearest(kd_tree, [0] * dim, dim, dist_sq_dim))

    """
    You can also define the distance function inline, like:

    print get_nearest(kd_tree, [0] * dim, dim, lambda a,b: dist_sq(a, b, dim))
    print get_nearest(kd_tree, [0] * dim, dim, lambda a,b: sum((a[i] - b[i]) ** 2 for i in range(dim)))
    """
