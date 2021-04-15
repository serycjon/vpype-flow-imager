# Copyright (C) 2021 Jonas Serych <jonas@sery.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from collections import deque
import numpy as np
import cv2
from opensimplex import OpenSimplex
import tqdm
import hnswlib
import contextlib

import click
import vpype as vp

import logging
logger = logging.getLogger(__name__)

eps = 1e-10


@click.command("flow_img")
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "-nc",
    "--noise_coeff",
    default=0.001,
    type=float,
    help=("Simplex noise coordinate multiplier. "
          "The smaller, the smoother the flow field."),
)
@click.option(
    "-nf",
    "--n_fields",
    default=1,
    type=int,
    help="Number of rotated copies of the flow field",
)
@click.option(
    "-ms",
    "--min_sep",
    default=0.8,
    type=float,
    help="Minimum flowline separation",
)
@click.option(
    "-Ms",
    "--max_sep",
    default=10,
    type=float,
    help="Maximum flowline separation",
)
@click.option(
    "-ml",
    "--min_length",
    default=0,
    type=float,
    help="Minimum flowline length",
)
@click.option(
    "-Ml",
    "--max_length",
    default=40,
    type=float,
    help="Maximum flowline length",
)
@click.option(
    "--max_size",
    default=800,
    type=int,
    help="The input image will be rescaled to have sides at most max_size px",
)
@click.option(
    "-s", "--seed", type=int, help="PRNG seed (overriding vpype seed)"
)
@click.option(
    "-fs", "--flow_seed", type=int,
    help="Flow field PRNG seed (overriding the main `--seed`)"
)
@vp.generator
def vpype_flow_imager(filename, noise_coeff, n_fields,
                      min_sep, max_sep,
                      min_length, max_length, max_size,
                      seed, flow_seed):
    """
    Generate flowline representation from an image.

    The generated flowlines are in the coordinates of the input image,
    resized to have dimensions at most `--max_size` pixels.
    """
    gray_img = cv2.imread(filename, 0)
    with tmp_np_seed(seed):
        numpy_paths = draw_image(gray_img, mult=noise_coeff, n_fields=n_fields,
                                 min_sep=min_sep, max_sep=max_sep,
                                 min_length=min_length, max_length=max_length,
                                 max_img_size=max_size, flow_seed=flow_seed)

    lc = vp.LineCollection()
    for path in numpy_paths:
        lc.append(path[:, 0] + path[:, 1] * 1.j)
    return lc


vpype_flow_imager.help_group = "Plugins"


def norm_2vec(x):
    return np.sqrt(x[0]**2 + x[1]**2)


def gen_flow_field(H, W, x_mult=1, y_mult=None):
    logger.info('Generating flow field')
    if y_mult is None:
        y_mult = x_mult
    x_noise = OpenSimplex(np.random.randint(9393931))
    y_noise = OpenSimplex(np.random.randint(9393931))
    field = np.zeros((H, W, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            x_val = x_noise.noise2d(x=x_mult * x, y=x_mult * y)
            y_val = y_noise.noise2d(x=y_mult * x, y=y_mult * y)
            norm = np.sqrt(x_val ** 2 + y_val ** 2)
            if norm > eps:
                x_val /= norm
                y_val /= norm
            else:
                x_val, y_val = 1, 0
            field[y, x, :] = (x_val, y_val)

    return field


def draw_image(gray_img, mult, max_img_size=800, n_fields=1,
               min_sep=0.8, max_sep=10,
               min_length=0, max_length=40,
               flow_seed=None):
    gray = resize_to_max(gray_img, max_img_size)
    H, W = gray.shape

    with tmp_np_seed(flow_seed):
        field = gen_flow_field(H, W, x_mult=mult)
    fields = [VectorField(field)]
    if n_fields > 1:
        angles = np.linspace(0, 360, n_fields + 1)
        for angle in angles:
            fields.append(VectorField(rotate_field(field, angle)))

    def d_sep_fn(pos):
        x, y = fit_inside(np.round(pos), gray)
        val = gray[int(y), int(x)] / 255
        val = val**2
        return remap(val, 0, 1, min_sep, max_sep)

    paths = draw_fields_uniform(fields, d_sep_fn,
                                seedpoints_per_path=40,
                                guide=gray,
                                min_length=min_length, max_length=max_length)
    return paths


class VectorField():
    def __init__(self, field_array):
        self.field = field_array
        self.shape = self.field.shape

    def __getitem__(self, pos):
        ''' pos should be (x, y) '''
        round_pos = np.round(pos[:2]).astype(np.int64)
        round_pos = fit_inside(round_pos, self.field)

        return self.field[round_pos[1], round_pos[0], :]


def rotate_field(field, degrees):
    s, c = np.sin(np.radians(degrees)), np.cos(np.radians(degrees))
    R = np.array([[c, -s],
                  [s, c]])
    return np.matmul(R, field.reshape(-1, 2).T).T.reshape(field.shape)


def fit_inside(xy, img):
    return np.clip(xy,
                   np.array([0, 0], xy.dtype),
                   np.array([img.shape[1] - 1, img.shape[0] - 1], xy.dtype))


def remap(x, src_min, src_max, dst_min, dst_max):
    x_01 = (x - src_min) / float(src_max - src_min)
    x_dst = x_01 * (dst_max - dst_min) + dst_min

    return x_dst


def draw_fields_uniform(fields, d_sep_fn, d_test_fn=None,
                        seedpoints_per_path=10,
                        guide=None,
                        min_length=0, max_length=20):
    logger.info('Drawing flowlines')
    if d_test_fn is None:
        def d_test_fn(*args, **kwargs):
            return d_sep_fn(*args, **kwargs) / 2

    H, W = fields[0].shape[:2]

    def should_stop(new_pos, searcher, path, d_sep_fn):
        if path.line_length < min_length:
            return False

        if not inside(np.round(new_pos), H, W):
            return True
        if searcher is not None:
            point = new_pos.copy()
            nearest = searcher.get_nearest(point)
            dist, pt = nearest
            if dist < d_sep_fn(new_pos):
                return True

        if path.line_length > max_length:
            return True

        # look for loops
        # candidate = np.round(new_pos).astype(np.int64).reshape(1, 2)
        # for pt in reversed(path):
        #     if np.all(candidate == np.round(pt).astype(np.int64)):
        #         return True
        return False

    searcher = HNSWSearcher([np.array([-10, -10])],
                            max_elements=64000)
    paths = []
    seed_pos = np.array((W / 2, H / 2))
    seedpoints = [seed_pos]
    seedpoints = deque(seedpoints)
    pbar = tqdm.tqdm()
    try:
        while True:
            # try to find a suitable seedpoint in the queue
            try:
                while True:
                    seed_pos = seedpoints.pop()
                    if not inside(np.round(seed_pos), H, W):
                        continue

                    dist, _ = searcher.get_nearest(seed_pos)
                    if dist < d_sep_fn(seed_pos):
                        continue

                    break
            except IndexError:
                # no more seedpoints
                break

            start_field = np.random.randint(len(fields))

            def select_field(path_len, direction):
                same_field_len = 10

                idx = int(direction * path_len // same_field_len) + start_field
                idx = idx % len(fields)
                return fields[idx]

            class MemorySelector():
                def __init__(self, fields):
                    self.same_field_len = 10
                    self.cur_len = 0
                    self.idx = np.random.randint(len(fields))
                    self.fields = fields

                def select_field(self, path_len, direction):
                    if (path_len - self.cur_len) > self.same_field_len:
                        self.cur_len = path_len
                        idx_delta = np.random.randint(-1, 1 + 1)
                        self.idx = (self.idx + idx_delta) % len(self.fields)

                    return self.fields[self.idx]

            selector = MemorySelector(fields)

            path = compute_streamline(selector.select_field, seed_pos,
                                      searcher,
                                      d_test_fn, d_sep_fn,
                                      should_stop_fn=should_stop)
            if len(path) <= 2:
                # nothing found
                # logging.debug('streamline ended immediately')
                continue

            for pt in path:
                searcher.add_point(pt)
            paths.append(np.array(path))
            # if len(paths) > 75:  # for debugging purposes
            #     break

            new_seedpoints = generate_seedpoints(path, d_sep_fn,
                                                 seedpoints_per_path)
            order = np.arange(len(new_seedpoints))
            np.random.shuffle(order)
            seedpoints.extend([new_seedpoints[i] for i in order])
            pbar.update(1)
    except KeyboardInterrupt:
        pass

    pbar.close()
    return paths


def inside(xy_pt, H, W):
    return (xy_pt[0] >= 0 and
            xy_pt[1] >= 0 and
            xy_pt[0] < W and
            xy_pt[1] < H)


def compute_streamline(field_getter, seed_pos, searcher, d_test_fn, d_sep_fn,
                       should_stop_fn):
    direction_sign = 1  # first go with the field
    pos = seed_pos.copy()
    paths = []
    path = LinePath()
    path.append(pos.copy())
    stop_tracking = False
    self_searcher = HNSWSearcher([(-20, -20)])
    while True:
        field = field_getter(path.line_length, direction_sign)
        rk_force = runge_kutta(field, pos, d_test_fn(pos)) * direction_sign
        new_pos = pos + d_test_fn(pos) * rk_force

        # test validity
        if should_stop_fn(new_pos, searcher, path, d_sep_fn):
            stop_tracking = True

        # prevent soft looping
        nearest_dist, _ = self_searcher.get_nearest(new_pos)
        if nearest_dist < d_sep_fn(pos):
            stop_tracking = True
        lookback = 15
        if len(path) >= 2 * lookback:
            self_searcher.add_point(path[-lookback])

        # fallback
        if len(path) >= 600:
            stop_tracking = True

        if not stop_tracking:
            path.append(new_pos.copy())

        if stop_tracking:
            paths.append(path.data)
            if direction_sign == 1:
                # go to the other side from the seed
                direction_sign = -1
                pos = seed_pos.copy()
                path = LinePath()
                path.append(pos.copy())
                # self_searcher = searcher([(-20, -20)])
                stop_tracking = False
            else:
                # both directions finished
                break
        else:
            pos = new_pos
    singleline = list(reversed(paths[1]))
    singleline.extend(paths[0])

    return singleline


def generate_seedpoints(path, d_sep_fn, N_seedpoints=10):
    # go along the path and create points perpendicular in d_sep distance
    seeds = []
    seedpoint_positions = np.linspace(0, len(path) - 1, N_seedpoints)
    seedpoint_ids = np.unique(np.round(seedpoint_positions)).tolist()

    cur_xy = path[0]
    direction = path[1] - path[0]
    direction /= max(norm_2vec(direction), eps)
    normal = np.array((direction[1], -direction[0]))
    margin = 1.1
    seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * normal)
    seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * normal)

    for i in range(1, len(path)):
        if i not in seedpoint_ids:
            continue
        last_xy = cur_xy.copy()
        cur_xy = path[i]
        direction = cur_xy - last_xy
        direction /= max(norm_2vec(direction), eps)
        normal = np.array((direction[1], -direction[0]))
        seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * normal)
        seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * normal)

    return seeds


def runge_kutta(field, pos, h):
    k1 = field[pos]

    k2_pos = pos + (h / 2) * k1
    k2 = field[k2_pos]

    k3_pos = pos + (h / 2) * k2
    k3 = field[k3_pos]

    k4_pos = pos + h * k3
    k4 = field[k4_pos]

    # Runge-Kutta for the win
    rk = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return rk


def resize_to_max(img, max_sz):
    H_scale = max_sz / img.shape[0]
    W_scale = max_sz / img.shape[1]

    scale = min(H_scale, W_scale)
    return cv2.resize(img, None, fx=scale, fy=scale)


class HNSWSearcher:
    def __init__(self, points, max_elements=1000):
        self.index = hnswlib.Index(space='l2', dim=2)
        self.max_elements = max_elements
        self.index.init_index(max_elements=self.max_elements,
                              ef_construction=200, M=16)
        self.index.set_ef(50)
        self.index.set_num_threads(4)
        for point in points:
            self.add_point(point)

    def add_point(self, point):
        if self.index.element_count == self.max_elements:
            self.resize_index()
        to_insert = np.array(point).reshape(1, 2)
        self.index.add_items(to_insert)

    def resize_index(self):
        self.max_elements = 2 * self.max_elements
        logger.info(f'Resizing searcher index to {self.max_elements}')
        self.index.resize_index(self.max_elements)
        logger.info('after resize:')
        logger.info(f"self.index.max_elements: {self.index.max_elements}")
        logger.info(f"self.index.element_count: {self.index.element_count}")

    def get_nearest(self, query):
        to_query = np.array(query).reshape(1, 2)
        labels, distances_sq = self.index.knn_query(to_query, k=1)
        distances = np.sqrt(distances_sq)
        return distances, labels


class LinePath:
    ''' wrapper around list of coordinates, that keeps current path length '''

    def __init__(self):
        self.data = []
        self.line_length = 0

    def append(self, point):
        self.data.append(point)
        if len(self.data) > 1:
            self.line_length += norm_2vec(self.data[-2] - self.data[-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


@contextlib.contextmanager
def tmp_np_seed(seed):
    if seed is None:
        yield
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)
