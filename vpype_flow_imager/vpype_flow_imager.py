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
try:
    import hnswlib
except ImportError:
    hnswlib = None

import contextlib
from .kdtree import KDTSearcher
from PIL import Image

import click
import vpype as vp
import vpype_cli

import logging
logger = logging.getLogger(__name__)

eps = 1e-10


@click.command("flow_img", context_settings={'show_default': True})
@click.argument("filename", type=vpype_cli.PathType(exists=True))
@click.option(
    "-nc",
    "--noise_coeff",
    default=0.001,
    type=vpype_cli.FloatType(),
    help=("Simplex noise coordinate multiplier. "
          "The smaller, the smoother the flow field."),
)
@click.option(
    "-nf",
    "--n_fields",
    default=1,
    type=vpype_cli.IntegerType(),
    help="Number of rotated copies of the flow field",
)
@click.option(
    "-ms",
    "--min_sep",
    default=0.8,
    type=vpype_cli.FloatType(),
    help="Minimum flowline separation",
)
@click.option(
    "-Ms",
    "--max_sep",
    default=10,
    type=vpype_cli.FloatType(),
    help="Maximum flowline separation",
)
@click.option(
    "-ml",
    "--min_length",
    default=0,
    type=vpype_cli.FloatType(),
    help="Minimum flowline length",
)
@click.option(
    "-Ml",
    "--max_length",
    default=40,
    type=vpype_cli.FloatType(),
    help="Maximum flowline length",
)
@click.option(
    "--max_size",
    default=800,
    type=vpype_cli.IntegerType(),
    help="The input image will be rescaled to have sides at most max_size px",
)
@click.option(
    "--search_ef",
    "-ef",
    default=50,
    type=vpype_cli.IntegerType(),
    help="HNSWlib search ef (higher -> more accurate, but slower)",
)
@click.option(
    "-s", "--seed", type=vpype_cli.IntegerType(), help="PRNG seed (overriding vpype seed)"
)
@click.option(
    "-fs", "--flow_seed", type=vpype_cli.IntegerType(),
    help="Flow field PRNG seed (overriding the main `--seed`)"
)
@click.option(
    "-tf", "--test_frequency", type=vpype_cli.FloatType(), default=2,
    help="Number of separation tests per current flowline separation",
)
@click.option(
    "-f", "--field_type",
    type=vpype_cli.ChoiceType(['noise', 'curl_noise'], case_sensitive=False),
    help="flow field type [default: noise]")
@click.option(
    "--transparent_val", type=click.IntRange(0, 255), default=127,
    help="Value to replace transparent pixels")
@click.option(
    "-tm", "--transparent_mask", is_flag=True,
    help="Remove lines from transparent parts of the source image.")
@click.option(
    "-efm", "--edge_field_multiplier", type=vpype_cli.FloatType(), default=None,
    help="flow along image edges")
@click.option(
    "-dfm", "--dark_field_multiplier", type=vpype_cli.FloatType(), default=None,
    help="flow swirling around dark image areas")
@click.option(
    "-kdt", "--kdtree_searcher", is_flag=True,
    help="Use exact nearest neighbor search with kdtree (slower, but more precise)")
@click.option(
    "--cmyk", is_flag=True,
    help="Split image to CMYK and process each channel separately.  The results are in consecutively numbered layers, starting from `layer`.")
@click.option(
    "--rotate", type=vpype_cli.FloatType(), default=0, metavar='DEGREES',
    help="rotate the flow field")
@click.option(
        "-l",
        "--layer",
        type=vpype_cli.LayerType(accept_new=True),
        default=None,
        help="Target layer or 'new'.  When CMYK enabled, this indicates the first (cyan) layer.",
    )
@vpype_cli.global_processor
def vpype_flow_imager(document, layer, filename, noise_coeff, n_fields,
                      min_sep, max_sep,
                      min_length, max_length, max_size,
                      seed, flow_seed, search_ef,
                      test_frequency,
                      field_type, transparent_val, transparent_mask,
                      edge_field_multiplier, dark_field_multiplier,
                      kdtree_searcher,
                      cmyk, rotate):
    """
    Generate flowline representation from an image.

    The generated flowlines are in the coordinates of the input image,
    resized to have dimensions at most `--max_size` pixels.
    """
    if kdtree_searcher:
        searcher_class = KDTSearcher
    else:
        if hnswlib is None:
            logger.warning("Could not import hnswlib, falling back to KD-tree searcher.  Make sure to install with vpype-flow-imager[all], if you want to use the default HNSWlib searcher.")
            searcher_class = KDTSearcher
        else:
            searcher_class = HNSWSearcher
    target_layer = vpype_cli.single_to_layer_id(layer, document)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    logger.debug(f"original img.shape: {img.shape}")
    with tmp_np_seed(seed):
        if cmyk:
            img_layers = split_cmyk(img.copy())
        else:
            img_layers = [img]

        alpha = get_alpha_channel(img)

        for layer_i, img_layer in enumerate(img_layers):
            logger.info(f"computing layer {layer_i+1}")
            numpy_paths = draw_image(img_layer, alpha,
                                     mult=noise_coeff, n_fields=n_fields,
                                     min_sep=min_sep, max_sep=max_sep,
                                     min_length=min_length, max_length=max_length,
                                     max_img_size=max_size, flow_seed=flow_seed,
                                     search_ef=search_ef,
                                     test_frequency=test_frequency,
                                     field_type=field_type,
                                     transparent_val=transparent_val,
                                     transparent_mask=transparent_mask,
                                     edge_field_multiplier=edge_field_multiplier,
                                     dark_field_multiplier=dark_field_multiplier,
                                     searcher_class=searcher_class,
                                     rotate=rotate,
                                     )

            lc = vp.LineCollection()
            for path in numpy_paths:
                lc.append(path[:, 0] + path[:, 1] * 1.j)

            document.add(lc, target_layer + layer_i)
    document.add_to_sources(filename)
    return document


vpype_flow_imager.help_group = "Plugins"


def get_alpha_channel(img):
    """ Return alpha channel from opencv image, or None. """
    if len(img.shape) == 3 and img.shape[2] == 4:
        return img[:, :, 3]


def split_cmyk(img):
    post_gamma = 1

    if img.shape[2] == 4:  # rgba
        img = img[:, :, :3]
    rgb = img[:, :, ::-1]
    p_rgb = Image.fromarray(rgb)
    cmyk = np.array(p_rgb.convert("CMYK")).astype(np.float64) / 255
    cmyk = cmyk ** post_gamma
    # this conversion does not use the black at all (icc profiles and stuff...)
    # so lets compute the black channel ourselves
    black_percentage = 1

    black = np.amin(cmyk[:, :, 0:3], axis=2, keepdims=True)
    black_mask = black == 1
    non_black_mask = np.logical_not(black_mask)
    cmyk[non_black_mask[..., 0], :] = ((cmyk[non_black_mask[..., 0], :] -
                                        black_percentage * black[non_black_mask, np.newaxis]))
    cmyk[non_black_mask[..., 0], :] /= (1 - black_percentage * black[non_black_mask, np.newaxis])

    cmyk[black_mask[..., 0]] = 0
    cmyk[:, :, 3] = black_percentage * black[:, :, 0]

    cmyk = 255 * (1 - cmyk)  # invert to get back intensity
    cmyk = np.clip(cmyk, 0, 255).astype(np.uint8)

    cmyk_channels = np.split(cmyk, 4, axis=2)
    # ch_names = ['c', 'm', 'y', 'k']
    # for i, ch in enumerate(cmyk_channels):
    #     ch_name = ch_names[i]
    #     cv2.imwrite(f'/tmp/00000_cmyk_{i}{ch_name}.png', ch)
    # sys.exit(1)
    return cmyk_channels


def norm_2vec(x):
    return np.sqrt(x[0]**2 + x[1]**2)


def gen_flow_field(H, W, x_mult=1, y_mult=None):
    if y_mult is None:
        y_mult = x_mult
    x_noise = OpenSimplex(np.random.randint(9393931))
    y_noise = OpenSimplex(np.random.randint(9393931))
    field = np.zeros((H, W, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            x_val = x_noise.noise2(x=x_mult * x, y=x_mult * y)
            y_val = y_noise.noise2(x=y_mult * x, y=y_mult * y)
            norm = np.sqrt(x_val ** 2 + y_val ** 2)
            if norm > eps:
                x_val /= norm
                y_val /= norm
            else:
                x_val, y_val = 1, 0
            field[y, x, :] = (x_val, y_val)

    return field


def gen_curl_flow_field(H, W, x_mult=1, y_mult=None):
    if y_mult is None:
        y_mult = x_mult
    noise = OpenSimplex(np.random.randint(9393931))
    field = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            val = noise.noise2(x=x_mult * x, y=x_mult * y)
            field[y, x] = val

    grad_y, grad_x = np.gradient(field)
    field = np.stack((grad_y, -grad_x), axis=2)
    field = normalize_flow_field(field)

    return field


def gen_edge_flow_field(H, W, intensities):
    from scipy.ndimage import distance_transform_edt
    edges = cv2.Canny(intensities, 100, 200)
    variable_mask = edges <= 0
    grad_y, grad_x = np.gradient(intensities)
    field = np.stack((grad_y, -grad_x), axis=2)
    field = normalize_flow_field(field)
    field[variable_mask, :] = 0

    for i in range(35):
        k_sz = 15
        new_field = cv2.blur(field, (k_sz, k_sz))
        field[variable_mask, :] = new_field[variable_mask, :]

    weights = distance_transform_edt(edges == 0)
    weights = weights[:, :, np.newaxis].astype(np.float32)
    max_dist = 100
    weights = (max_dist - np.clip(weights, 0, max_dist)) / max_dist
    return normalize_flow_field(field), weights


def gen_darkness_curl_flow_field(H, W, intensities):
    assert len(intensities.shape) == 2
    blur_kernel = 2 * 87 + 1
    heights = cv2.GaussianBlur(intensities.astype(np.float32),
                               (blur_kernel, blur_kernel), 0)

    grad_y, grad_x = np.gradient(heights)

    field = np.stack((grad_y, -grad_x), axis=2)
    weights = 1 - (intensities[:, :, np.newaxis].astype(np.float32) / 255)
    return normalize_flow_field(field), weights


def normalize_flow_field(field):
    norm = np.sqrt(np.sum(field ** 2, axis=2)).reshape(*field.shape[:2], 1)
    return field / (norm + 1e-10)


def draw_image(gray_img, alpha,
               mult, max_img_size=800, n_fields=1,
               min_sep=0.8, max_sep=10,
               min_length=0, max_length=40,
               flow_seed=None,
               search_ef=50, test_frequency=2,
               transparent_val=127, transparent_mask=True,
               field_type='noise',
               edge_field_multiplier=None, dark_field_multiplier=None,
               searcher_class=None, rotate=0):
    logger.debug(f"gray_img.shape: {gray_img.shape}")
    gray = resize_to_max(gray_img, max_img_size)
    logger.debug(f"gray.shape: {gray.shape}")
    if len(gray.shape) == 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    H, W, C = gray.shape
    if alpha is not None:
        data_mask = resize_to_max(alpha, max_img_size) > 0
    else:
        data_mask = np.ones((H, W)) > 0

    background_mask = np.logical_not(data_mask)
    gray = cv2.cvtColor(gray[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray[background_mask] = transparent_val

    logger.info('Generating flow field')
    with tmp_np_seed(flow_seed):
        if field_type == 'curl_noise':
            noise_field = gen_curl_flow_field(H, W, x_mult=mult)
        else:
            noise_field = gen_flow_field(H, W, x_mult=mult)

    field = np.zeros_like(noise_field)
    weights = np.zeros_like(noise_field)

    if edge_field_multiplier is not None:
        edge_field, edge_weights = gen_edge_flow_field(H, W, gray)
        field += edge_weights * edge_field * edge_field_multiplier
        weights += edge_weights * edge_field_multiplier

    if dark_field_multiplier is not None:
        dark_field, dark_weights = gen_darkness_curl_flow_field(H, W, gray)
        field += dark_weights * dark_field * dark_field_multiplier
        weights += dark_weights * dark_field_multiplier

    field += np.clip(1 - weights, 0, 1) * noise_field

    field[background_mask, :] = noise_field[background_mask, :]
    field = normalize_flow_field(field)
    field = rotate_field(field, rotate)
    fields = [VectorField(field)]
    if n_fields > 1:
        angles = np.linspace(0, 360, n_fields + 1)
        for angle in angles:
            fields.append(VectorField(rotate_field(field, angle)))

    guide = gray

    def d_sep_fn(pos):
        x, y = fit_inside(np.round(pos), guide)
        val = guide[int(y), int(x)] / 255
        val = val**2
        return remap(val, 0, 1, min_sep, max_sep)

    logger.info('Drawing flowlines')
    paths = draw_fields_uniform(fields, d_sep_fn,
                                seedpoints_per_path=40,
                                guide=guide,
                                min_length=min_length, max_length=max_length,
                                search_ef=search_ef,
                                test_frequency=test_frequency,
                                searcher_class=searcher_class)

    if transparent_mask:
        paths = mask_paths(paths, data_mask)
    return paths


def mask_paths(paths, fg_mask):
    """ Remove paths not on foreground mask """
    logger.debug(f"np.sum(fg_mask > 0): {np.sum(fg_mask > 0)}")
    logger.debug(f"fg_mask.size: {fg_mask.size}")
    masked_paths = []
    for path in paths:
        current_path = []
        for i in range(len(path)):
            pt = path[i, :]
            x, y = fit_inside(np.round(pt), fg_mask)
            mask_val = fg_mask[int(y), int(x)]
            pt_on_fg = mask_val > 0

            if not pt_on_fg:
                if len(current_path) >= 2:
                    masked_paths.append(np.array(current_path))
                current_path = []
            else:
                current_path.append(pt)
        if len(current_path) >= 2:
            masked_paths.append(np.array(current_path))
    return masked_paths


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
                        min_length=0, max_length=20,
                        search_ef=50, test_frequency=2,
                        searcher_class=None):
    if d_test_fn is None:
        def d_test_fn(*args, **kwargs):
            return d_sep_fn(*args, **kwargs) / test_frequency

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

    searcher = searcher_class([np.array([-10, -10])],
                              max_elements=64000,
                              search_ef=search_ef)
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
                                      should_stop_fn=should_stop,
                                      searcher_class=searcher_class)
            if len(path) <= 2:
                # nothing found
                # logging.debug('streamline ended immediately')
                continue

            for pt in path:
                searcher.add_point(pt)
            paths.append(path)

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
                       should_stop_fn, searcher_class):
    direction_sign = 1  # first go with the field
    pos = seed_pos.copy()
    paths = []
    path = LinePath()
    path.append(pos.copy())
    stop_tracking = False
    self_searcher = searcher_class([(-20, -20)])
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
    singleline.extend(paths[0][1:])
    singleline = np.array(singleline)

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

    seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * direction)
    seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * direction +
                 margin * d_sep_fn(cur_xy) * normal)
    seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * direction -
                 margin * d_sep_fn(cur_xy) * normal)

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

    seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * direction)
    seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * direction +
                 margin * d_sep_fn(cur_xy) * normal)
    seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * direction -
                 margin * d_sep_fn(cur_xy) * normal)

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
    def __init__(self, points, max_elements=1000, search_ef=50):
        self.index = hnswlib.Index(space='l2', dim=2)
        self.max_elements = max_elements
        self.index.init_index(max_elements=self.max_elements,
                              ef_construction=200, M=16)
        self.search_ef = search_ef
        self.index.set_ef(search_ef)
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
        logger.debug(f'Resizing searcher index to {self.max_elements}')
        self.index.resize_index(self.max_elements)
        logger.debug('after resize:')
        logger.debug(f"self.index.max_elements: {self.index.max_elements}")
        logger.debug(f"self.index.element_count: {self.index.element_count}")
        self.index.set_ef(self.search_ef)

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
