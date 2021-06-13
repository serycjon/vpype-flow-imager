# vpype flow imager

<img src="https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee.jpg?raw=true" width="300" /> <img src="https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee_out.png?raw=true" width="300" />

[`vpype`](https://github.com/abey79/vpype) plug-in to convert images to flow field line art inspired by Sean M. Puckett's work and the "Creating evenly-spaced streamlines of arbitrary density" paper by Jobard and Lefer.

## Getting Started

You will need a C++ compiler before running the flow imager installation. One way of getting the compiler on Windows is installing Visual Studio with C++ package ([tutorial](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-160)).

Install `vpype flow imager` using the following commands:
```bash
$ python3.8 -m venv my_virtual_env
$ source my_virtual_env/bin/activate
$ pip install git+https://github.com/serycjon/vpype-flow-imager.git#egg=vpype-flow-imager
```

`vpype` is automatically installed with `vpype flow imager`, so no further steps are required.

You can confirm that the installation was successful with the following command, which also happens to tell you all
you need to know to use `vpype flow imager`:

```bash
$ vpype flow_img --help
Usage: vpype flow_img [OPTIONS] FILENAME

  Generate flowline representation from an image.

  The generated flowlines are in the coordinates of the input image, resized
  to have dimensions at most `--max_size` pixels.

Options:
  -nc, --noise_coeff FLOAT        Simplex noise coordinate multiplier. The
                                  smaller, the smoother the flow field.
                                  [default: 0.001]

  -nf, --n_fields INTEGER         Number of rotated copies of the flow field
                                  [default: 1]

  -ms, --min_sep FLOAT            Minimum flowline separation  [default: 0.8]
  -Ms, --max_sep FLOAT            Maximum flowline separation  [default: 10]
  -ml, --min_length FLOAT         Minimum flowline length  [default: 0]
  -Ml, --max_length FLOAT         Maximum flowline length  [default: 40]
  --max_size INTEGER              The input image will be rescaled to have
                                  sides at most max_size px  [default: 800]

  -ef, --search_ef INTEGER        HNSWlib search ef (higher -> more accurate,
                                  but slower)  [default: 50]

  -s, --seed INTEGER              PRNG seed (overriding vpype seed)
  -fs, --flow_seed INTEGER        Flow field PRNG seed (overriding the main
                                  `--seed`)

  -tf, --test_frequency FLOAT     Number of separation tests per current
                                  flowline separation  [default: 2]

  -f, --field_type [noise|curl_noise]
                                  flow field type [default: noise]
  --transparent_val INTEGER RANGE
                                  Value to replace transparent pixels
                                  [default: 127]

  -efm, --edge_field_multiplier FLOAT
                                  flow along image edges
  -dfm, --dark_field_multiplier FLOAT
                                  flow swirling around dark image areas
  -kdt, --kdtree_searcher         Use exact nearest neighbor search with
                                  kdtree (slower, but more precise)  [default:
                                  False]

  --cmyk                          Split image to CMYK and process each channel
                                  separately.  The results are in
                                  consecutively numbered layers, starting from
                                  `layer`.  [default: False]

  -l, --layer LAYER               Target layer or 'new'.  When CMYK enabled,
                                  this indicates the first (cyan) layer.

  --help                          Show this message and exit.  [default:
                                  False]
```

To create a SVG, combine the `flow_img` command with the `write` command (check `vpype`'s documentation for more
information). Here is an example:

```bash
$ vpype flow_img input.jpg write output.svg
```

## Examples

The example output was generated with:
```bash
$ cd examples
$ vpype flow_img -nf 6 coffee.jpg write coffee_out.svg show
```
It took around 3 minutes on my laptop.
In this example, the flow field was rotated 6 times to get hexagonal structure in the result.

The default:
```bash
$ vpype flow_img coffee.jpg write coffee_out.svg show
```
produces a smoother result like:

<img src="https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee_single.png?raw=true" width="300" />

You can control the result line density by changing the `--min_sep` and `--max_sep` parameters.

You can also locally override the vpype PRNG seed using the `--seed` and `--flow_seed` parameters.  The `--flow_seed` is used only in the flow field construction, so if you want to create a multi-layer svg (e.g. CMYK), you can do something like
```bash
vpype flow_img -fs 42 -l 1 C.jpg flow_img -fs 42 -l 2 M.jpg flow_img -fs 42 -l 3 Y.jpg flow_img -fs 42 -l 4 K.jpg write --layer-label "Pen%d" cmyk.svg show
```
By specifying the same `-fs` (`--flow_seed`) for all the layers, you will get the same flowline directions on all the layers.

The following is an example with `curl_noise` and `dark_field` enabled:
```bash
vpype -v -s 42 flow_img -f curl_noise -dfm 1 -nc 0.03 examples/coffee.jpg write examples/coffee_dark.svg
```

<img src="https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee_dark.png?raw=true" width="300" />

## Parameters
Starting from the most interesting / useful:
* `min_sep`, `max_sep` - Control the flowline density (separation between flowlines)
* `min_length`, `max_length` - Control the flowline length.  (setting `min_length > 0` breaks the flowline density constraints)
* `field_type` - Set to `noise` (default) to get opensimplex noise flow field, set to `curl_noise` to get curly flow field.
* `cmyk` - convert the input RGB image into CMYK and output four layers.
* `n_fields` - Number of rotated copies of the flow field (default: 1). For example, try out 3, 4, or 6 to get triangular, rectangular, or hexagonal patterns.
* `edge_field_multiplier` - When set to a number (try 1 first), the input image is processed to detect edges. A new flow field, that follows the edges is then calculated and merged with the noise field based on the distance to the image edge and this `edge_field_multiplier`, i.e. the resulting flow follows image edges when close to them and the noise field when far from edges.
* `dark_field_multiplier` - Similarly, when you set `dark_field_multiplier` (again, try 1), a new flow field is constructed. This one curls in dark image areas and gets added to the other flows, weighted by darkness and the `dark_field_multiplier`.  You can combine both `edge_field_multiplier` and `dark_field_multiplier` at the same time.
* `seed`, `flow_seed` - Set `seed` to a number to get reproducible results. Set `flow_seed` to a number to get reproducible flow field (but the resulting flowlines are still pseudorandom).
* `kdtree_searcher` - use exact nearest neighbor search.  This gets rid of occasional dark clumps, but the computation is much slower.
* `transparent_val` - Transparent pixels (from e.g. RGBA png image) get replaced by this 0-255 intensity (default 127). The transparent image parts always use the noise field (either `noise` or `curl_noise`) without image-controlled fields (`edge_field`, `dark_field`).  This can be used to obtain contrasting background.

(feel free to create a pull request with better documentation)
## License

GNU GPLv3. See the [LICENSE](LICENSE) file for details.
Example coffee photo by [jannoon028](https://www.freepik.com/free-photo/cup-coffee-viewed-from_992559.htm)
Kd-tree searcher CC0 from [Python-KD-Tree](https://github.com/Vectorized/Python-KD-Tree).

