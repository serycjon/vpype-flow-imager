# vpype flow imager

<img src="https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee.jpg?raw=true" width="300" /> <img src="https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee_out.png?raw=true" width="300" />

[`vpype`](https://github.com/abey79/vpype) plug-in to convert images to flow field line art inspired by Sean M. Puckett's work and the "Creating evenly-spaced streamlines of arbitrary density" paper by Jobard and Lefer.

## Getting Started

Install `vpype flow imager` with the following command, ideally in a virtual environment:

```bash
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
  -nc, --noise_coeff FLOAT  Simplex noise coordinate multiplier. The smaller,
                            the smoother the flow field.

  -nf, --n_fields INTEGER   Number of rotated copies of the flow field
  -ms, --min_sep FLOAT      Minimum flowline separation
  -Ms, --max_sep FLOAT      Maximum flowline separation
  -Ml, --max_length FLOAT   Maximum flowline length
  --max_size INTEGER        The input image will be rescaled to have sides at
                            most max_size px

  -l, --layer LAYER         Target layer or 'new'.
  --help                    Show this message and exit.
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


## License

GNU GPLv3. See the [LICENSE](LICENSE) file for details.
example coffee photo by [jannoon028](https://www.freepik.com/free-photo/cup-coffee-viewed-from_992559.htm)
