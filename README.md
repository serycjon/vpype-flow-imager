# vpype flow imager

![source image](https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee.jpg?raw=true)
![resulting image](https://github.com/serycjon/vpype-flow-imager/blob/master/examples/coffee_out.png?raw=true)

[`vpype`](https://github.com/abey79/vpype) plug-in to convert images to flow field line art.

## Examples

_to be completed_


## Installation

See the [installation instructions](https://vpype.readthedocs.io/en/stable/install.html) for information on how
to install `vpype`.


### Existing `vpype` installation

Use this method if you have an existing `vpype` installation (typically in an existing virtual environment) and you
want to make this plug-in available. You must activate your virtual environment beforehand.

```bash
$ pip install git+https://github.com/serycjon/vpype-flow-imager.git#egg=vpype-flow-imager
```

Check that your install is successful:

```
$ vpype --help
Usage: vpype [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  -v, --verbose
  -I, --include PATH  Load commands from a command file.
  --help              Show this message and exit.

Commands:
[...]
  Plugins:
    flow_img
[...]
```

### Stand-alone installation

Use this method if you need to edit this project. First, clone the project:

```bash
$ git clone https://github.com/serycjon/vpype-flow-imager.git
$ cd vpype-flow-imager
```

Create a virtual environment:

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
```

Install `vpype-flow-imager` and its dependencies (including `vpype`):

```bash
$ pip install -e .
```

Check that your install is successful:

```
$ vpype --help
Usage: vpype [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  -v, --verbose
  -I, --include PATH  Load commands from a command file.
  --help              Show this message and exit.

Commands:
[...]
  Plugins:
    flow_img
[...]
```


## Documentation

The complete plug-in documentation is available directly in the CLI help:

```bash
$ vpype flow_img --help
```


## License

GNU GPLv3. See the [LICENSE](LICENSE) file for details.
example coffee photo by [jannoon028](https://www.freepik.com/free-photo/cup-coffee-viewed-from_992559.htm)
