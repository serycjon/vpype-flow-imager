from setuptools import setup


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="vpype-flow-imager-nohnsw",
    version="0.1.6",
    description="Convert images to flow field line art.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jonas Serych",
    url="https://github.com/serycjon/vpype-flow-imager",
    license=license,
    packages=["vpype_flow_imager"],
    install_requires=[
        'click',
        'vpype',
        'opencv-python-headless',  # headless not to conflict with QT versions in vpype show
        'opensimplex',
        'tqdm',
    ],
    entry_points='''
            [vpype.plugins]
            vpype_flow_imager=vpype_flow_imager.vpype_flow_imager:vpype_flow_imager
        ''',
)
