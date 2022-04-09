from setuptools import setup


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="vpype-flow-imager",
    version="1.0.5",
    description="Convert images to flow field line art.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jonas Serych",
    url="https://github.com/serycjon/vpype-flow-imager",
    license=license,
    packages=["vpype_flow_imager"],
    install_requires=[
        'click',
        'vpype>=1.10,<2.0',
        'opencv-python-headless',  # headless not to conflict with QT versions in vpype show
        'opensimplex==0.4',
        'tqdm',
        'hnswlib>=0.5.0',
        'scikit-image',
        'pillow',
    ],
    entry_points='''
            [vpype.plugins]
            vpype_flow_imager=vpype_flow_imager.vpype_flow_imager:vpype_flow_imager
        ''',
)
