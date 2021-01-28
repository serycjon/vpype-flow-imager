from setuptools import setup


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="vpype-flow-imager",
    version="0.1.0",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jonas Serych",
    url="",
    license=license,
    packages=["vpype_flow_imager"],
    install_requires=[
        'click',
        'vpype',
    ],
    entry_points='''
            [vpype.plugins]
            vpype_flow_imager=vpype_flow_imager.vpype_flow_imager:vpype_flow_imager
        ''',
)
