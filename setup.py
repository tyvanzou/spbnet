# MOFTransformer version 2.0.0
import re
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt", "r") as f:
    install_requires = f.readlines()
# extras_require = {"docs": ["sphinx", "livereload", "myst-parser"]}

with open("spbnet/__init__.py") as f:
    version = re.search(r"__version__ = [\'\"](?P<version>.+)[\.\"]", f.read()).group(
        "version"
    )


setup(
    name="spbnet",
    version=version,
    description="spbnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jiawen Zou",
    author_email="20307130189@fudan.edu.cn",
    packages=find_packages(),
    package_data={
        "spbnet": [
            "data/libs/**/*",
            "**/*.yaml",
            "visualize/template/**.html",
            # "libs/GRIDAY/*",
            # "libs/GRIDAY/scripts/*",
            # "libs/GRIDAY/FF/*",
            # "assets/*.json",
            # "examples/dataset/*",
            # "examples/dataset/**/*",
            # "examples/raw/*",
            # "examples/visualize/dataset/*",
            # "examples/visualize/dataset/test/*",
        ]
    },
    install_requires=install_requires,
    # extras_require=extras_require,
    scripts=[],
    url="https://tyvanzou.github.io/spbnet/",
    download_url="https://github.com/tyvanzou/spbnet",
    entry_points={
        "console_scripts": [
            "spbnet=spbnet.cli.main:main",
        ]
    },
    # python_requires=">=3.8",
)
