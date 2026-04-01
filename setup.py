import io
import os

from setuptools import find_packages, setup

for line in open("snac/__init__.py"):
    line = line.strip()
    if "__version__" in line:
        context = {}
        exec(line, context)
        VERSION = context["__version__"]


def read(*paths, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8")) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="snac",
    version=VERSION,
    author="Kadir Nar",
    author_email="kadir.nar@hotmail.com",
    description="Fast inference engine for SNAC neural audio codec",
    url="https://github.com/kadirnar/fast-snac",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)
