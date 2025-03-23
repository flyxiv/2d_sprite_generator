from setuptools import find_packages, setup

_install_requires = [
    'pyqt5',
    'tqdm',
    'jupyter',
    'scikit-learn',
    'diffusers',
    'transformers',
    'matplotlib',
    'scipy',
    'selenium',
    'torch',
    'torchvision',
    'matplotlib',
    'lxml',
]

_package_excludes = [
    '*.tests'
]

setup(
    name='2d_sprite_generator',
    version='0.0.1',
    packages=find_packages(exclude=_package_excludes),
    install_requires=_install_requires,
    dependency_links=[
        "https://download.pytorch.org/whl/cu126"
    ],
    python_requires='>=3.11'
)
