from setuptools import find_packages, setup

setup(
    name='GP_CP_models',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'jaxkern @ git+https://github.com/JaxGaussianProcesses/JaxKern.git',
        'distrax==0.1.5'
    ]
)
