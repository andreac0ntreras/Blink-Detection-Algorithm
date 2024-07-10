from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='BlinkDetection',
    packages=find_packages(include=['BlinkDetection']),
    version='0.1.0',
    description='Python Toolbox for Blink Detection from Eye Tracker Devices',
    author='Andrea Contreras',
    install_requires=requirements,
    setup_requires=['pytest-runner']
)
