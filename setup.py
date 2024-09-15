from setuptools import setup, find_packages

setup(
    name='my_flask_app',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Flask>=2.0.0',
        'torch>=1.10.0',
        'opencv-python>=4.5.3',
        'numpy>=1.21.2',
        'pyyaml==5.1',
        # detectron2 is not typically included here; install it separately
    ],
    include_package_data=True,
)
