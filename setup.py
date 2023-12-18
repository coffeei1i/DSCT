from setuptools import setup, find_packages

setup(
    name='DSCT',
    version='0.0.1',
    description='Efficient Spatial Transcriptomic Cell Typing Method Using Deep Learning and Attention Mechanisms',
    url='https://github.com/coffeei1i/DSCT/',
    author='Yiheng Xu',
    license='GPLv3',
    packages=find_packages(),
    scripts=['src/DSCT_train.py', 'src/DSCT_load.py'],
    include_package_data=True,
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.23.4',
        'pandas>=2.0.3',
        'scanpy>=1.9.4',
        'anndata>=0.9.2',
        'diopy>=0.5.5',
        'cosg>=1.0.1',
        'matplotlib>=3.3.4',
        'scipy>=1.5.4'
    ]
)
