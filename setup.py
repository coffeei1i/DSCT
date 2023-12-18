from setuptools import setup,find_packages

setup(
    name='DSCT',               # 应用名
    version='0.0.1',              # 版本号
    description='Efficient Spatial Transcriptomic Cell Typing Method Using Deep Learning and Attention Mechanisms',
    url='https://github.com/coffeei1i/DSCT/',
    author='Yiheng Xu',
    license='GPLv3',
    packages=find_packages(),
    scripts=['blob/master/DSCT_train.py','blob/master/DSCT_load.py'],
    include_package_data=True,    # 启用清单文件MANIFEST.in
    exclude_package_date={'':['.gitignore']},
    install_requires = ['scanpy',
                        'torch',
                        'diopy',
                       'pandas',
                       'matplotlib',
                        'cosg',
                       'scipy',
                       'annadata']
)
