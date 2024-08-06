from setuptools import setup, find_packages

setup(
    name='autonomous_vehicle_project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'autonomous_vehicle=main:main',
        ],
    },
)
