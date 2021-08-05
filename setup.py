import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read().replace("\r\n", "\n")

setuptools.setup(
    name='sensorutils',
    version='0.12.0',
    description='A utils for sensor data',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    author='watashi',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7, <4',
    install_requires=['numpy>=1.19', 'pandas>=1.2', 'scipy>=1.6'],
    tests_require=["pytest", "pytest-cov"],
)