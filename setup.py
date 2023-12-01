import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read().replace("\r\n", "\n")

setuptools.setup(
    name='sensorutils',
    version='0.14.1',
    description='A utils for sensor data',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    author='watashi',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10, <4',
    install_requires=['numpy>=1.23', 'pandas>=2.0', 'scipy>=1.10'],
    tests_require=["pytest", "pytest-cov"],
)