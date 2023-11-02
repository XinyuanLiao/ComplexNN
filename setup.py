from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A toolbox for using complex valued standard network modules in PyTorch.'

setup(
    name="complexNN",
    version=VERSION,
    author="Xinyuan Liao",
    author_email="liaoxinyuan@mail.nwpu.edu.cn",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'pytorch', 'complex number', 'windows', 'mac', 'neural network'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
