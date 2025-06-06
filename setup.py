from setuptools import find_packages, setup

setup(
    name="complexNN",
    version="0.5.1",
    description="A toolbox for using complex valued standard network modules in PyTorch.",
    author="Xinyuan Liao",
    author_email="xnyuanliao@gmail.com",
    url="https://github.com/XinyuanLiao/ComplexNN",
    packages=find_packages(),
    install_requires=["torch", "numpy"],
    python_requires=">=3.6",
    license="Apache-2.0 License",
    zip_safe=False,
    keywords="pytorch, deep learning, complex values, time series",
    classifiers=[""],
)
