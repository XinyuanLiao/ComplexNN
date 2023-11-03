from setuptools import find_packages, setup

setup(
    name="complexNN",
    version="0.1.1",
    description="A toolbox for using complex valued standard network modules in PyTorch.",
    long_description=open("README.md").read().strip(),
    long_description_content_type="text/markdown",
    author="Xinyuan Liao",
    author_email="liaoxinyuan@mail.nwpu.edu.cn",
    url="https://github.com/XinyuanLiao/ComplexNN",
    packages=find_packages(),
    install_requires=["torch", "numpy"],
    python_requires=">=3.6",
    license="Apache-2.0 License",
    zip_safe=False,
    keywords="pytorch, deep learning, complex values, time series",
    classifiers=[""],
)
