from setuptools import setup, find_packages

setup(
    name="QuantumComputing",
    version="0.2",
    packages=find_packages(),  
    install_requires=["numpy",   "sympy", ],
    python_requires=">=3.6",  
    description="A Quantum Computing library with various quantum operations",
    author="Pranat1729",
    author_email="pranat32@gmail.com",  
    url="https://github.com/Pranat1729/Quantum-Computing",  
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    keywords="quantum computing matrix multiplication eigenvalues tensor product",
    entry_points={
        'console_scripts': [
            # Define any command-line tools you want here (optional)
            # 'quantum-cli = quantumcomputing.cli:main',  # Example
        ],
    },
)
