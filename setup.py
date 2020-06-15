import pathlib
from setuptools import setup, find_packages
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
        name="torchexplain",
        version="0.4.0",
        packages=["explain"],
        setup_requires=["setuptools_scm"],
        install_requires=[
            "numpy",
            "torch",
            "torchvision",
            "opencv-python"
        ],
        author=("Liam Hiley"),
        author_email="hileyl@cardiff.ac.uk",
        description="Extension to PyTorch to include explanations via Layerwise Relevance Propagation",
        long_description=README,
        license="MIT License",
        keywords=["explainability"],
        url="https://github.com/liamhiley/torchexplain",
        project_urls={
            "Tracker":"https://github.com/liamhiley/torchexplain/issues",
        },
        platforms="any",
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
    ],
)

