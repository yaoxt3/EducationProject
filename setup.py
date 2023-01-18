from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym==0.21.0',
    'numpy==1.24.1',
    'mujoco-py==2.1.2.14'
]


setup(
    name='curi_sim',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
