from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym==0.21.0',
    'numpy==1.24.1',
    'mujoco-py==2.1.2.14',
    'scipy==1.10.1',
    'rospkg==1.4.0',
    'pyyaml==6.0',
    'numpy-quaternion==2022.4.3'
]


setup(
    name='curi_sim',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
