from setuptools import setup

package_name = 'mpcc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='f1tenth mpcc',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpcc_node = mpcc.mpcc_node:main',
        ],
    },
)
