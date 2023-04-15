from setuptools import find_packages, setup
setup(
    name='TCLtask',
    packages=find_packages(include=['TCLtask', 'TCLtask.Teachers', 'TCLtask.Utils']),
    #packages=find_packages(),
    version='0.1.0',
    description='Library for TCL task.',
    author='Sergio Filho',
    license='MIT',
    install_requires=['scikit-learn==0.24.1', 
    				  'scipy==1.6.1',     				  
                      'pandas==1.2.2', 
                      'numpy==1.20.1',
                      'func-timeout==4.3.5'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.2.0'],
    test_suite = 'tests',
)