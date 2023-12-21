from setuptools import find_packages, setup

def get_requirements(file_path:str):
    '''
         This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
    name = "Age_Classifier",
    version = '0.0.1',
    author = 'Om',
    author_email = 'oms9635@gmail.com',
    packages = setuptools.find_packages(
        where='src',
        where='packages.txt',
    ),
    package_dir = {
        "" : ".",
        "" : ".packages.txt"
    }
    install_requires = get_requirements('requirements.txt')
)
