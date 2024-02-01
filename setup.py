import setuptools

with open('requirements.txt') as f:
    requires = [x.strip() for x in f.readlines()]

setuptools.setup(
    name='ind_mod',
    version='0.1.10',
    author='Robert James',
    author_email='robert.james1.19@unimelb.edu.au',
    url='https://bitbucket.org/darkmatteraustralia/ind_mod/src/main/',
    python_requires=">=3.11",
    include_package_data=True,
    package_data={
        'ind_mod': ['models/*.ini', 'data/background_sim_spectra/*.root'],
    },
    install_requires=requires,
    packages=setuptools.find_packages())
