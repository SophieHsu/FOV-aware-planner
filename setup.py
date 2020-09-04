#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='overcooked_ai',
      version='0.0.1',
      description='Cooperative multi-agent environment based on Overcooked',
<<<<<<< HEAD
      author='Micah Carroll<mdc@berkeley.edu>, Matt Fontaine<mfontain@usc.edu>, Stefanos Nikolaidis<nikolaid@usc.edu>, Yulun Zhang<yulunzha@usc.edu>',
=======
      author='Micah Carroll', 'Matt Fontaine', 'Stefanos Nikolaidis', 'Yulun Zhang'
      author_email='mdc@berkeley.edu', 'mfontain@usc.edu', 'nikolaid@usc.edu', 'yulunzha@usc.edu'
>>>>>>> parent of 050516c... add mdp agent planner and agent
      packages=find_packages(),
      install_requires=[
        'numpy',
        'tqdm',
        'gym',
        'pygame',
        'torch',
        'matplotlib',
        'pandas',
      ]
    )