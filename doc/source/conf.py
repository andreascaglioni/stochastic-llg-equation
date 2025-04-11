# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
sys.path.insert(0, str(Path('..', '..').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'stochastic-llg-equation'
copyright = '2025, Andrea Scaglioni'
author = 'Andrea Scaglioni'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinxcontrib.spelling',
    'sphinxcontrib.bibtex',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/', None)
                       }

# -- Options for bibtex extension ------------------------------------------
bibtex_bibfiles = ['references.bib']


# -- For code snippets in docstrings --------------------------------
pygments_style = 'sphinx'