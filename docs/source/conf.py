# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'trimap-tools'
copyright = '2025, Kai Cao'
author = 'caokai'
release = '1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_logo = "_static/logo.png"

nb_execution_mode = "off"

highlight_language = "python"
myst_highlight_code_blocks = True
nb_render_docutils = True

# 关键修复：告诉 Sphinx ipython3 = python
from sphinx.highlighting import lexers
from pygments.lexers import PythonLexer
lexers['ipython3'] = PythonLexer()

html_title = ""
html_short_title = ""