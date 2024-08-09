# Configuration file for the Sphinx documentation builder.

import os
project = 'NeuroDOT_py'
copyright = '2024, Brain Light Laboratory'
author = 'Brain Light Laboratory'
release = '1.0.0'

sphinx_gallery_conf = {    
    "default_thumb_file": '_static/logo_small.png',
    "thumbnail_size": (160, 112),
} 

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'nbsphinx',
    "jupyter_sphinx",
    "sphinx.ext.githubpages",
    "sphinx_favicon",
    'autoapi.extension',
    "sphinx.ext.githubpages"
    ]

favicons = [
   "favicon.png"
]
source_suffix = [".rst",".md"]

autoapi_dirs = ['../../src']
templates_path = ['_templates']
exclude_patterns = ['_build','_templates']
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
default_thumb_file = '_static/logo_small.png'
html_theme_options = {
  "header_links_before_dropdown": 6,
  "navigation_depth": 4,
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WUSTL-ORL/NeuroDOT_py",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "NITRC",
            "url": "https://www.nitrc.org/projects/neurodot",
            "icon": "fa-solid fa-brain",
            "type": "fontawesome",
        },
        {
            "name": "Registration",
            "url": "https://forms.gle/astp7aX5Ydg6qsDt8",
            "icon": "fa-solid fa-file-signature",
            "type": "fontawesome"
        }
    ],
  
}
html_extra_path = [
]
html_context = {
   "default_mode": "dark"
}
autosummary_generate = True
