import pathlib

# Add some paths here
## do these break if I make a notebook elsewhere? maybe I should not use relative paths.
PATH = pathlib.Path.cwd()
DATA_DIR = PATH.parent / 'data'
MODELS_DIR = PATH.parent / 'models'
VIZ_DIR = PATH.parent / 'visualisations'