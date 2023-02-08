__all__ = ['ROOT_PATH', 'random_array']

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import numpy as np

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

ROOT_PATH = Path(find_dotenv()).parent

random_array = np.random.rand(2,3)