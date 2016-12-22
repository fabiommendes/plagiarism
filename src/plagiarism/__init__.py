from .__meta__ import __author__, __version__
from .utils import count, count_all, sorted_tokens_all
from .weigths import compute_weights, apply_weights
from . import math_utils as math
from .tokenizers import tokenize, tokenize_all
from .stopwords import *
from .ngrams import *
from .bag_of_words import *
from .clusterization import kmeans
from .reports import *