import string
from collections import Counter, OrderedDict

import numpy as np
import scipy as sp
import scipy.cluster.vq

from plagiarism.stopwords import get_stop_words
from plagiarism.tokenizers import split_to_words
from plagiarism.tokens import CodeTokenizer


class Document:
    """
    Represents a single object of analysis.

    Each document offers a series of metrics that are used to perform the
    k-means clusterization.

    You may override this class to provide different metrics.
    """

    def __init__(self, data):
        self.data = data

    def core_metrics(self):
        """
        Return a list/array of core metric values.

        All documents in a Job must provide the same core metrics. Documents may
        also provide additional metrics using the optional_metrics() method.
        """

        return []

    def optional_metrics(self):
        """
        Return a mapping of metrics and their respective values.

        These metrics may not be used in the analysis if the Job controller
        deems it as non-informative.
        """

        return {}

    def metrics(self):
        """
        Return the (possibly cached) pair of metrics (core, optional).
        """

        try:
            return self._metrics
        except AttributeError:
            self._metrics = self.core_metrics(), self.optional_metrics()
            return self._metrics


def ascii_moving_average(text):
    N = len(text)
    weight = N * (N - 1) / 2
    return sum(i * ord(x) for (i, x) in enumerate(text)) / weight


class TextDocument(Document):
    """
    Common metrics for all text-based documents.
    """

    def __init__(self, data):
        super().__init__(data)
        self._size = len(self.data)

    def core_metrics(self, segmentation=None):
        if isinstance(segmentation, str):
            segmentation = getattr(self, segmentation)
        elif segmentation is None:
            segmentation = self.words
        segments = segmentation()

        return [
            self.size(),
            self.whitespace_ratio(),
            self.letters_ratio(),
            self.digits_ratio(),
            self.lower_ratio(),
            self.upper_ratio(),
            self.ascii_moving_average(),
            self.ascii_moving_average_rev(),
            self.mean_segment_size(segments),
            self.std_segment_size(segments),
        ]

    # Segmentation functions
    def words(self):
        """
        Return a list of case-folded tokens in the given text.

        Remove all extra whitespace, punctuations, digits, etc.
        """

        return split_to_words(self.data)

    def mean_segment_size(self, segments=None):
        """
        Return the average segment size.
        """

        return np.array(list(map(len, segments))).mean()

    def std_segment_size(self, segments=None):
        """
        Return the average segment size.
        """

        return np.array(list(map(len, segments))).std()

    # Standard text metrics
    def size(self):
        """
        Text size
        """

        return self._size

    def whitespace_ratio(self):
        """
        Fraction whitespace characters.
        """

        return self.group_ratio(string.whitespace)

    def letters_ratio(self):
        """
        Fraction ascii letters.
        """

        return self.group_ratio(string.ascii_letters)

    def digits_ratio(self):
        """
        Fraction digit characters.
        """

        return self.group_ratio(string.digits)

    def lower_ratio(self):
        """
        Fraction digit characters.
        """

        letters = string.ascii_letters
        lowercase = string.ascii_lowercase
        return self.group_count(lowercase) / self.group_count(letters)

    def upper_ratio(self):
        """
        Fraction digit characters.
        """

        letters = string.ascii_letters
        uppercase = string.ascii_uppercase
        return self.group_count(uppercase) / self.group_count(letters)

    def group_ratio(self, group):
        """
        Fraction of characters that belong to the given group.
        """

        return self.group_count(group) / self._size

    def group_count(self, group):
        """
        Return the number of occurrences of characters in the group.
        """

        return sum(1.0 for x in self.data if x in group)

    def ascii_moving_average(self):
        """
        Moving average for ascii ordinals.
        """

        return ascii_moving_average(self.data)

    def ascii_moving_average_rev(self):
        """
        Moving average for ascii ordinals in reverse order.
        """

        return ascii_moving_average(self.data[::-1])


class CodeDocument(TextDocument):
    """
    Common metrics for documents based on source code.
    """

    def __init__(self, data, tokenizer=None):
        super().__init__(data)
        self.tokenizer = tokenizer or CodeTokenizer()

    def core_metrics(self):
        metrics = super().core_metrics()
        metrics.extend([
            self.comment_ratio(),
        ])
        return metrics

    def comment_ratio(self):
        """
        Fraction of characters in the source code consisting of comments.
        """

        return 0

    def optional_metrics(self):
        data = Counter()
        for tk in self.tokenizer(self.data):
            if tk.type == 'comment':
                for word in self.words(str(tk)):
                    data['comment:' + word] += 1
            elif tk.type == 'unknown':
                pass
            else:
                data[str(tk)] += 1
        return data


class MultiDocument(Document):
    def core_metrics(self):
        # Join a sorted list of metrics
        metrics = []
        for _, document in sorted(self.data.items()):
            metrics.extend(document.core_metrics())
        return metrics

    def optional_metrics(self):
        # Adds a "name:"  prefix to all optional metric names
        data = {}
        for name, document in self.data.items():
            data.update({'%s:%s' % (name, k): v for (k, v) in document.items()})
        return data


class Job:
    def __init__(self, data=None, cls=TextDocument, stop_words=None):
        self.cls = cls
        self.data = OrderedDict(data or {})
        for name, document in self.data.items():
            self.data[name] = cls(document)

        # Register common tokens and the corresponding comment tokens.
        self.stop_words = get_stop_words(stop_words)
        self.comment_words = ['comment:' + w for w in self.stop_words]
        self.bad_metrics = self.stop_words + self.comment_words

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data.values())

    def count_optional(self):
        """
        Return a Counter() objects telling how many times each optional metric
        name appear in the registered documents.
        """
        counter = Counter()
        for document in self.data.values():
            for key in document.metrics()[1]:
                counter[key] += 1
        return counter

    def select_metrics(self, limit=None):
        """
        Return a list of all optional metric names that are recommended to be
        used for classification.
        """

        # Keep only elements that appear in more than one document
        metrics = {k: v for (k, v) in self.count_optional().items() if v > 1}

        # Remove all common tokens
        common_words = self.bad_metrics
        metrics = {k: v for (k, v) in metrics.items() if k not in common_words}

        # Select the most common, if required
        if limit is None:
            return sorted(metrics.keys())
        counter = Counter(metrics)
        common = counter.most_common(limit)
        return sorted(common.keys())

    def ndarray(self, metrics=None, whiten=True):
        """
        Convert elements into a numpy array.
        """

        data = np.array(self.elements(metrics), dtype=float)
        if whiten:
            data -= data.mean(axis=0)
            std = data.std(axis=0)
            data = data / np.where(std != 0, std, 1)
        return data

    def elements(self, metrics=None):
        """
        Return a list of elements associated with each document.
        """

        if metrics is None:
            metrics = self.select_metrics()
        element = self.element
        return [element(document, metrics) for document in self.data.values()]

    def element(self, document, metrics):
        """
        Return a single element from document and the given metrics.
        """

        core, optional = document.metrics()
        return core + [optional.get(metric, 0) for metric in metrics]

    def kmeans(self, k, data=None):
        data = data or self.ndarray()
        return sp.cluster.vq.kmeans2(data, k, iter=50)

    def classify(self, metrics=None, whiten=True):
        """
        Classify all
        Returns
        -------

        """
