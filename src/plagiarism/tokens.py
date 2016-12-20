from plagiarism.tokenizers import split_to_words


class Token:
    """
    Base class for token types.
    """

    __slots__ = ('type', 'value')
    TYPE_WORD = 'word'
    TYPE_COMMENT = 'comment'

    def __init__(self, value, type=TYPE_WORD):
        self.value = value
        self.type = type

    def __str__(self):
        return self.value

    def __repr__(self):
        return 'Token(%r, %r)' % (self.value, self.type)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.value == other.value and self.type == other.type
        return NotImplemented


class WordTokenizer:
    def __call__(self, data):
        tok_type = Token.TYPE_WORD
        for word in split_to_words(data):
            yield Token(word, tok_type)


class CodeTokenizer(WordTokenizer):
    # Bogus implementation
    pass


