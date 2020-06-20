from abc import ABC, abstractmethod


class Filter(ABC):
    
    def __init__(self):
        super().__init__()


    @abstractmethod
    def is_valid(self):
        """Function to be run on every word of Post title."""
        pass


class StopWordFilter(Filter):

    def __init__(self, filename='stopwords.txt'):
        self.stopwords = set()
        with open(filename, encoding='utf-8') as f:
            lst = f.read().splitlines()
            self.stopwords = set(lst)

    
    def is_valid(self, word):
        return word not in self.stopwords


class WordLengthFilter(Filter):

    def __init__(self, min_len=3, max_len=8):
        self.min_len = min_len
        self.max_len = max_len


    def is_valid(self, word):
        return self.min_len <= len(word) <= self.max_len


class InfrequentWordFilter(Filter):
    #  TODO
    pass

