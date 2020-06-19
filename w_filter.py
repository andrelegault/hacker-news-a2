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
        with open(filename, encoding='utf-8') as f:
            for stopword in f:
                self.stopwords.add(stopword)
        #super().__init__()

    
    def is_valid(self, word):
        return word not in self.stopwords


class WordLengthFilter(Filter):

    def __init__(self, min_len=3, max_len=8):
        self.min_len = min_len
        self.max_len = max_len


    def is_valid(self, word):
        return min_len <= len(word) <= max_len


class InfrequentWord(Filter):
    #  TODO
    pass

if __name__ == '__main__':
    lol = Filter()
    lol.is_valid('test')
