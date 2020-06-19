class Model(ABC):

    def __init__(self, name, data, word_filter = None):
        self.name = name
        self.data = data
        self.training_data = {}
        self.testing_data = {}
        self.filter : Filter = word_filter
    

    @abstractmethod
    def result_to_file(self, filename):
        pass


    @abstractmethod
    def model_to_file(self, filename):
        pass

