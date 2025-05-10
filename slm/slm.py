from abc import ABC, abstractmethod

class SLM(ABC):
    @abstractmethod
    def generate_embeddings(self, inputs: list[str]):
        '''
        Generate embeddings from input array of string
        '''
        pass
