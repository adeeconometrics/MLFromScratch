from numpy import ndarray
from abc import abstractmethod

class BaseLayer:
    @abstractmethod
    def forward(self, input:ndarray) -> float: 
        raise NotImplementedError 
    
    @abstractmethod 
    def backward(self, input:ndarray) -> ndarray: 
        raise NotImplementedError