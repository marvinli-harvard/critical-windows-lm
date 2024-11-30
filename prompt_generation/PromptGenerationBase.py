from abc import ABC, abstractmethod

class PromptGenerationBase(ABC):
    @abstractmethod
    def get_question_tokens(self, question: str, include_stepbystep: bool = True):
        pass

    @abstractmethod
    def get_noise_denoise_question(self, question: str, response_tokens, stop_frac: float = 0.2):
        pass

    @abstractmethod
    def complete_with_answer(self, existing_tokens):
        pass
