from connect4_rl.agents.training.custom_agent_base import Agent
from connect4_rl.agents.training.custom_net import CustomNetwork

class TrainableAgent(Agent):
    """ Agent with a trainable neural network. """

    def __init__(self,
                 model: CustomNetwork,
                 name: str = 'Trainable Agent',
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.model = model

    def save_weights(self, file_path: str, training_hparams: dict = None) -> None:
        self.model.save_weights(file_path=file_path, training_hparams=training_hparams)

    def load_weights(self, file_path: str) -> None:
        self.model.load_weights(file_path=file_path)
