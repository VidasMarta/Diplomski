import os
import yaml
import datasets
import models
from settings import Settings


# staviti da se spremaju u neki log file metrike po epohama ili tako nesto
class Evaluation:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize other attributes or perform other setup tasks

    def evaluate(self):
        # Add your evaluation code here
        pass