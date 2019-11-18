import argparse
import json


class Parser:
    """
    Command line parser for the training
    """

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--name", type=str, required=True, help="Run name")
        parser.add_argument("--enable_HER", type=bool, required=False, help="Enable Hindsight Experience replay", default=True)
        parser.add_argument("--num_sampled_goals", type=int, required=False, help="path to images", default=4)
        parser.add_argument("--exploration_fraction", type=float, help="Fraction of training spent exploring",default=0.5)
        parser.add_argument("--tensorboard_log_path", type=str, required=False, help="Path to log files",default='/srv/share/nkannabiran3')
        parser.add_argument("--num_training_steps", type=int, required=False, help="Number of training steps", default=1000000)
        parser.add_argument("--goal_selection_strategy", type=str, required=False, help="Goal selection strategy for HER",default='future')
        self.parser = parser
        self.args = None
    def parse(self, data=None):

        # if data:
        #     return self.parser.parse_args(data)
        self.args = self.parser.parse_args()
        print(self.args)
        return self.args

    def save_args(self):
        json.dump(
            vars(self.args), open(self.args.tensorboard_log_path+"/" + self.args.name + "/config.json", "w")
        )

