import argparse
import json


class Parser:
    """
    Command line parser for the training
    """

    def __init__(self,mode='HER'):
        parser = argparse.ArgumentParser()
        if mode == 'HER':
            parser.add_argument("--name", type=str, required=True, help="Run name")
            parser.add_argument("--enable_HER", type=bool, required=False, help="Enable Hindsight Experience replay", default=True)
            parser.add_argument("--num_sampled_goals", type=int, required=False, help="path to images", default=4)
            parser.add_argument("--exploration_fraction", type=float, help="Fraction of training spent exploring",default=0.5)
            parser.add_argument("--tensorboard_log_path", type=str, required=False, help="Path to log files",default='/srv/share/nkannabiran3')
            parser.add_argument("--num_training_steps", type=int, required=False, help="Number of training steps", default=1000000)
            parser.add_argument("--goal_selection_strategy", type=str, required=False, help="Goal selection strategy for HER",default='future')
        if mode == 'DQN':
            parser.add_argument("--gamma", type=float,required=False, default=0.99)
            parser.add_argument("--learning_rate", type=float,required=False,default=5e-4)
            parser.add_argument("--buffer_size", type=int, required=False,default=50000)
            parser.add_argument("--exploration_fraction", type=float,required=False,default=0.1)
            parser.add_argument("--exploration_final_eps", type=float,required=False,default=0.02)
            parser.add_argument("--train_freq", type=int, required=False,default=1)
            parser.add_argument("--batch_size", type=int, required=False,default=32)
            parser.add_argument("--double_q",type=bool,required=False,default=True)
            parser.add_argument("--learning_starts", type=int, required=False,default=1000)
            parser.add_argument("--target_network_update_freq", type=int, required=False,default=500)
            parser.add_argument("--prioritized_replay",type=bool,required=False,default=False)
            parser.add_argument("--prioritized_replay_alpha", type=float,required=False,default=0.6)
            parser.add_argument("--prioritized_replay_beta0", type=float,required=False,default=0.4)
            parser.add_argument("--prioritized_replay_beta_iters", type=int, required=False,default=None)
            parser.add_argument("--prioritized_replay_eps", type=float,required=False,default=1e-6)
            parser.add_argument("--param_noise",type=bool,required=False,default=False)
            parser.add_argument("--n_cpu_tf_sess", type=int, required=False,default=None)
            parser.add_argument("--verbose", type=int, required=False,default=0)
            parser.add_argument("--num_learning_steps", type=int, required=False,default=1000000)
            parser.add_argument("--tensorboard_log_name",type=str,required=False,default="DQN")
        if mode == 'A2C':
            parser.add_argument("--gamma", type=float,required=False, default=0.99)
            parser.add_argument("--learning_rate", type=float,required=False,default=0.0007)
            parser.add_argument("--verbose", type=int, required=False,default=0)
            parser.add_argument("--tensorboard_log_name",type=str,required=False,default="A2C")
            parser.add_argument("--num_learning_steps", type=int, required=False,default=1000000)
            parser.add_argument("--lr_schedule",type=str,required=False,default="linear")
            parser.add_argument("--ent_coef", type=float,required=False,default=0.01)
            parser.add_argument("--full_tensorboard_log",type=bool,required=False,default=False)
            
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

