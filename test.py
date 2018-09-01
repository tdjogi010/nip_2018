import pickle
from train import build_env
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args
from baselines import logger
import numpy as np

def main():
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}

    pickle_in = open("./tmp/make_model.pkl","rb")
    # pickle_in = open("./tmp/my_model","rb")
    make_model = pickle.load(pickle_in)
    model = make_model()
    model.load("./tmp/my_model")#can use checkpoints


    logger.log("Running trained model")
    env = build_env(args)
    obs = env.reset()
    # print(obs)
    while True:
        actions = model.step(obs)[0]#0th are actions ... few more other array in step .. need to check for ppo 
        obs, _, done, _  = env.step(actions)
        # env.render()
        # done = done.any() if isinstance(done, np.ndarray) else done
        done = done.all() if isinstance(done, np.ndarray) else done
        print("step")
        if done:
            break
            # obs = env.reset()



if __name__ == '__main__':
    main()