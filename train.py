import sys
import multiprocessing 
import os
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from baselines.common.tf_util import get_session
from baselines import bench, logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import atari_wrappers, retro_wrappers


from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.retro_wrappers import RewardScaler

from osim.env import ProstheticsEnv
from osim_rl_helper.helper.wrappers import ClientToEnv, DictToListFull, ForceDictObservation, JSONable

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = set([
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
])


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    alg_kwargs['network'] = "mlp"

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env

def my_make_vec_env(env_id, env_type, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0):
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = ProstheticsEnv(visualize=False)
            env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            env = ForceDictObservation(env)
            env = DictToListFull(env)
            env = JSONable(env)
        
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)))
            
            if reward_scale != 1: return RewardScaler(env, reward_scale)
            else: return env
        return _thunk
    set_global_seeds(seed)
    if num_env > 1: return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else: return DummyVecEnv([make_env(start_index)])


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu /= 2
    nenv = args.num_env or ncpu
    # nenv = 2
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    get_session(tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1))
    env = my_make_vec_env(env_id, env_type, nenv, seed, reward_scale=args.reward_scale)
    env = VecNormalize(env)
    
    return env


def get_env_type(env_id):
    
    # return env_type, env_id
    return "nips", "ProstheticsEnv"

def get_default_network(env_type):
    return 'mlp'
    
def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def parse(v):
    '''
    convert value of a command-line arg to a python object if possible, othewise, keep as string
    '''

    assert isinstance(v, str)
    try:
        return eval(v)
    except (NameError, SyntaxError):
        return v


def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}


    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure("./tmp")
    else:
        logger.configure("./tmp",format_strs = [])
        rank = MPI.COMM_WORLD.Get_rank()

    model, _ = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)


    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs, _, done, _  = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()



if __name__ == '__main__':
    main()