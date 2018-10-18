from baselines_energyplus.trpo_mpi.run_energyplus import train
from baselines_energyplus.common.energyplus_util import energyplus_arg_parser

if __name__ == '__main__':
    args = energyplus_arg_parser().parse_args()

    # default
    train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed)
