from baselines_energyplus.trpo_mpi.run_energyplus import train
from baselines_energyplus.common.energyplus_util import energyplus_arg_parser
from algorithms import * # please keep it
import sys

# using customized learning algorithms and policy functions

if __name__ == '__main__':
    args = energyplus_arg_parser().parse_args()

    train(env_id=args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          learn=getattr(sys.modules['algorithms'], args.learn),
          policy_fn_class=eval(args.policy))
