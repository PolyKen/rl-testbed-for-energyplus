from baselines.ppo1.mlp_policy import MlpPolicy


class DefaultMlpPolicy(MlpPolicy):
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers):
        MlpPolicy.__init__(self, name, ob_space, ac_space, hid_size, num_hid_layers)


# rewrite policy functions here
class NewMlpPolicy(MlpPolicy):
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers):
        MlpPolicy.__init__(self, name, ob_space, ac_space, hid_size, num_hid_layers)