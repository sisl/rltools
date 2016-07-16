class Baseline(object):
    def __init__(self, obsfeat_space):
        self.obsfeat_space = obsfeat_space

    def get_params(self, sess):
        raise NotImplementedError()

    def set_params(self, sess, val):
        raise NotImplementedError()

    def fit(self, sess, trajs):
        raise NotImplementedError()

    def predict(self, sess, trajs):
        raise NotImplementedError()

    def update_obsnorm(self, sess, obs):
        raise NotImplementedError()
