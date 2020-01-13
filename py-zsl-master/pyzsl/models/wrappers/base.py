class BaseWrapper:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        return self

    def predict_scores(self, *args, **kwargs):
        """ Returns score for each label
        """

        raise NotImplementedError

    def predict_topk(self, *args, **kwargs):
        """ Returns top-k labels
        """

        raise NotImplementedError

    def predict_ranks(self, *args, **kwargs):
        """ Returns ranks of the positive items
        """

        raise NotImplementedError
