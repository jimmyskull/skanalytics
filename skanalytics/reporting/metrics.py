
class Metric(object):
    def __init__(self, name=None):
        self.name = name

    def register(self, table):
        table.add(self)


class CustomMetric(Metric):
    def __init__(self, function, name=None):
        super(CustomMetric, self).__init__(name)
        self.function = function


class BinaryMetric(Metric):
    def __init__(self, threshold=None, name=None):
        super(BinaryMetric, self).__init__(name)
        self.threshold = threshold


class AccuracyScore(Metric):
    def __init__(self, normalize=None, sample_weight=None, threshold=None,
                 name=None):
        super(AccuracyScore, self).__init__(threshold, name)
        self.normalize = None
        self.sample_weight = None


class AUCScore(Metric):
    def __init__(self, reorder=True, name=None):
        super(AccuracyScore, self).__init__(threshold, name)
        self.normalize = None
        self.sample_weight = None
