class Interval:
    def __init__(self, points):
        self.points = points

    def get_rad(self):
        return (max(self.points) - min(self.points)) / 2

    def get_mid(self):
        return (self.points[0] + self.points[1]) / 2

    def get_point(self):
        interval_r = (self.points[1] - self.points[0]) / 2
        return self.points[1] - interval_r

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
