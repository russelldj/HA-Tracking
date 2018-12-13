class HandTrack(object):
    def __init__(self, ID, contour):
        self.ID = ID
        self.contour = contour
        self.diff = 0
        self.object_track = None

    def has_object_track(self):
        return not self.object_track is None 

    def add_track(self, track):
        self.object_track = track

    def remove_track(self):
        self.object_track = None

    def update_object_track(self):
        pass
