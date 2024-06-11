from mathutils import Vector


class BoundingBox:
    def __init__(self):
        self.min_bound = Vector((float('inf'), float('inf'), float('inf')))
        self.max_bound = Vector((float('-inf'), float('-inf'), float('-inf')))

    def expand(self, points):
        """
        Expands the bounding box to include the given points.

        Args:
            points (list of mathutils.Vector): Points to include in the bounding box.
        """
        for point in points:
            self.min_bound = Vector((min(self.min_bound[i], point[i]) for i in range(3)))
            self.max_bound = Vector((max(self.max_bound[i], point[i]) for i in range(3)))

    def get_bounds(self):
        """
        Returns the minimum and maximum bounds of the bounding box.

        Returns:
            tuple: (min_bound, max_bound)
        """
        return self.min_bound, self.max_bound
