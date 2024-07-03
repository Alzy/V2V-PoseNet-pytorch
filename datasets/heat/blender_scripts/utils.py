import bpy
import math
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

    def get_longest_side_length(self):
        """
        Returns the length of the longest side of the bounding box.

        Returns:
            float: Length of the longest side.
        """
        side_lengths = self.max_bound - self.min_bound
        return max(side_lengths)


def derive_bounding_box():
    """
    Derives bounding box of the scene (minus cube mesh).
    Bounding box is cubed shaped to match target voxel grid.
    :return:
    """
    bounding_box = BoundingBox()

    scene = bpy.context.scene
    for obj in scene.objects:
        if 'Cube' in obj.name or obj.type != 'MESH':
            continue

        matrix_world = obj.matrix_world
        mesh = obj.data
        world_vertices = [matrix_world @ vertex.co for vertex in mesh.vertices]
        bounding_box.expand(world_vertices)

    og_min_bounds,og_max_bounds = bounding_box.get_bounds()
    model_floor = og_min_bounds.z
    longest_side = bounding_box.get_longest_side_length()

    max_length = math.ceil(longest_side * 2) / 2
    half_length = max_length / 2

    # Check if half_length is smaller than the absolute value of the x or y component
    min_x, min_y = abs(og_min_bounds.x), abs(og_min_bounds.y)
    max_x, max_y = abs(og_max_bounds.x), abs(og_max_bounds.y)
    max_component = max(min_x, min_y, max_x, max_y)

    if half_length < max_component:
        half_length = math.ceil(max_component * 4) / 4
        max_length = half_length * 2

    bounding_box.expand([
        Vector((-half_length, -half_length, 0)),
        Vector((half_length, half_length, model_floor + max_length)),
    ])

    return bounding_box