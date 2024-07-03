import bpy
import numpy as np
import math
from utils import BoundingBox, derive_bounding_box
from mathutils import Vector


def get_body_vertices():
    scene = bpy.context.scene
    vertices = []
    for obj in scene.objects:
        if 'Cube' in obj.name or obj.type != 'MESH':
            continue

        obj.data.calc_loop_triangles()
        mesh = obj.data
        matrix_world = obj.matrix_world

        if mesh.loop_triangles:
            # Using indexed vertices
            for tri in mesh.loop_triangles:
                for loop_index in tri.loops:
                    vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
                    vertex_world = matrix_world @ vertex
                    # Adjusting for THREE.js coordinate system
                    vertices.extend([vertex_world.x, vertex_world.z, -vertex_world.y])
        else:
            # No index, export all vertices directly
            for vertex in mesh.vertices:
                vertex_world = matrix_world @ vertex.co
                # Adjusting for THREE.js coordinate system
                vertices.extend([vertex_world.x, vertex_world.z, -vertex_world.y])

    return vertices


def export_body(filepath):
    vertices = get_body_vertices()

    # Get final bounding box
    bounding_box = derive_bounding_box()
    blend_min_bounds, blend_max_bounds = bounding_box.get_bounds()
    print(blend_min_bounds, blend_max_bounds)
    # Adjusting for THREE.js coordinate system, inverting y not necessary in this case
    min_bounds = [blend_min_bounds.x, blend_min_bounds.z, blend_min_bounds.y]
    max_bounds = [blend_max_bounds.x, blend_max_bounds.z, blend_max_bounds.y]

    package = []
    package.extend(min_bounds)
    package.extend(max_bounds)
    package.extend(vertices)
    vertices_np = np.array(package, dtype=np.float32)

    # Save to file
    with open(filepath, 'wb') as file:
        vertices_np.tofile(file)


if __name__ == "__main__":
    # Specify the file path
    file_path = 'D:/file.bin'

    # Export vertices
    export_body(file_path)

    print(f'Vertices exported to {file_path}')
