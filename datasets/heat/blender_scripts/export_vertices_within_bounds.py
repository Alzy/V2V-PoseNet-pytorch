import bpy
import numpy as np
from mathutils import Vector


def export_vertices_within_bounds(filepath):
    scene = bpy.context.scene
    vertices = []
    model_bounds = []

    # Find the "Cube" object
    cube = bpy.context.scene.objects.get("Cube")
    if not cube:
        print("Cube not found in the scene.")
        exit()

    cube_matrix_world = cube.matrix_world
    cube_bounds_min = cube_matrix_world @ Vector(cube.bound_box[0])
    cube_bounds_max = cube_matrix_world @ Vector(cube.bound_box[6])
    cube_bottom_center = (cube_bounds_min + cube_bounds_max) / 2
    cube_bottom_center.z = cube_bounds_min.z  # Adjust to the bottom center
    model_bounds.extend([cube_bounds_min.x, cube_bounds_min.y, cube_bounds_min.z])
    model_bounds.extend([cube_bounds_max.x, cube_bounds_max.y, cube_bounds_max.z])

    for obj in scene.objects:
        if obj.type == 'MESH' and obj.name != cube.name:
            obj.data.calc_loop_triangles()
            mesh = obj.data
            matrix_world = obj.matrix_world

            if mesh.loop_triangles:
                # Using indexed vertices
                for tri in mesh.loop_triangles:
                    triangle_vertices = []
                    all_within_bounds = True
                    for loop_index in tri.loops:
                        vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
                        vertex_world = matrix_world @ vertex
                        # Check if the vertex is within the bounds
                        if not (cube_bounds_min.x <= vertex_world.x <= cube_bounds_max.x and
                                cube_bounds_min.y <= vertex_world.y <= cube_bounds_max.y and
                                cube_bounds_min.z <= vertex_world.z <= cube_bounds_max.z):
                            all_within_bounds = False
                            break
                        # Adjusting for THREE.js coordinate system and making coordinates relative
                        adjusted_vertex = vertex_world - cube_bottom_center
                        triangle_vertices.extend([adjusted_vertex.x, adjusted_vertex.z, -adjusted_vertex.y])

                    if all_within_bounds:
                        vertices.extend(triangle_vertices)
            else:
                # No index, export all vertices directly
                for vertex in mesh.vertices:
                    vertex_world = matrix_world @ vertex.co
                    if cube_bounds_min.x <= vertex_world.x <= cube_bounds_max.x and \
                       cube_bounds_min.y <= vertex_world.y <= cube_bounds_max.y and \
                       cube_bounds_min.z <= vertex_world.z <= cube_bounds_max.z:
                        # Adjusting for THREE.js coordinate system and making coordinates relative
                        adjusted_vertex = vertex_world - cube_bottom_center
                        vertices.extend([adjusted_vertex.x, adjusted_vertex.z, -adjusted_vertex.y])

    # Convert to numpy array with dtype float32
    package = []
    package.extend(model_bounds)
    package.extend(vertices)
    vertices_np = np.array(package, dtype=np.float32)

    # Save to file
    with open(filepath, 'wb') as file:
        vertices_np.tofile(file)
        print(f'Vertices exported to {filepath}')


if __name__ == "__main__":
    # Specify the file path
    file_path = 'D:/file_1.vrt_hand.bin'

    # Export vertices
    export_vertices_within_bounds(file_path)
