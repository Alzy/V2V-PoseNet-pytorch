import bpy
import numpy as np

def export_vertices(filepath):
    scene = bpy.context.scene
    vertices = []
    for obj in scene.objects:
        if obj.type == 'MESH':
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

    # Convert to numpy array with dtype float32
    vertices_np = np.array(vertices, dtype=np.float32)

    # Save to file
    with open(filepath, 'wb') as file:
        vertices_np.tofile(file)

# Specify the file path
file_path = 'D:/file.bin'

# Export vertices
export_vertices(file_path)

print(f'Vertices exported to {file_path}')
