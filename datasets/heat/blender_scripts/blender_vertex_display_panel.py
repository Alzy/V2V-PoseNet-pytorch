import bpy
import bmesh
from mathutils import Vector
from bpy.types import Panel
from bpy.utils import register_class, unregister_class

class VIEW3D_PT_DisplayVertexCoords(Panel):
    """Panel to display the global coordinates of selected vertices"""
    bl_label = "Vertex Global Coordinates"
    bl_idname = "VIEW3D_PT_display_vertex_coords"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Item'

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)
        matrix_world = obj.matrix_world

        cube = bpy.context.scene.objects.get("Cube")
        cube_matrix_world = cube.matrix_world
        cube_bounds_min = cube_matrix_world @ Vector(cube.bound_box[0])
        cube_bounds_max = cube_matrix_world @ Vector(cube.bound_box[6])
        cube_bottom_center = (cube_bounds_min + cube_bounds_max) / 2
        cube_bottom_center.z = cube_bounds_min.z  # Adjust to the bottom center

        layout.label(text="Cube Bounding Box: ")
        layout.label(text=f"({cube_bounds_min.x:.3f}, {cube_bounds_min.y:.3f}, {cube_bounds_min.z:.3f})")
        layout.label(text=f"({cube_bounds_max.x:.3f}, {cube_bounds_max.y:.3f}, {cube_bounds_max.z:.3f})")
        layout.label(text="Cube Bounding Box Bottom Center: ")
        layout.label(text=f"({cube_bottom_center.x:.3f}, {cube_bottom_center.y:.3f}, {cube_bottom_center.z:.3f})")

        for vert in bm.verts:
            if vert.select:
                global_coord = matrix_world @ vert.co
                layout.label(text=f"Vertex {vert.index}")
                layout.label(text=f"({global_coord.x:.3f}, {global_coord.y:.3f}, {global_coord.z:.3f})")

def register():
    register_class(VIEW3D_PT_DisplayVertexCoords)

def unregister():
    unregister_class(VIEW3D_PT_DisplayVertexCoords)

if __name__ == "__main__":
    register()
