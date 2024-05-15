import bpy
from mathutils import Vector


def get_cube_bounds():
    cube = bpy.context.scene.objects.get("Cube")
    if not cube:
        return None
    cube_matrix_world = cube.matrix_world
    cube_bounds_min = cube_matrix_world @ Vector(cube.bound_box[0])
    cube_bounds_max = cube_matrix_world @ Vector(cube.bound_box[6])
    return {
        "min": cube_bounds_min,
        "max": cube_bounds_max
    }


def get_normalized_position_in_cube(position: Vector):
    bounds = get_cube_bounds()
    if not bounds:
        print("Cube not found.")
        return None

    # Get min and max bounds
    min_bound = bounds['min']
    max_bound = bounds['max']

    # Normalize the position within the cube where top back left is (0,0,0)
    # and the bottom front right is (1,1,1)
    normalized_x = (position.x - min_bound.x) / (max_bound.x - min_bound.x)
    normalized_y = (position.y - min_bound.y) / (max_bound.y - min_bound.y)
    normalized_z = (position.z - min_bound.z) / (max_bound.z - min_bound.z)

    # Return normalized coordinates with respect to the top back left
    # Top back left would be (0,1,0) in Blender's coordinate system (Y-down, Z-forward)
    print("  World Head:", position.x, position.y, position.z)
    return Vector((normalized_x, 1 - normalized_y, 1 - normalized_z))



def export_bone_world_coordinates():
    armature = None
    # Find the first armature in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        print("No armature found in the scene.")
        return None

    matrix_world = armature.matrix_world
    bone_data = []

    # Find the specific bone "heat_Hand_l"
    target_bone = armature.pose.bones.get("heat_Hand_l")
    if not target_bone:
        print("Bone 'heat_Hand_l' not found in the armature.")
        return None

    # Calculate the world position of the head of the target bone
    target_head_world = matrix_world @ target_bone.head

    # Function to traverse and collect data
    def collect_bone_data(bone, target_head_world):
        # Calculate the world position of the bone's head
        head_world = matrix_world @ bone.head
        # Calculate relative position to the target bone's head
        relative_head = head_world - target_head_world

        # Store the relative world coordinates of the bone's head
        bone_data.append((bone.name, relative_head))

        # Print the relative world coordinates of the bone's head
        print(f"Bone: {bone.name}")
        print(f"  Relative Head: ({relative_head.x:.3f}, {relative_head.y:.3f}, {relative_head.z:.3f})")
        norm_pos_in_cube = get_normalized_position_in_cube(head_world)
        print(f"  Normal Point in Cube: {norm_pos_in_cube}\n")

        # Recursively collect data for child bones
        for child in bone.children:
            collect_bone_data(child, target_head_world)

    # Start the data collection from the target bone and its children
    collect_bone_data(target_bone, target_head_world)

    return bone_data

export_bone_world_coordinates()
print(get_cube_bounds())
