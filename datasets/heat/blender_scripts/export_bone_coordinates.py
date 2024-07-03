import bpy
import numpy as np
from utils import BoundingBox, derive_bounding_box
from mathutils import Vector


def get_normalized_position_in_bounds(position: Vector, min_bound, max_bound):
    # Normalize the position within the cube where bottom back left is (0,0,0)
    # and the top front right is (1,1,1)
    normalized_x = (position.x - min_bound.x) / (max_bound.x - min_bound.x)
    normalized_y = (position.y - min_bound.y) / (max_bound.y - min_bound.y)
    normalized_z = (position.z - min_bound.z) / (max_bound.z - min_bound.z)

    # Return normalized coordinates with respect to the bottom back left
    return Vector((normalized_x, normalized_z, 1 - normalized_y))


def get_normalized_bone_coordinates():
    armature = None
    # Find the first armature in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    # Export bone coordinates if an armature is found
    if 'armature' in locals():
        pass
    else:
        print("No armature found in the scene.")
        return None

    bounding_box = derive_bounding_box()
    min_bounds, max_bounds = bounding_box.get_bounds()
    matrix_world = armature.matrix_world
    bone_data = {}

    for bone in armature.pose.bones:
        # Calculate the world position of the head and tail of each bone
        head_world = matrix_world @ bone.head
        norm_pos_in_cube = get_normalized_position_in_bounds(head_world, min_bounds, max_bounds)

        bone_data[bone.name] = norm_pos_in_cube

        # Print the world coordinates of each bone
        # print(f"Bone: {bone.name}")
        # print(f"  Head: ({head_world.x:.3f}, {head_world.y:.3f}, {head_world.z:.3f})")

    return bone_data


def export_bone_coordinates(bone_filename):
    bone_coordinates = get_normalized_bone_coordinates()

    package = []
    package_order = [
        'heat_Hips',
        'heat_Spine',
        'heat_Spine1',
        'heat_Spine2',
        'heat_Neck',
        'heat_Head',
        'heat_HeadTop_End',
        'heat_Shoulder_l',
        'heat_Shoulder_r',
        'heat_UpperArm_l',
        'heat_UpperArm_r',
        'heat_LowerArm_l',
        'heat_LowerArm_r',
        'heat_Hand_l',
        'heat_Hand_r',
        'heat_UpLeg_l',
        'heat_UpLeg_r',
        'heat_Leg_l',
        'heat_Leg_r',
        'heat_Foot_l',
        'heat_Foot_r',
        'heat_ToeBase_l',
        'heat_ToeBase_r',
    ]

    # reorder bone map
    for bone in package_order:
        if bone_coordinates[bone]:
            _bone_coordinates = bone_coordinates[bone]
            package.extend([_bone_coordinates.x, _bone_coordinates.y, _bone_coordinates.z])

    # write to file
    with open(bone_filename, "wb") as file:
        coordinates_np = np.array(package, dtype=np.float32)
        coordinates_np.tofile(file)
        print(f'Coordinates exported to {bone_filename}')


if __name__ == "__main__":
    file_path = 'D:/file.crd.bin'
    export_bone_coordinates(file_path)
