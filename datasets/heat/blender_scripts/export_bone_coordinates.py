import bpy


def export_bone_world_coordinates():
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

    matrix_world = armature.matrix_world
    bone_data = []

    for bone in armature.pose.bones:
        # Calculate the world position of the head and tail of each bone
        head_world = matrix_world @ bone.head
        tail_world = matrix_world @ bone.tail
        bone_data.append((bone.name, head_world, tail_world))

        # Print the world coordinates of each bone
        print(f"Bone: {bone.name}")
        print(f"  Head: ({head_world.x:.3f}, {head_world.y:.3f}, {head_world.z:.3f})")
        print(f"  Tail: ({tail_world.x:.3f}, {tail_world.y:.3f}, {tail_world.z:.3f})\n")

    return bone_data

export_bone_world_coordinates()
