import bpy
import sys
import os

# Add the directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Import methods from current directory packages
from export_body import export_body
from export_bone_coordinates import export_bone_coordinates

blender_files_dir = 'D:\HEAT Dataset'
export_files_dir = 'D:\HEAT Dataset\processed_body'


def get_current_blender_file_id():
    # Get the full path of the current file
    file_path = bpy.data.filepath
    # Use basename to get just the file name from the full path
    file_name = bpy.path.basename(file_path)
    # Split the file name at the first period and take the first part
    file_id = file_name.split('.')[0]
    return file_id


def get_blend_files(directory):
    blend_files_tmp = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.blend'):
                file_dir = os.path.join(root, file)
                blend_files_tmp.append(file_dir)
                # print(os.path.join(root, file))

    return blend_files_tmp


def process_blend_file(blend_file_path, export_file_path):
    print(f"Opened .blend file: {blend_file_path}")
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    current_file_id = get_current_blender_file_id()
    print("Current File ID:", current_file_id)

    vertices_file_path = os.path.join(export_file_path, f"{current_file_id}.vrt_body.bin")
    coords_file_path = os.path.join(export_file_path, f"{current_file_id}.crd_body.bin")

    if os.path.exists(vertices_file_path) and os.path.exists(coords_file_path):
        print('-- already processed. Continuing...')
        return

    export_body(vertices_file_path)
    export_bone_coordinates(coords_file_path)


if __name__ == "__main__":
    # blend_files = get_blend_files(blender_files_dir)
    blend_files = [
        "D:\HEAT Dataset\\fe1dd3b831eb44a2a2e9e00af931f265.glb.blend"
    ]

    os.makedirs(export_files_dir, exist_ok=True)
    for blend_file in blend_files:
        process_blend_file(blend_file, export_files_dir)
    print('Done.')
