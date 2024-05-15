import bpy

def get_current_blender_file_id():
    # Get the full path of the current file
    file_path = bpy.data.filepath
    # Use basename to get just the file name from the full path
    file_name = bpy.path.basename(file_path)
    # Split the file name at the first period and take the first part
    file_id = file_name.split('.')[0]
    return file_id

# Example usage:
current_file_id = get_current_blender_file_id()
print("Current File ID:", current_file_id)
