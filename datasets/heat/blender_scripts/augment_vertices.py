import bpy
import mathutils
import math
import numpy as np


def transform_vector_around_point(v2, pivot, location=None, rotation=None, scale=None):
    transform_matrix = mathutils.Matrix.LocRotScale(location, rotation, scale)

    # Translate the vector to the origin relative to the pivot point
    translated_v2 = v2 - pivot
    # Apply the transformation
    transformed_translated_v2 = transform_matrix @ translated_v2
    # Translate the vector back to the pivot point
    transformed_v2 = transformed_translated_v2 + pivot

    return transformed_v2


def apply_random_transformation_to_vectors(file_path):
    # Read the values from the file
    values = np.fromfile(file_path, dtype=np.float32)

    min_bounds = values[:3]
    max_bounds = values[3:6]
    vert_values = values[6:]

    vectors = [mathutils.Vector(vert_values[i:i+3]) for i in range(0, len(vert_values), 3)]

    # The first vector is assigned as the pivot point
    pivot_point = vectors[0]

    # Define a random transformation
    max_rotation_angle = math.radians(30)  # Max rotation 30 degrees
    max_scale_factor = 1.5  # Max scale 1.5
    rotation = mathutils.Euler((np.random.uniform(-max_rotation_angle, max_rotation_angle),
                                np.random.uniform(-max_rotation_angle, max_rotation_angle),
                                np.random.uniform(-max_rotation_angle, max_rotation_angle)), 'XYZ')
    scale = mathutils.Vector((np.random.uniform(1.0, max_scale_factor),
                              np.random.uniform(1.0, max_scale_factor),
                              np.random.uniform(1.0, max_scale_factor)))

    # Apply the transformation on all vectors
    transformed_vectors = [transform_vector_around_point(v, pivot_point, None, rotation, scale) for v in vectors]

    # Convert the transformed vectors back to a flat float32 array
    transformed_values = np.array([component for vec in transformed_vectors for component in vec], dtype=np.float32)

    # return the results, min and max bounds should remain unchanged
    return np.concatenate((min_bounds, max_bounds, transformed_values))


if __name__ == "__main__":
    vrt_file_path = "D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.vrt_hand.bin"
    new_values = apply_random_transformation_to_vectors(vrt_file_path)
    print("Transformed Values:", new_values)

    # Save transformed values
    new_values.tofile("D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.a.vrt_hand.bin")
