import bpy
import mathutils
import math
import numpy as np
from utils import BoundingBox
import os


def transform_vector_around_point(v2, pivot, location=None, rotation=None, scale=None):
    transform_matrix = mathutils.Matrix.LocRotScale(location, rotation, scale)

    # Translate the vector to the origin relative to the pivot point
    translated_v2 = v2 - pivot
    # Apply the transformation
    transformed_translated_v2 = transform_matrix @ translated_v2
    # Translate the vector back to the pivot point
    transformed_v2 = transformed_translated_v2 + pivot

    return transformed_v2


def get_reference_pivot_point(normalized_pivot_point, min_bounds, max_bounds):
    """
    Converts a normalized vector to a position within the min and max bounds.

    Args:
        normalized_pivot_point (mathutils.Vector): Normalized coordinates (0-1).
        min_bounds (mathutils.Vector): Minimum bounds of the bounding box.
        max_bounds (mathutils.Vector): Maximum bounds of the bounding box.

    Returns:
        mathutils.Vector: Resolved position within the bounding box.
    """
    # Ensure the input is within the normalized range
    normalized_pivot_point = mathutils.Vector([min(max(coord, 0.0), 1.0) for coord in normalized_pivot_point])
    # Compute resolved point and return
    size = mathutils.Vector(max_bounds) - mathutils.Vector(min_bounds)

    return mathutils.Vector(min_bounds) + size * normalized_pivot_point
    # disregard min bounds as vertices are already relative to inside of box
    # return size * normalized_pivot_point


def fit_final_values_in_bounding_box(vert_values, joint_values, min_bounds, max_bounds, v_pivot, j_pivot):
    """
    Ensures all vectors within vert_values fit within min_bounds and max_bounds.
    Adjusts joint_values accordingly.

    Args:
        vert_values (list of mathutils.Vector): List of vertex coordinates.
        joint_values (list of mathutils.Vector): List of joint coordinates.
        min_bounds (array-like): Minimum bounds of the bounding box.
        max_bounds (array-like): Maximum bounds of the bounding box.

    Returns:
        (list of mathutils.Vector, list of mathutils.Vector): Adjusted vert_values and joint_values.
    """
    min_bounds = mathutils.Vector(min_bounds)
    max_bounds = mathutils.Vector(max_bounds)
    length_vector = max_bounds - min_bounds
    length_scalar = length_vector.x

    vert_bbox = BoundingBox()
    vert_bbox.expand(vert_values)
    vert_bbox_min, vert_bbox_max = vert_bbox.get_bounds()
    vert_bbox_length_vector = vert_bbox_max - vert_bbox_min

    scale_vector = mathutils.Vector((1.0, 1.0, 1.0))
    for i in range(3):
        if vert_bbox_length_vector[i] > length_vector[i]:
            scale_vector[i] = length_vector[i] / vert_bbox_length_vector[i]

    print('adjusted scale', scale_vector)
    vert_values = [transform_vector_around_point(v, v_pivot, None, None, scale_vector) for v in vert_values]
    joint_values = [transform_vector_around_point(v, j_pivot, None, None, scale_vector) for v in joint_values]

    vert_bbox = BoundingBox()
    vert_bbox.expand(vert_values)
    vert_bbox_min, vert_bbox_max = vert_bbox.get_bounds()
    vert_bbox_length_vector = vert_bbox_max - vert_bbox_min

    # Determine the translation needed to fit vert_values within the bounds
    translation_vector = mathutils.Vector((0, 0, 0))
    for v in vert_values:
        for i in range(3):
            if v[i] < min_bounds[i]:
                translation_vector[i] = min_bounds[i] - v[i]
            if v[i] > max_bounds[i]:
                translation_vector[i] = max_bounds[i] - v[i]

    print('adjusted translation', translation_vector)
    # Translate vert_values
    adjusted_vert_values = [v + translation_vector for v in vert_values]

    # Calculate the normalized translation for joint_values
    normalized_translation_vector = mathutils.Vector((translation_vector[i] / length_vector[i] for i in range(3)))

    # Translate joint_values
    adjusted_joint_values = [v + normalized_translation_vector for v in joint_values]

    return adjusted_vert_values, adjusted_joint_values


def apply_random_transformation_to_vectors(vert_file_path, coords_file_path):
    # Read the values from the file
    values = np.fromfile(vert_file_path, dtype=np.float32)
    # print("Original Values:", values[6:])
    joint_coords = np.fromfile(coords_file_path, dtype=np.float32)

    min_bounds = values[:3]
    max_bounds = values[3:6]
    vert_values = values[6:]

    vectors = [mathutils.Vector(vert_values[i:i+3]) for i in range(0, len(vert_values), 3)]
    joint_vectors = [mathutils.Vector(joint_coords[i:i+3]) for i in range(0, len(joint_coords), 3)]

    # testing: using new first joint location transcribed within bounds as pivot point
    pivot_point = get_reference_pivot_point(joint_vectors[0], min_bounds, max_bounds)
    print('pivot', pivot_point)

    # Define a random transformation
    max_rotation_angle = math.radians(30)  # Max rotation 30 degrees
    min_scale_factor = 0.7
    max_scale_factor = 1.5
    rotation = mathutils.Euler((np.random.uniform(-max_rotation_angle, max_rotation_angle),
                                np.random.uniform(-max_rotation_angle, max_rotation_angle),
                                np.random.uniform(-max_rotation_angle, max_rotation_angle)), 'XYZ')
    scale = mathutils.Vector((np.random.uniform(min_scale_factor, max_scale_factor),
                              np.random.uniform(min_scale_factor, max_scale_factor),
                              np.random.uniform(min_scale_factor, max_scale_factor)))

    print('rotation', rotation)
    print('scale', scale)
    # Apply the transformation on all vectors
    transformed_vectors = [transform_vector_around_point(v, pivot_point, None, rotation, scale) for v in vectors]
    transformed_joints = [transform_vector_around_point(v, joint_vectors[0], None, rotation, scale) for v in joint_vectors]

    transformed_vectors, transformed_joints = fit_final_values_in_bounding_box(
        transformed_vectors,
        transformed_joints,
        min_bounds,
        max_bounds,
        pivot_point,
        joint_vectors[0]
    )

    # Convert the transformed vectors back to a flat float32 array
    transformed_values = np.array([component for vec in transformed_vectors for component in vec], dtype=np.float32)
    transformed_joint_values = np.array([component for vec in transformed_joints for component in vec], dtype=np.float32)

    # return the results, min and max bounds should remain unchanged
    vert_np = np.concatenate((min_bounds, max_bounds, transformed_values))
    print('done')
    return vert_np, transformed_joint_values


def get_max_scale_factors(true_bounds, cube_bounds):
    """
    Returns the maximum scaling factors on each axis before the true bounds of the model touch the cube bounds.

    Args:
        true_bounds (tuple of mathutils.Vector): True minimum and maximum bounds of the model.
        cube_bounds (tuple of mathutils.Vector): Minimum and maximum bounds of the cube.

    Returns:
        mathutils.Vector: Maximum scaling factors on x, y, and z axes.
    """
    def round_down(number, decimals):
        factor = 10 ** decimals
        return np.floor(number * factor) / factor

    true_min, true_max = true_bounds
    cube_min, cube_max = cube_bounds

    # Calculate the dimensions of the true bounds and the cube bounds
    true_dimensions = true_max - true_min
    cube_dimensions = cube_max - cube_min

    # Calculate the distances to the nearest edges of the cube bounds from the true bounds
    distance_min = true_min - cube_min
    distance_max = cube_max - true_max

    factor_min = mathutils.Vector((
        (distance_min.x / abs(cube_min.x)) + 1,
        (distance_min.y / abs(cube_min.y)) + 1 if cube_min.y != 0 else 1,
        (distance_min.z / abs(cube_min.z)) + 1,
    ))
    factor_max = mathutils.Vector((
        (distance_max.x / cube_max.x) + 1,
        (distance_max.y / cube_max.y) + 1,
        (distance_max.z / cube_max.z) + 1,
    ))

    # Calculate the maximum scale factors based on the nearest edges
    max_scale_factors = mathutils.Vector((
        round_down(min(factor_min.x, factor_max.x), 1),
        round_down(max(factor_min.y, factor_max.y), 1),  # y is special case where we want max since we start at 0
        round_down(min(factor_min.z, factor_max.z), 1),
    ))

    return max_scale_factors


def apply_random_scale_transformation(vert_file_path, coords_file_path):
    # Read the values from the file
    values = np.fromfile(vert_file_path, dtype=np.float32)
    # print("Original Values:", values[6:])
    joint_coords = np.fromfile(coords_file_path, dtype=np.float32)

    min_bounds = mathutils.Vector(values[:3])
    max_bounds = mathutils.Vector(values[3:6])
    vert_values = values[6:]

    vectors = [mathutils.Vector(vert_values[i:i+3]) for i in range(0, len(vert_values), 3)]
    joint_vectors = [mathutils.Vector(joint_coords[i:i+3]) for i in range(0, len(joint_coords), 3)]

    true_bbox = BoundingBox()
    true_bbox.expand(vectors)
    true_min_bounds, true_max_bounds = true_bbox.get_bounds()
    # pivot is bottom center
    joint_pivot_point = mathutils.Vector([0.5, 0, 0.5])
    # print('joint pivot', joint_pivot_point)
    vert_pivot_point = mathutils.Vector((
        min_bounds.x + ((max_bounds.x - min_bounds.x) * joint_pivot_point.x),
        min_bounds.y + ((max_bounds.y - min_bounds.y) * joint_pivot_point.y),
        min_bounds.z + ((max_bounds.z - min_bounds.z) * joint_pivot_point.z),
    ))
    # vert_pivot_point = mathutils.Vector((0,0,0))
    # print('vert pivot', vert_pivot_point)

    # Define a random transformation
    min_scale_factor = 0.5
    max_scale_factor = get_max_scale_factors(
        (true_min_bounds, true_max_bounds),
        (min_bounds, max_bounds)
    )

    print('...max scale factors', max_scale_factor)
    scale = mathutils.Vector((np.random.uniform(min_scale_factor, max_scale_factor.x),
                              np.random.uniform(min_scale_factor, max_scale_factor.y),
                              np.random.uniform(min_scale_factor, max_scale_factor.z)))

    print('...scale', scale)
    # Apply the transformation on all vectors
    transformed_vectors = [transform_vector_around_point(v, vert_pivot_point, None, None, scale) for v in vectors]
    transformed_joints = [transform_vector_around_point(v, joint_pivot_point, None, None, scale) for v in joint_vectors]

    # Convert the transformed vectors back to a flat float32 array
    transformed_values = np.array([component for vec in transformed_vectors for component in vec], dtype=np.float32)
    transformed_joint_values = np.array([component for vec in transformed_joints for component in vec], dtype=np.float32)

    # return the results, min and max bounds should remain unchanged
    np_min_bounds = np.array(min_bounds.to_tuple(), dtype=np.float32)
    np_max_bounds = np.array(max_bounds.to_tuple(), dtype=np.float32)
    vert_np = np.concatenate((np_min_bounds, np_max_bounds, transformed_values))
    # print('done')
    return vert_np, transformed_joint_values



def process_files_in_directory(directory):
    # Create the augmented directory if it doesn't exist
    augmented_dir = os.path.join(directory, 'augmented')
    os.makedirs(augmented_dir, exist_ok=True)

    # Find all vrt_hand.bin and crd_hand.bin pairs
    files = os.listdir(directory)
    vrt_files = [f for f in files if f.endswith('.vrt_body.bin')]
    num_files = len(vrt_files)

    for index, vrt_file in enumerate(vrt_files):
        base_name = vrt_file.replace('.vrt_body.bin', '')
        crd_file = base_name + '.crd_body.bin'
        print(f"{index}/{num_files}: {base_name}")

        if crd_file in files:
            vrt_file_path = os.path.join(directory, vrt_file)
            crd_file_path = os.path.join(directory, crd_file)

            for i in range(20):
                augmented_vrt_file = os.path.join(augmented_dir, f"{base_name}.a{i}.vrt_body.bin")
                augmented_crd_file = os.path.join(augmented_dir, f"{base_name}.a{i}.crd_body.bin")
                if os.path.exists(augmented_vrt_file):
                    print("_", end="")
                    continue

                # new_vert_values, new_joint_values = apply_random_transformation_to_vectors(vrt_file_path, crd_file_path)
                new_vert_values, new_joint_values = apply_random_scale_transformation(vrt_file_path, crd_file_path)

                # Save transformed values
                new_vert_values.tofile(augmented_vrt_file)
                new_joint_values.tofile(augmented_crd_file)
                print("#", end="")
                # print(f"written: {augmented_vrt_file}")
                # print(f"written: {augmented_crd_file}")


if __name__ == "__main__":
    dataset_directory = "D:\\HEAT Dataset\\processed_body"
    process_files_in_directory(dataset_directory)

    # # process single file
    # vrt_file_path = "D:\\HEAT Dataset\processed_body\\5cd4a778a4ba439c94af38fb2c5aab82.vrt_body.bin"
    # crd_file_path = "D:\\HEAT Dataset\processed_body\\5cd4a778a4ba439c94af38fb2c5aab82.crd_body.bin"
    # new_vert_values, new_joint_values = apply_random_scale_transformation(vrt_file_path, crd_file_path)
    #
    # # Save transformed values
    # new_vert_values.tofile("D:\\HEAT Dataset\processed_body\\5cd4a778a4ba439c94af38fb2c5aab82.a.vrt_body.bin")
    # new_joint_values.tofile("D:\\HEAT Dataset\processed_body\\5cd4a778a4ba439c94af38fb2c5aab82.a.crd_body.bin")
