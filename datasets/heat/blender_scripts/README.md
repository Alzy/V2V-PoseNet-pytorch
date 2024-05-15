# Blender Training Export Scripts

The scripts in this folder provide methods for generating test data from blend files.

The .bin files exported by these scripts contain f32 array data with the following structure:

* First 3 f32 values are the minimum x,y,z bounds 

* Next 3 f32 values are the maximum x,y,z bounds 

* Subsequent data is the model vertex data

The minimum and maximum bound data is useful for calculating the voxel size by the voxelizer. (longest side/voxel density)