// voxelizer.hlsl
// Buffer containing vertices of the mesh
StructuredBuffer<float> vertices : register(t0);
// A read-write buffer for the voxel grid
RWStructuredBuffer<uint> voxelGrid : register(u0);
// A read buffer for voxelization params of the form [voxelSize, gridXSize, gridYSize, gridZSize]
StructuredBuffer<float> params : register(t1);

struct Vector3 {
    float x;
    float y;
    float z;
};

struct Box3 {
    Vector3 min;
    Vector3 max;
};

Box3 createBox3() {
    Box3 box;
    box.min = Vector3(1e8, 1e8, 1e8);
    box.max = Vector3(-1e8, -1e8, -1e8);
    return box;
}

void expandBox3(inout Box3 box, Vector3 vertex) {
    box.min.x = min(box.min.x, vertex.x);
    box.min.y = min(box.min.y, vertex.y);
    box.min.z = min(box.min.z, vertex.z);
    box.max.x = max(box.max.x, vertex.x);
    box.max.y = max(box.max.y, vertex.y);
    box.max.z = max(box.max.z, vertex.z);
}

bool Box3IntersectsBox3(Box3 boxA, Box3 boxB) {
    return !(
        boxB.max.x < boxA.min.x ||
        boxB.min.x > boxA.max.x ||
        boxB.max.y < boxA.min.y ||
        boxB.min.y > boxA.max.y ||
        boxB.max.z < boxA.min.z ||
        boxB.min.z > boxA.max.z
    );
}

[numthreads(64, 1, 1)]
void main(uint3 global_id : SV_DispatchThreadID) {
    float voxelSize = params[0];
    uint gridXSize = (uint)params[1];
    uint gridYSize = (uint)params[2];
    uint gridZSize = (uint)params[3];

    // Each thread processes one voxel
    uint numVertices = vertices.Length;
    float halfVoxelSize = voxelSize * 0.5;

    // Calculate the index of the voxel this thread is responsible for
    uint voxelIndex = global_id.x + global_id.y * gridXSize + global_id.z * (gridXSize * gridYSize);
    if (voxelIndex >= gridXSize * gridYSize * gridZSize) {
        return; // out of bounds
    }

    float voxelX = (float)(voxelIndex % gridXSize);
    float voxelY = (float)((voxelIndex / gridXSize) % gridYSize);
    float voxelZ = (float)(voxelIndex / (gridXSize * gridYSize));

    // Construct voxel bounding box
    Box3 voxelBox = createBox3();

    float startX = (float)gridXSize * 0.5 * voxelSize - halfVoxelSize;
    float startZ = (float)gridZSize * 0.5 * voxelSize - halfVoxelSize;

    float minX = -startX + (voxelX * voxelSize) - halfVoxelSize;
    float minY = voxelY * voxelSize;
    float minZ = -startZ + (voxelZ * voxelSize) - halfVoxelSize;

    float maxX = -startX + (voxelX * voxelSize) + halfVoxelSize;
    float maxY = minY + voxelSize;
    float maxZ = -startZ + (voxelZ * voxelSize) + halfVoxelSize;

    Vector3 minVoxelX = Vector3(minX, minY, minZ);
    Vector3 maxVoxelX = Vector3(maxX, maxY, maxZ);
    expandBox3(voxelBox, minVoxelX);
    expandBox3(voxelBox, maxVoxelX);

    // Check if any of the triangles intersect this voxel
    for (uint i = 0; i < numVertices; i += 9) {
        // Construct triangle bounding box
        Box3 triangleBox = createBox3();

        Vector3 pointA = Vector3(vertices[i], vertices[i+1], vertices[i+2]);
        Vector3 pointB = Vector3(vertices[i+3], vertices[i+4], vertices[i+5]);
        Vector3 pointC = Vector3(vertices[i+6], vertices[i+7], vertices[i+8]);

        expandBox3(triangleBox, pointA);
        expandBox3(triangleBox, pointB);
        expandBox3(triangleBox, pointC);

        // If any triangle intersects, set voxel as active
        if (Box3IntersectsBox3(triangleBox, voxelBox)) {
            voxelGrid[voxelIndex] = 1;
            break; // Done processing this voxel
        }
    }
}
