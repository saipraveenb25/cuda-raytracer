#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif
//#include "application.h"
#include "bsdf.h"
#include "ray.h"

#include <stack>
#include <random>
#include <algorithm>

#include "CMU462/CMU462.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"
#include "CMU462/lodepng.h"

#include "GL/glew.h"

#include "static_scene/sphere.h"
#include "static_scene/triangle.h"
#include "static_scene/light.h"

// COLLADA
#include "collada/collada.h"
#include "collada/light_info.h"
#include "collada/sphere_info.h"
#include "collada/polymesh_info.h"
#include "collada/material_info.h"

#include "dynamic_scene/ambient_light.h"
#include "dynamic_scene/environment_light.h"
#include "dynamic_scene/directional_light.h"
#include "dynamic_scene/area_light.h"
#include "dynamic_scene/point_light.h"
#include "dynamic_scene/spot_light.h"
#include "dynamic_scene/sphere.h"
#include "dynamic_scene/mesh.h"
#include "dynamic_scene/widgets.h"
#include "dynamic_scene/skeleton.h"
#include "dynamic_scene/joint.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


#include "cuda_image.h"
#include "bvh.h"

#define TREE_WIDTH 8
#define RAYS_PER_BLOCK 128
#define RAYS_PER_BLOCK_LOG2 7
#define QUEUE_LENGTH_LOG2 18
#define LEVEL_INDEX_SIZE 4096
#define MAX_LEVELS 10
#define SAMPLES_PER_PIXEL 4
#define MAX_TRIANGLES 128
#define MAX_T_DISTANCE 10000.0
#define MAX_INTERSECTIONS 4

namespace cutracer {
struct CuRay {
    float3 o;
    float3 d;
    float3 importance; // Importance weight for this ray at this point.
    float3 light; // Direct light that is passing through this point so far.
    float3 lightImportance; // Non-zero for a light-intersection connection.
    float2 ss;  // Screen Space coordinates for update.
    float maxT; // Maximum length.
    int sid; // Sample ID.
    int id; // Ray ID.

    // Copied from CuIntersection. 
    // To allow easy direct light estimation.
    float3 n;
    float t;
    float3 wi;
};

struct CuTriangle {
    float3 a;
    float3 b;
    float3 c;

    // TODO:  Check if necessary. Edge normals.
    float3 n0;
    float3 n1;
    float3 n2;

    int bsdf; // BSDF index.
    int emit; // Emitter index.
};

struct CuEmitter {
    float3 radiance;
    float3 position;
    float3 direction;
    float3 dim_x;
    float3 dim_y;
    float area;
};

struct CuBSDF{
    int fn; // 0 - diffuse, 1 - specular.
    float3 albedo;  // For diffuse.
    float nu;       // For specular.
};

struct CuBVHSubTree { 
    uint64_t outlets[TREE_WIDTH];
    
    uint64_t start;
    uint64_t range;
    
    float3 minl[TREE_WIDTH];
    float3 maxl[TREE_WIDTH];
};

struct CuIntersection {
    float3 pt;
    float3 n;
    float3 light;
    float3 wi;
    float3 importance;
    float t;
    float2 ss;
    int sid;
    int bsdf;
    int id;
    int is_new; // Waiting for the next overwrite.
    bool valid;
}; 

class CudaRenderer {

private:

    Image* image;
    //SceneName sceneName;

    // Device pointers.
    CuBSDF* deviceBSDFs;
    CuEmitter* deviceEmitters;
    CuTriangle* deviceTriangles;
    CuBVHSubTree* deviceBVHSubTrees;
    CuIntersection* deviceIntersections;
    CuRay* deviceRays;
    int* deviceLevelIndices;
    float* deviceImageData;
    float* deviceSSImageData;
    uint* deviceQueueCounts;
    float* deviceMinT;
    uint* deviceIntersectionTokens;
    CuIntersection* deviceMultiIntersections;

    // Host structures.
    std::vector<CuBSDF> bsdfs;
    std::vector<CuEmitter> emitters;
    std::vector<CuTriangle> triangles;
    std::vector<CuBVHSubTree> subtrees;
    int* levelIndices;
    std::vector<int> levelCounts;
    
    
    // Camera data.
    Vector3D c_origin;
    Vector3D c_lookAt;
    Vector3D c_up;
    Vector3D c_left;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(std::string name);

    void allocOutputImage(int width, int height);

    void clearImage();
    
    void clearIntersections();

    void advanceAnimation();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
    
    DynamicScene::Scene* loadFromSceneInfo(Collada::SceneInfo* sceneInfo);
};

}
#endif