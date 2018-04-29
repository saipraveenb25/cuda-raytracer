#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "cuda_image.h"
//#include "noise.h"
//#include "sceneLoader.h"
//#include "util.h"
//#include "cycleTimer.h"
#define SCAN_BLOCK_DIM 1024  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"

#include "cuda_util.h"

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

namespace cutracer {

float3 v2f3(Vector3D v) {
    return make_float3(v.x, v.y, v.z);
}

Vector3D f32v(float3 f) {
    return Vector3D(f.x, f.y, f.z);
}
/*struct CuRay {
    float3 o;
    float3 d;
    float3 importance; // Importance weight for this ray at this point.
    float3 light; // Direct light that is passing through this point so far.
    float3 lightImportance; // Non-zero for a light-intersection connection.
    float2 ss;  // Screen Space coordinates for update.
    float maxt; // Maximum length.
    int sid; // Sample ID.
};

struct CuTriangle {
    float3 a;
    float3 b;
    float3 c;

    // TODO:  Check if necessary. Edge normals.
    float3 n1;
    float3 n2;
    float3 n3;

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
    
    uint64_t outlets[16];
    
    uint64_t start;
    uint64_t range;
    
    float3 minl[16];
    float3 maxl[16];
};

struct CuIntersection {
    float3 pt;
    float3 n;
    float3 light;
    float3 wi;
    float3 importance;
    float t;
    float maxT;
    float2 ss;
    int sid;
    int bsdf;
};*/ 
void init_camera(Collada::CameraInfo &cameraInfo,
                              const Matrix4x4 &transform) {
  //camera.configure(cameraInfo, screenW, screenH);
  //canonicalCamera.configure(cameraInfo, screenW, screenH);
  //set_projection_matrix();
}

DynamicScene::SceneLight *init_light(Collada::LightInfo &light,
                                                  const Matrix4x4 &transform) {
  switch (light.light_type) {
    case Collada::LightType::NONE:
      break;
    case Collada::LightType::AMBIENT:
      return new DynamicScene::AmbientLight(light);
    case Collada::LightType::DIRECTIONAL:
      return new DynamicScene::DirectionalLight(light, transform);
    case Collada::LightType::AREA:
      return new DynamicScene::AreaLight(light, transform);
    case Collada::LightType::POINT:
      return new DynamicScene::PointLight(light, transform);
    case Collada::LightType::SPOT:
      return new DynamicScene::SpotLight(light, transform);
    default:
      break;
  }
  return nullptr;
}

/**
 * The transform is assumed to be composed of translation, rotation, and
 * scaling, where the scaling is uniform across the three dimensions; these
 * assumptions are necessary to ensure the sphere is still spherical. Rotation
 * is ignored since it's a sphere, translation is determined by transforming the
 * origin, and scaling is determined by transforming an arbitrary unit vector.
 */
DynamicScene::SceneObject *init_sphere(
    Collada::SphereInfo &sphere, const Matrix4x4 &transform) {
  const Vector3D &position = (transform * Vector4D(0, 0, 0, 1)).projectTo3D();
  double scale = (transform * Vector4D(1, 0, 0, 0)).to3D().norm();
  return new DynamicScene::Sphere(sphere, position, scale);
}

DynamicScene::SceneObject *init_polymesh(
    Collada::PolymeshInfo &polymesh, const Matrix4x4 &transform) {
  return new DynamicScene::Mesh(polymesh, transform);
}

struct GlobalConstants {

    // Image data.
    int imageWidth;
    int imageHeight;
    float4* ssImageData; // Super sampled image data.
    float4* imageData; // Final image data.
    int sampleCount;

    // Camera data.
    float3 c_origin;
    float3 c_lookAt;
    float3 c_up;
    float3 c_left;

    // Ray queues: SxR (S=Number of subtrees, R=Max number of rays per queue) 
    // (Gigantic ~32M entries)
    CuRay* queues;

    // Queue counts. Initialized to 0.
    uint* qCounts;

    // Ray intersection buffers. (Large 1K-32K entries)
    CuIntersection* intersections;

    // Triangle list. (Huge 1000-1000000 entries)
    CuTriangle* triangles;

   // Emitter list. (Tiny 1-2 entries)
    CuEmitter* emitters;

    // BSDF list. (Small ~10 entries).
    CuBSDF* bsdfs;

    // BVHNode list
    // (Large 10-4000 entries)
    CuBVHSubTree* bvhSubTrees;

    int* levelIndices;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

__device__ __inline__ void cudaswap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ __inline__ float intersectBBox(float3 o, float3 d, float3 min, float3 max) {
    float tmin = (min.x - o.x) / d.x; 
    float tmax = (max.x - o.x) / d.x; 

    if (tmin > tmax) cudaswap(tmin, tmax); 

    float tymin = (min.y - o.y) / d.y; 
    float tymax = (max.y - o.y) / d.y; 

    if (tymin > tymax) cudaswap(tymin, tymax); 

    if ((tmin > tymax) || (tymin > tmax)) 
        return -1; 

    if (tymin > tmin) 
        tmin = tymin; 

    if (tymax < tmax) 
        tmax = tymax; 

    float tzmin = (min.z - o.z) / d.z; 
    float tzmax = (max.z - o.z) / d.z; 

    if (tzmin > tzmax) cudaswap(tzmin, tzmax); 

    if ((tmin > tzmax) || (tzmin > tmax)) 
        return -1;

    if (tzmin > tmin) 
        tmin = tzmin; 

    if (tzmax < tmax) 
        tmax = tzmax; 

    return tmin;

}

/*__device__ __inline__ float3 crossProduct(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x - b.z, a.x * b.y - a.y * b.x);
  }

  __device__ __inline__ float dotProduct(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
  }*/

__device__ __inline__ float intersectRayTriangle(float3 v0, float3 v1, float3 v2, float3 orig, float3 dir) {
    //float e1[3],e2[3],h[3],s[3],q[3];
    //float a,f,u,v;
    //vector(e1,v1,v0);
    //vector(e2,v2,v0);
    // compute plane's normal
    float3 v0v1 = v1 - v0; 
    float3 v0v2 = v2 - v0; 
    // no need to normalize
    float3 N = cross(v0v1, v0v2); // N 
    float area2 = length(N);

    // Step 1: finding P

    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N, dir); 
    if (abs(NdotRayDirection) < 1e-5) // almost 0 
        return -1; // they are parallel so they don't intersect ! 

    // compute d parameter using equation 2
    float d = dot(N, v0); 

    // compute t (equation 3)
    float t = (dot(N, orig) + d) / NdotRayDirection; 

    // check if the triangle is in behind the ray
    if (t < 0) return -1; // the triangle is behind 

    // compute the intersection point using equation 1
    float3 P = orig + t * dir; 

    // Step 2: inside-outside test
    float3 C; // vector perpendicular to triangle's plane 

    // edge 0
    float3 edge0 = v1 - v0;
    float3 vp0 = P - v0;
    C = cross(edge0, vp0);
    if (dot(N, C) < 0) return -1; // P is on the right side 

    // edge 1
    float3 edge1 = v2 - v1; 
    float3 vp1 = P - v1; 
    C = cross(edge1, vp1); 
    if (dot(N, C) < 0)  return -1; // P is on the right side 

    // edge 2
    float3 edge2 = v0 - v2; 
    float3 vp2 = P - v2; 
    C = cross(edge2, vp2); 
    if (dot(N, C) < 0) return -1; // P is on the right side; 

    return t; // this ray hits the triangle 
}

__global__ void kernelClearBuffers(float3* positions, float* radii, float3* colors, int* queues) {
    //int idx = threadIdx.x;
    //int block = blockIdx.x;
    //int layer = blockIdx.y;

    //positions[idx + blockDim.x * block + blockDim.x * gridDim.x * layer] = make_float3(0.f,0.f,0.f);
    //radii[idx + blockDim.x * block + blockDim.x * gridDim.x * layer] = 0.f;
    //colors[idx + blockDim.x * block + blockDim.x * gridDim.x * layer] = make_float3(0.f,0.f,0.f);
    //queues[idx + blockDim.x * block + blockDim.x * gridDim.x * layer] = 0;
}
// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}



// PT functions.

// Generate camera rays and insert into queue.
__global__ void kernelPrimaryRays( ) {

    // For each pixel
    // For each sample per pixel
    // Create ray and set to queue at the falling icicomputed offset.

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    //int raycount = cuConstRendererParams.rayCount;


    int sampleCount = cuConstRendererParams.sampleCount;

    for(int i = 0; i < sampleCount; i++) {
        int destIndex = i + (imageX * height + imageY) * sampleCount;

        float2 sample = make_float2(0.5f, 0.5f);

        float xs = imageX + sample.x;
        float ys = imageY + sample.y;

        float2 ss = make_float2(xs, ys);

        float a = (ss.x / width) - 0.5;
        float b = (ss.y / height) - 0.5;
        float c = sqrt(1 - (a*a + b*b));

        float px = a * cuConstRendererParams.c_left.x + b * cuConstRendererParams.c_up.x + c * cuConstRendererParams.c_lookAt.x;
        float py = a * cuConstRendererParams.c_left.y + b * cuConstRendererParams.c_up.y + c * cuConstRendererParams.c_lookAt.y;
        float pz = a * cuConstRendererParams.c_left.z + b * cuConstRendererParams.c_up.z + c * cuConstRendererParams.c_lookAt.z;

        CuRay *r = &cuConstRendererParams.queues[destIndex];
        r->o = cuConstRendererParams.c_origin;
        r->d = make_float3(px, py, pz);
        r->importance = make_float3(1, 1, 1);
        r->lightImportance = make_float3(0, 0, 0);
        r->light = make_float3(0, 0, 0);
        r->maxT = INFINITY; // TODO: Put inf.
        r->ss = ss;
        r->sid = i;
    }

}

// Generate direct light rays from intersections.
__global__ void kernelDirectLightRays() {
    // For each element in intersection. (Map each intersection to a thread).
    // For each light
    // Create a ray from light to intersection.
    // Add ray to queue.

    int iid = blockIdx.x * blockDim.x + threadIdx.x;

    CuIntersection *its = &cuConstRendererParams.intersections[iid];

    CuEmitter *e = &cuConstRendererParams.emitters[0];

    // Generate the sample.
    float sampleX;
    float sampleY;

    //float lX = (sampleX * e->minl.x + (1 - sampleX) * e->maxl.x);
    //float lY = (sampleY * e->minl.y + (1 - sampleY) * e->maxl.y);
    //float lZ = e->lmin.z; // Assume e->lmin.z = e->lmax.z;
    //Vector2D sample = sampler.get_sample() - Vector2D(0.5f, 0.5f);
    CuRay *r = &cuConstRendererParams.queues[iid];
    
    float3 d = e->position + sampleX * e->dim_x + sampleY * e->dim_y - its->pt;
    float cosTheta = dot(d, e->direction);
    float sqDist = dot(d,d);
    float dist = sqrt(sqDist);
    r->d = d / dist;
    float distToLight = dist;
    float pdf = sqDist / (e->area * abs(cosTheta));
    float fpdf = abs(dot(its->n, r->d))/ pdf;
    r->lightImportance = its->importance * make_float3(fpdf, fpdf, fpdf);
    r->maxT = distToLight;
    r->importance = its->importance;
    r->sid = its->sid;
    r->light = its->light;

    //return cosTheta < 0 ? radiance : Spectrum();


    // Conenct light to point.

    /*float dX = -its->pt.x + lX;
    float dY = -its->pt.y + lY;
    float dZ = -its->pt.z + lZ;
    

    float l = sqrt(dX * dX + dY * dY + dZ * dZ);


    float3 d = make_float3(dX / l, dY / l, dZ / l);
    float3 o = its->pt;
    r->d = d;
    r->o = o;
    //float3 importance;// copy total importance.
    //float3 lightImportance; // Compute (n.l)/(p - l)^2

    //float3 light; // copy light.

    r->light = its->light;
    r->importance = its->importance;
    float invLight = 1.0f / dot(r->d - its->pt, r->d - its->pt);

    //r->lightImportance = its->importance * e->radiance * e->area * dot(e->direction, r->d) * dot(its->n, r->d) * invLight;// Compute (n.l) / (p - l)^2
    //r->maxT = sqrt(dot(r->d - its->pt, r->d - its->pt)) - 0.01f;
    //r->lightImportance = make_float3(xf, xf, xf);*/
    //r->sid = its->sid;
}
#define BSDF_DIFFUSE_MULTIPLIER 1.0
// Generate secondary rays from the given intersections.
__global__ void kernelProcessIntersections( ) {

    // For each element in intersection.
    // Check BSDF.
    // If 0 (Diffuse):
    // Randomly sample each intersection.
    // If 1 (Specular):
    // Find reflected ray.

    // Create new ray.
    // Compute importance for this ray.
    // Add to ray list at the same space as the intersection.

    int iid = blockIdx.x * blockDim.x + threadIdx.x;
    CuIntersection *its = &cuConstRendererParams.intersections[iid];
    CuRay *r = &cuConstRendererParams.queues[0];

    // Compute a random sample. Hemispherical Random Sample.
    float sampleX;
    float sampleY;
    float sampleZ;

    float3 n = its->n;
    float3 dpdu; //TODO:Compute;
    float3 dpdv; //TODO:Compute;

    float dX = n.x * sampleZ + sampleX * dpdu.x + sampleY * dpdv.x;
    float dY = n.y * sampleZ + sampleX * dpdu.y + sampleY * dpdv.y;
    float dZ = n.z * sampleZ + sampleX * dpdu.z + sampleY * dpdv.z;

    float3 d = make_float3(dX, dY, dZ);
    float3 o = its->pt;
    r->d = d;
    r->o = o;

    int bsdfID = its->bsdf;
    CuBSDF *bsdf = &cuConstRendererParams.bsdfs[bsdfID];

    if(bsdf->fn == 0) {
        r->importance = its->importance * dot(r->d, its->n) * bsdf->albedo * BSDF_DIFFUSE_MULTIPLIER; // TODO: Compute with BSDF.
        r->light = its->light;
        r->lightImportance = make_float3(0, 0, 0);
        r->maxT = INFINITY;
        r->sid = its->sid;
    } else if(bsdf->fn == 1){
        // TODO: Implement specular stuff.
        r->importance = its->importance * dot(r->d, its->n); // TODO: Compute with BSDF.
        r->light = its->light;
        r->lightImportance = make_float3(0, 0, 0);
        r->maxT = INFINITY;
        r->sid = its->sid;
    }
}

__global__ void kernelUpdateSSImage( ) {
    // For each element in intersection list.
    // Update the its.ss pixels using a reconstruction filter into
    // imageData.

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    int sampleCount = cuConstRendererParams.sampleCount;

    int iid = blockIdx.x * blockDim.x + threadIdx.x;

    CuIntersection *its = &cuConstRendererParams.intersections[iid];

    int x = static_cast<int>(its->ss.x);
    int y = static_cast<int>(its->ss.y);

    int sid = its->sid;

    float4 *fx = &cuConstRendererParams.imageData[((y * height + x) * sampleCount + sid)];
    *fx = make_float4(its->light.x, its->light.y, its->light.z, 1.0);

}

// Box Filter.
// Soon change to Gaussian.
__global__ void kernelReconstructImage( ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int scount = cuConstRendererParams.sampleCount;

    float4 color = make_float4(0, 0, 0, 0);
    for(int i = 0; i < scount; i++) {
        float4 localColor;
        localColor = cuConstRendererParams.ssImageData[idx * scount + i];
        /*localColor.x = cuConstRendererParams.ssImageData[idx * scount + i].x;
        localColor.y = cuConstRendererParams.ssImageData[idx * scount + i].y;
        localColor.z = cuConstRendererParams.ssImageData[idx * scount + i].y;
        localColor.w = cuConstRendererParams.ssImageData[idx * scount + i].w;*/
        

        /*color.x += localColor[0] / scount;
        color.y += localColor[1] / scount;
        color.z += localColor[2] / scount;
        color.w += localColor[3] / scount;*/

        color += localColor / scount;
        
    }

    cuConstRendererParams.imageData[idx] = color;
}

// Intersection functions
// Performs ray intersect on a single node.
__device__ void rayIntersectSingle(int snode, int index) {

    // Mapping: Each block takes a subset of rays.

    // Each thread takes one ray.

    // Combined load of the data for the snode to shared memory.

    // Test all 16 child node BBoxes against every ray.
 
    // If hit:
    // (TODO) Add to the queue for that outlet.

    // If miss:
    // Leave it. 


    int sampleCount = cuConstRendererParams.sampleCount;
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int rayCount = imageWidth * imageHeight * sampleCount;

    if(index < rayCount) {
        return;
    }

    CuRay *r = &cuConstRendererParams.queues[rayCount * snode + index];
    CuRay *raylist = &cuConstRendererParams.queues[rayCount * snode];
    
    bool is_leaf = (cuConstRendererParams.bvhSubTrees[snode].outlets[index] == TREE_WIDTH);
    // Combined load.
    if(!is_leaf) {
        __shared__ CuBVHSubTree subtree;
        __shared__ uint _outlets[TREE_WIDTH * RAYS_PER_BLOCK];
        __shared__  uint _c_outlets[TREE_WIDTH * RAYS_PER_BLOCK];
        __shared__ uint _compacter[TREE_WIDTH * RAYS_PER_BLOCK];
        __shared__ uint _c_qid[TREE_WIDTH * RAYS_PER_BLOCK];
        //__shared__ uint _scratch[TREE_WIDTH * RAYS_PER_BLOCK];
        
        if(index < 1) {
            subtree.start = cuConstRendererParams.bvhSubTrees[snode].start;
            subtree.range = cuConstRendererParams.bvhSubTrees[snode].range;
        }

        if(index < TREE_WIDTH) {
            subtree.outlets[index] = cuConstRendererParams.bvhSubTrees[snode].outlets[index];

            subtree.minl[index * 2 + 0] = cuConstRendererParams.bvhSubTrees[snode].minl[index * 2 + 0];
            subtree.minl[index * 2 + 1] = cuConstRendererParams.bvhSubTrees[snode].minl[index * 2 + 1];
            subtree.maxl[index * 2 + 0] = cuConstRendererParams.bvhSubTrees[snode].maxl[index * 2 + 0];
            subtree.maxl[index * 2 + 1] = cuConstRendererParams.bvhSubTrees[snode].maxl[index * 2 + 1];
        }

        __syncthreads();


        for(int i = 0; i < TREE_WIDTH; i++) {
            // Intersect the rays here.
            float t = intersectBBox(r->o, r->d, subtree.minl[i], subtree.maxl[i]);

            // TODO: Make sure we account for points inside the box too.

            if( t >= 0 ) {
                // If intersected, place a mark.
                _outlets[i * RAYS_PER_BLOCK + index] = 1;
            }
        }

        // Perform compaction on every 512x:(512x + 512)
        //_assignments[threadIdx.x] = assignments[(blockIdx.x << sizelog) + threadIdx.x];
        //if(blockIdx.x == 1000 && _assignments[threadIdx.x]){// && _assignments[(blockIdx.x << sizelog) + threadIdx.x]) {
        //    printf("%d: %u\n", threadIdx.x, blockIdx.x << sizelog);
        //}
        //_compacter[threadIdx.x] = compacter[blockIdx.x * size + threadIdx.x];
        //if(threadIdx.x % 10 == 0)
        //    printf("hello\n");
        // Wait for sync.
        __syncthreads();


        //sharedMemExclusiveScan(thread, _assignments, _compacter[sliceIdx * size], &scratch[sliceIdx * size], size);
        for(int i = 0; i < TREE_WIDTH; i++) {
            sharedMemExclusiveScan(threadIdx.x, &_outlets[i * RAYS_PER_BLOCK], &_c_outlets[i * RAYS_PER_BLOCK], &_c_qid[i * RAYS_PER_BLOCK], RAYS_PER_BLOCK);
        }

        __syncthreads();

        // Rearrange.
        for(int i = 0; i < TREE_WIDTH; i++) {
            if(index >= RAYS_PER_BLOCK) 
                continue;

            uint k0 = _c_outlets[i * RAYS_PER_BLOCK + index + 0];
            uint k1 = _c_outlets[i * RAYS_PER_BLOCK + index + 1];
            

            if(index != RAYS_PER_BLOCK - 1) {
                if(k0 + 1 == k1) 
                    _c_qid[k0] = index;
            } else {
                if(_outlets[i * RAYS_PER_BLOCK + index])
                    _c_qid[k0] = index;
            }
        }

        // Write out.
        for(int i = 0; i < TREE_WIDTH; i++) {
            int target = subtree.outlets[i];

            int rayid = _c_qid[i * RAYS_PER_BLOCK + index];

            if(index < _c_outlets[(i+1) * RAYS_PER_BLOCK - 1]) {
                // Leave.
                break;
            }

            __shared__ int tindex;

            __syncthreads();
            // Atomic grab.
            if (i == 0) {
                tindex = atomicAdd(&cuConstRendererParams.qCounts[target], _c_outlets[(i+1) * RAYS_PER_BLOCK - 1]);
            }

            __syncthreads();


            cuConstRendererParams.queues[rayCount * target + tindex + index] = raylist[rayid];
        }

    } else {
        // This is a leaf node. Quickly load whatever all triangles.
        __shared__ CuTriangle _triangles[MAX_TRIANGLES];
        int num_triangles = cuConstRendererParams.bvhSubTrees[snode].range;

        // Copy to shared memory
        if(index < num_triangles) {
            _triangles[index] = cuConstRendererParams.triangles[cuConstRendererParams.bvhSubTrees[snode].start + index];
        }

        float t = MAX_T_DISTANCE;

        CuTriangle tri;
        // Perform triangle intersect.
        for(int i = 0; i < num_triangles; i++) {
            float thist;
            if((thist = intersectRayTriangle(_triangles[i].a, _triangles[i].b, _triangles[i].c, r->o, r->d)) < t){
                t = thist;
                tri = _triangles[i];
            }

        }

        int x = static_cast<int>(r->ss.x);
        int y = static_cast<int>(r->ss.y);

        int sid = r->sid;

        int imageWidth = cuConstRendererParams.imageWidth;
        int imageHeight = cuConstRendererParams.imageHeight;
        
        //float4 *fx = &imageData[((y * height + x) * sampleCount + sid)];
        CuIntersection *its = &cuConstRendererParams.intersections[((y * imageHeight+ x) * sampleCount + sid)];
        if(its->t < t) {
            return;
        }

        bool direct_light = !(r->lightImportance == make_float3(0.0, 0.0, 0.0));
        if(!direct_light) {
            // Overwrite the intersection.
            its->t = t;
            its->pt = r->o + r->d * t;
            //its->lightImportance = r->lightImportance;
            its->light = r->light;
            its->importance = r->importance;

            //float3 n = normalize(cross(tri.a - tri.b, tri.b - tri.c));
            //its->n = ((dot(n, r->d) < 0) ? -1 : 1) * n;

            // Compute barycentric coordinates.
            float total = length(cross(tri.a - tri.b, tri.b - tri.c));
            //float bC = 0.0;
            //float bA = 0.0;
            //float bB = 0.0;

            float bC = length(cross(tri.a - its->pt, tri.b - its->pt)) / total;
            float bA = length(cross(tri.b - its->pt, tri.c - its->pt)) / total;
            float bB = length(cross(tri.c - its->pt, tri.a - its->pt)) / total;
            its->n = bA * tri.n0 + bB * tri.n1 + bC * tri.n2;

            // Make 2 more axes.
            float3 ax = normalize(cross(make_float3(0.1, 0.1, 1), its->n));
            float3 ay = normalize(cross(ax, its->n));
            its->wi = normalize(make_float3(dot(ax, r->d), dot(ay, r->d), dot(its->n, r->d)));

            its->ss = r->ss;
            its->sid = r->sid;
            its->bsdf = tri.bsdf;
        } else {
            // If direct light estimate, then only estimate the light at this point.
            its->light = r->light + ((t < r->maxT) ? (r->lightImportance) : make_float3(0.0)); // TODO: Make update.

        }

    }


}

__global__ void kernelRayIntersectSingle(int snode) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    rayIntersectSingle(snode, index);
}

// Intersection function.
// Performs ray intersection on a full level.
__global__ void kernelRayIntersectLevel(int level) {
    // Mapping: Each block takes a subset of rays split for that level.
    // Use blockID bits to split the rays to each snode in the level.
    // Each block only works on one particular snode.

    // Each thread takes one ray.

    // Combined load of the data for the snode to shared memory.

    // Test all 16 child node BBoxes against every ray.

    // If hit:
    // (TODO) Add to the queue for that outlet.

    // If miss:
    // Leave it.
    
    // Use a table to compute the nodes at this level.
    
    // Compute the queue index.
    int levelIndex = (blockIdx.x * blockDim.x) >> QUEUE_LENGTH_LOG2;
    
    int nodeIndex = cuConstRendererParams.levelIndices[level * LEVEL_INDEX_SIZE + levelIndex];
    int rayIndex = (blockIdx.x * blockDim.x) & (QUEUE_LENGTH_LOG2 - 1);

    rayIntersectSingle(nodeIndex, rayIndex);
}


    //#define NUM_CIRCLES_PER_BLOCK 1024
#define NUM_LAYERS 3

    //#define LOG_CIRCLES_PER_BLOCK 13
    CudaRenderer::CudaRenderer() {
        image = NULL;
        deviceBSDFs = NULL;
        deviceEmitters = NULL;
        deviceBVHSubTrees = NULL;
        deviceTriangles = NULL;
        deviceRays = NULL;
        deviceIntersections = NULL;
        deviceImageData = NULL;
        deviceSSImageData = NULL;
        deviceLevelIndices = NULL;
    }

    CudaRenderer::~CudaRenderer() {

        if (image) {
            delete image;
        }

        if (bsdfs.size() > 0) {
            bsdfs.clear();
            emitters.clear();
            subtrees.clear();
            triangles.clear();
            delete [] levelIndices;
        }

        if (deviceBSDFs) {
            cudaFree(deviceBSDFs);
            cudaFree(deviceEmitters);
            cudaFree(deviceBVHSubTrees);
            cudaFree(deviceTriangles);   
            cudaFree(deviceRays);
            cudaFree(deviceIntersections);
            cudaFree(deviceSSImageData);
            cudaFree(deviceImageData);
            cudaFree(deviceLevelIndices);
        }
    }

    const Image* CudaRenderer::getImage() {

        // Need to copy contents of the rendered image from device memory
        // before we expose the Image object to the caller

        printf("Copying image data from device\n");

        cudaMemcpy(image->data,
                deviceImageData,
                sizeof(float) * 4 * image->width * image->height,
                cudaMemcpyDeviceToHost);

        return image;
    }
    
    DynamicScene::Scene* CudaRenderer::loadFromSceneInfo(Collada::SceneInfo* sceneInfo) {
    
  vector<Collada::Node> &nodes = sceneInfo->nodes;
  vector<DynamicScene::SceneLight *> lights;
  vector<DynamicScene::SceneObject *> objects;

  // save camera position to update camera control later
  Collada::CameraInfo *c;
  Vector3D c_pos = Vector3D();
  Vector3D c_dir = Vector3D();

  int len = nodes.size();
  for (int i = 0; i < len; i++) {
    Collada::Node &node = nodes[i];
    Collada::Instance *instance = node.instance;
    const Matrix4x4 &transform = node.transform;

    switch (instance->type) {
      case Collada::Instance::CAMERA:
        c = static_cast<Collada::CameraInfo *>(instance);
        c_pos = (transform * Vector4D(c_pos, 1)).to3D();
        c_dir = (transform * Vector4D(c->view_dir, 1)).to3D().unit();
        init_camera(*c, transform);
        break;
      case Collada::Instance::LIGHT: {
        lights.push_back(
            init_light(static_cast<Collada::LightInfo &>(*instance), transform));
        break;
      }
      case Collada::Instance::SPHERE:
        objects.push_back(
            init_sphere(static_cast<Collada::SphereInfo &>(*instance), transform));
        break;
      case Collada::Instance::POLYMESH:
        objects.push_back(
            init_polymesh(static_cast<Collada::PolymeshInfo &>(*instance), transform));
        break;
      case Collada::Instance::MATERIAL:
        //init_material(static_cast<Collada::MaterialInfo &>(*instance));
        std::cout << "Unable to handle material.\n" << std::endl;
        break;
    }
  }

	// TODO: TEmporarily disabled this to test environment lights.
  if (lights.size() == 0) {  // no lights, default use ambient_light
    Collada::LightInfo default_light = Collada::LightInfo();
    lights.push_back(new DynamicScene::AmbientLight(default_light));
  }
  DynamicScene::Scene* scene = new DynamicScene::Scene(objects, lights);

  const BBox &bbox = scene->get_bbox();
  if (!bbox.empty()) {
    //Vector3D target = bbox.centroid();
    //canonical_view_distance = bbox.extent.norm() / 2 * 1.5;

    //double view_distance = canonical_view_distance * 2;
    //double min_view_distance = canonical_view_distance / 10.0;
    //double max_view_distance = canonical_view_distance * 20.0;

    //canonicalCamera.place(target, acos(c_dir.y), atan2(c_dir.x, c_dir.z),
    //                      view_distance, min_view_distance, max_view_distance);

    //camera.place(target, acos(c_dir.y), atan2(c_dir.x, c_dir.z), view_distance,
    //             min_view_distance, max_view_distance);

    //set_scroll_rate();
  }

  // set default draw styles for meshEdit -
  //scene->set_draw_styles(&defaultStyle, &hoverStyle, &selectStyle);

  // cerr << "==================================" << endl;
  // cerr << "CAMERA" << endl;
  // cerr << "      hFov: " << camera.hFov << endl;
  // cerr << "      vFov: " << camera.vFov << endl;
  // cerr << "        ar: " << camera.ar << endl;
  // cerr << "     nClip: " << camera.nClip << endl;
  // cerr << "     fClip: " << camera.fClip << endl;
  // cerr << "       pos: " << camera.pos << endl;
  // cerr << " targetPos: " << camera.targetPos << endl;
  // cerr << "       phi: " << camera.phi << endl;
  // cerr << "     theta: " << camera.theta << endl;
  // cerr << "         r: " << camera.r << endl;
  // cerr << "      minR: " << camera.minR << endl;
  // cerr << "      maxR: " << camera.maxR << endl;
  // cerr << "       c2w: " << camera.c2w << endl;
  // cerr << "   screenW: " << camera.screenW << endl;
  // cerr << "   screenH: " << camera.screenH << endl;
  // cerr << "screenDist: " << camera.screenDist<< endl;
  // cerr << "==================================" << endl;
        return scene;
    }

    void CudaRenderer::loadScene(std::string sceneFilePath) {
        //sceneName = scene;
        
        Collada::SceneInfo* sceneInfo = new Collada::SceneInfo();
        if (Collada::ColladaParser::load(sceneFilePath.c_str(), sceneInfo) < 0) {
            printf("Error: parsing failed!\n");
            delete sceneInfo;
            exit(0);
        }
    
        DynamicScene::Scene* dscene = this->loadFromSceneInfo(sceneInfo);
        StaticScene::Scene* scene = dscene->get_static_scene();

        std::vector<StaticScene::Primitive *> primitives;
        for (StaticScene::SceneObject *obj : scene->objects) {
            const vector<StaticScene::Primitive *> &obj_prims = obj->get_primitives();
            primitives.reserve(primitives.size() + obj_prims.size());
            primitives.insert(primitives.end(), obj_prims.begin(), obj_prims.end());
        }
        
        //std::vector<CuTriangle> cuts;
        for(auto prim : primitives) {
           StaticScene::Triangle* t = reinterpret_cast<StaticScene::Triangle*>(prim); 
           Vector3D v0,v1,v2;
           t->positions(v0, v1, v2);
           Vector3D n0,n1,n2;
           t->normals(n0, n1, n2);

           CuTriangle ct;
           ct.a = make_float3(v0.x, v0.y, v0.z);
           ct.b = make_float3(v1.x, v1.y, v1.z);
           ct.c = make_float3(v2.x, v2.y, v2.z);

           ct.n0 = make_float3(n0.x, n0.y, n0.z);
           ct.n1 = make_float3(n1.x, n1.y, n1.z);
           ct.n2 = make_float3(n2.x, n2.y, n2.z);

           ct.bsdf = 0;
           ct.emit = -1; 

           triangles.push_back(ct);
        }
        
        // Add BSDFs
        // TODO: Make this automated soon.
        //std::vector<CuBSDF> bsdfs;
        CuBSDF b;
        b.albedo = make_float3(0.6, 0.6, 0.6);
        b.fn = 0;
        b.nu = 0;

        bsdfs.push_back(b);

        // Add Emitters
        // TODO: Make this automated soon.
        CuEmitter e;
        auto l = scene->lights[0];
        auto al = reinterpret_cast<StaticScene::AreaLight*>(l);
        e.position = v2f3(al->position);
        e.direction = v2f3(al->direction);
        e.dim_x = v2f3(al->dim_x);
        e.dim_y = v2f3(al->dim_y);
        e.radiance = make_float3(al->radiance.r, al->radiance.g, al->radiance.b);

        //std::vector<CuEmitter> emitters;
        emitters.push_back(e);


        auto bvh = new StaticScene::BVHAccel(primitives);
        
        int scount;
        //this->subtrees = bvh->compressedTree();
        auto tmp_subtrees = bvh->compactedTree(); // Tree compaction system to make the tree smaller.

        std::vector<StaticScene::C_BVHSubTree> tree;
        std::vector<int> levelCounts;
        
        this->levelIndices = (int*) malloc(sizeof(int) * LEVEL_INDEX_SIZE * MAX_LEVELS);
        tmp_subtrees->compress(tree, this->levelIndices, LEVEL_INDEX_SIZE, levelCounts, 0, MAX_LEVELS); // Compressed subtree.

        for(auto entry : tree) {
            CuBVHSubTree cutree;
            cutree.start = entry.start;
            cutree.range = entry.range;
            for(int i = 0; i < TREE_WIDTH; i++){
                cutree.minl[i] = v2f3(entry.min[i]);
                cutree.maxl[i] = v2f3(entry.max[i]);
                cutree.outlets[i] = entry.outlets[i];
            }
            this->subtrees.push_back(cutree);
        }
        
    }
    
    void CudaRenderer::setup() {

        int deviceCount = 0;
        bool isFastGPU = false;
        std::string name;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        printf("---------------------------------------------------------\n");
        printf("Initializing CUDA for CudaRenderer\n");
        printf("Found %d CUDA devices\n", deviceCount);

        for (int i=0; i<deviceCount; i++) {
            cudaDeviceProp deviceProps;
            cudaGetDeviceProperties(&deviceProps, i);
            name = deviceProps.name;
            if (name.compare("GeForce GTX 1040") == 0)
            {
                isFastGPU = true;
            }

            printf("Device %d: %s\n", i, deviceProps.name);
            printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
            printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
            printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        }
        printf("---------------------------------------------------------\n");
        if (!isFastGPU)
        {
            printf("WARNING: "
                    "You're not running on a fast GPU, please consider using "
                    "NVIDIA GTX 480, 670 or 780.\n");
            printf("---------------------------------------------------------\n");
        }

        // By this time the scene should be loaded.  Now copy all the key
        // data structures into device memory so they are accessible to
        // CUDA kernels
        //
        // See the CUDA Programmer's Guide for descriptions of
        // cudaMalloc and cudaMemcpy
        // Compute Pow2 and Log2 versions of numCircles, this is important for the compaction algorithm,
        // which only works with powers of 2/


        // TODO: WARN: Temporary override.
        //numCircles = 1024;
        
        int numRays = SAMPLES_PER_PIXEL * image->width * image->height;
        int queueSize = numRays * subtrees.size();

        cudaMalloc(&deviceBSDFs, sizeof(CuBSDF) * bsdfs.size());
        cudaMalloc(&deviceEmitters, sizeof(CuEmitter) * emitters.size());
        cudaMalloc(&deviceTriangles, sizeof(CuTriangle) * triangles.size());
        cudaMalloc(&deviceBVHSubTrees, sizeof(CuBVHSubTree) * subtrees.size());
        cudaMalloc(&deviceRays, sizeof(CuRay) * queueSize);
        cudaMalloc(&deviceIntersections, sizeof(CuIntersection) * numRays);
        cudaMalloc(&deviceLevelIndices, sizeof(int) * LEVEL_INDEX_SIZE * MAX_LEVELS); 
        cudaMalloc(&deviceSSImageData, sizeof(float) * 4 * image->width * image->height * SAMPLES_PER_PIXEL);
        cudaMalloc(&deviceImageData, sizeof(float) * 4 * image->width * image->height);

        //int* hcounts = reinterpret_cast<int*>(calloc((image->width >> KWIDTH) * (image->height >> KWIDTH), sizeof(int)));

        //cudaMemcpy(counts, hcounts, sizeof(int) * (image->width >> KWIDTH) * (image->height >> KWIDTH), cudaMemcpyHostToDevice);
        //cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
        //cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
        //cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
        //cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceBSDFs, &bsdfs[0], sizeof(CuBSDF) * bsdfs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceEmitters, &emitters[0], sizeof(CuEmitter) * emitters.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceTriangles, &triangles[0], sizeof(CuTriangle) * triangles.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceBVHSubTrees, &subtrees[0], sizeof(CuBVHSubTree) * subtrees.size(), cudaMemcpyHostToDevice);


        // Initialize parameters in constant memory.  We didn't talk about
        // constant memory in class, but the use of read-only constant
        // memory here is an optimization over just sticking these values
        // in device global memory.  NVIDIA GPUs have a few special tricks
        // for optimizing access to constant memory.  Using global memory
        // here would have worked just as well.  See the Programmer's
        // Guide for more information about constant memory.

        GlobalConstants params;

        // Compute Pow2 and Log2 versions of numCircles, this is important for the compaction algorithm,
        // which only works with powers of 2/
        //int powlevel = 0;
        //int temp = numCircles;
        //while(temp >>= 1) powlevel ++;
        //params.numCirclesLog2 = powlevel+1;
        //params.numCirclesPow2 = 1 << (powlevel+1);
        //printf("params.numCirclesPow2 %d\n", 1 << (powlevel + 1));
        params.imageWidth = image->width;
        params.imageHeight = image->height;
        params.bsdfs = deviceBSDFs;
        params.emitters = deviceEmitters;
        params.bvhSubTrees = deviceBVHSubTrees;
        params.triangles = deviceTriangles;
        params.queues = deviceRays;
        params.intersections = deviceIntersections;
        params.ssImageData = (float4*)deviceSSImageData;
        params.imageData = (float4*)deviceImageData;

        cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

        // Also need to copy over the noise lookup tables, so we can
        // implement noise on the GPU
        int* permX;
        int* permY;
        float* value1D;
        //getNoiseTables(&permX, &permY, &value1D);
        //cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
        //cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
        //cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

        // Copy over the color table that's used by the shading
        // function for circles in the snowflake demo

        /*float lookupTable[COLOR_MAP_SIZE][3] = {
            {1.f, 1.f, 1.f},
            {1.f, 1.f, 1.f},
            {.8f, .9f, 1.f},
            {.8f, .9f, 1.f},
            {.8f, 0.8f, 1.f},
        };

        cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);*/
        printf("Finished allocation and copy\n");
    }

    // allocOutputImage --
    //
    // Allocate buffer the renderer will render into.  Check status of
    // image first to avoid memory leak.
    void CudaRenderer::allocOutputImage(int width, int height) {
        printf("Image alloc\n");
        if (image)
            delete image;
        image = new Image(width, height);
        printf("Done Image alloc\n");
    }

    // clearImage --
    //
    // Clear the renderer's target image.  The state of the image after
    // the clear depends on the scene being rendered.
    void CudaRenderer::clearImage() {

        dim3 blockDim(16, 16, 1);
        dim3 gridDim(
                (image->width + blockDim.x - 1) / blockDim.x,
                (image->height + blockDim.y - 1) / blockDim.y);

        kernelClearImage<<<gridDim, blockDim>>>(0.f, 0.f, 0.f, 0.f);

        cudaDeviceSynchronize();
        printf("Done cleaning\n");
    }

    void CudaRenderer::render() {
        //printf("Started rendering %d\n", batchSize);fflush(stdout);
        // 256 threads per block is a healthy number
        //dim3 blockDim(NUM_THREADS, 1);
        //dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
        //dim3 gridDim(numCircles);

        //dim3 blockDim(1024);
        //dim3 gridDim(1024 * (NUM_CIRCLES_PER_BLOCK >> 10));
        //dim3 blockDim((batchSize > 1024) ? 1024 : batchSize);
        //dim3 gridDim(1024 * ((numCircles > 1024) ? numCircles >> 10 : 1));


        //double start = CycleTimer::currentSeconds();

        //kernelClearBuffers<<<gridDimCl, blockDimCl>>>((float3*)qPositions, qRadii, (float3*)qColors, queues);
        //cudaDeviceSynchronize();

        //double clear = CycleTimer::currentSeconds();
        //printf("Executing kernel. %d, %d\n", numCircles, NUM_THREADS);
/*        kernelRenderCircles<<<gridDim, blockDim>>>(assignments, counts);
        cudaDeviceSynchronize(); 

        //double render = CycleTimer::currentSeconds();

        kernelExScan<<<gridDim3, blockDim3>>>((uint*)(assignments), (uint*)(compactor), (uint*)(scratch));
        cudaDeviceSynchronize();

        //double exscan = CycleTimer::currentSeconds();

        kernelCompact<<<gridDimC, blockDimC>>>((uint*)(assignments), (uint*)(compactor), (uint*)(queues), (float3*)qPositions, (float*)qRadii, (float3*)qColors, (int*) layerCount);
        cudaDeviceSynchronize();

        //double compact = CycleTimer::currentSeconds();

        if(sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME){
            //printf("Snowflake rendering.\n");
            kernelRenderSnowPixels<<<gridDim2, blockDim2>>>((uint*)queues, (float3*)qPositions, (float*)qRadii, (float3*)qColors, (float4*)layered, (int*) layerCount);
        }else{
            kernelRenderPixels<<<gridDim2, blockDim2>>>((uint*)queues, (float3*)qPositions, (float*)qRadii, (float3*)qColors, (float4*)layered, (int*) layerCount); 
        }
        cudaDeviceSynchronize();*/

        //double pixrender = CycleTimer::currentSeconds();


        //double combine = CycleTimer::currentSeconds();

        /*printf("ClearBuffers: %.4fms \n", (clear - start) * 1000.f);
          printf("RenderCircles: %.4fms \n", (render - clear) * 1000.f);
          printf("ExScan: %.4fms \n", (exscan - render) * 1000.f);
          printf("Compact: %.4fms \n", (compact - exscan) * 1000.f);
          printf("RenderPixels: %.4fms \n", (pixrender - compact) * 1000.f);
          printf("Combine: %.4fms \n", (combine - pixrender) * 1000.f);
          printf("Total Render: %.4fms \n", (combine - start) * 1000.f);*/

        //short* host_a = reinterpret_cast<short*>(malloc(sizeof(short) * 32 * 32 * numCircles));
        //int* host_c = reinterpret_cast<int*>(malloc(sizeof(int) *  32 * 32 ));
        //cudaMemcpy(host_a, assignments, sizeof(short) * 32 * 32 * numCircles, cudaMemcpyDeviceToHost);
        //cudaMemcpy(host_c, counts, sizeof(int) * 32 * 32, cudaMemcpyDeviceToHost);
        //printf("Item at : %d\n",host_c[5]);
        //printf("Finished rendering\n");
        //printf("Done rendering\n");fflush(stdout);
        //cudaDeviceSynchronize();
    }

}
