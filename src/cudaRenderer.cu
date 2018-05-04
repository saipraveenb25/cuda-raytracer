#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "cudaRenderer.h"
#include "cuda_image.h"
//#include "noise.h"
//#include "sceneLoader.h"
//#include "util.h"
//#include "cycleTimer.h"

#define SCAN_BLOCK_DIM 512  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"
#include "samplers.cu_inl"
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
        // Two buffers for double buffering
        CuRay* queues1;
        CuRay* queues2;

        // Queue counts. Initialized to 0.
        uint* qCounts;

        // Ray intersection buffers.
        // (Gigantic ~32M entries)
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

        // Intersection atomicity helpers.
        CuIntersection* multiIntersections;

        uint* intersectionTokens;

        float* minT;
        
        curandState* randomStates;
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
        
        if((o.x >= min.x && o.x <= max.x) 
            && (o.y >= min.y && o.y <= max.y) 
            && (o.z >= min.z && o.z <= max.z))
            return 0.0;

        float tmin = (min.x - o.x) / d.x; 
        float tmax = (max.x - o.x) / d.x; 

        //printf("tmin: %f\n", tmin);
        //printf("tmax: %f\n\n", tmax);
        if (tmin > tmax) cudaswap(tmin, tmax); 

        float tymin = (min.y - o.y) / d.y; 
        float tymax = (max.y - o.y) / d.y; 
        //printf("tymin: %f\n", tymin);
        //printf("tymax: %f\n\n", tymax);

        if (tymin > tymax) cudaswap(tymin, tymax); 

        if ((tmin > tymax) || (tymin > tmax)) 
            return -1.0; 

        if (tymin > tmin) 
            tmin = tymin; 
        //printf("tmin: %f\n", tmin);
        //printf("tmax: %f\n\n", tmax);

        if (tymax < tmax) 
            tmax = tymax; 

        float tzmin = (min.z - o.z) / d.z; 
        float tzmax = (max.z - o.z) / d.z; 
        //printf("tzmin: %f\n", tzmin);
        //printf("tzmax: %f\n\n", tzmax);

        if (tzmin > tzmax) cudaswap(tzmin, tzmax); 

        if ((tmin > tzmax) || (tzmin > tmax)) 
            return -1.0;

        if (tzmin > tmin) 
            tmin = tzmin; 

        //printf("tmin: %f\n", tmin);
        //printf("tmax: %f\n", tmax);

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
        if (abs(NdotRayDirection) < 1e-6) // almost 0 
            return -1; // they are parallel so they don't intersect ! 

        // compute d parameter using equation 2
        float d = dot(N, v0); 

        // compute t (equation 3)
        float t = (d - dot(N, orig)) / NdotRayDirection; 

        // check if the triangle is in behind the ray
        if (t < 0) return t; // the triangle is behind 

        // compute the intersection point using equation 1
        float3 P = orig + t * dir; 

        // Step 2: inside-outside test
        float3 C; // vector perpendicular to triangle's plane 

        // edge 0
        float3 edge0 = v1 - v0;
        float3 vp0 = P - v0;
        C = cross(edge0, vp0);
        if (dot(N, C) < 0) return -3; // P is on the right side 

        // edge 1
        float3 edge1 = v2 - v1; 
        float3 vp1 = P - v1; 
        C = cross(edge1, vp1); 
        if (dot(N, C) < 0)  return -4; // P is on the right side 

        // edge 2
        float3 edge2 = v0 - v2; 
        float3 vp2 = P - v2; 
        C = cross(edge2, vp2); 
        if (dot(N, C) < 0) return -5; // P is on the right side; 

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

        int offset = (imageY * width + imageX);
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
        //printf("x:%d, y:%d\n", imageX, imageY);
        
        int id = imageX * height + imageY;

        int sampleCount = cuConstRendererParams.sampleCount;
        curandState rand = cuConstRendererParams.randomStates[0];

        for(int i = 0; i < sampleCount; i++) {
            int destIndex = i + (imageX * height + imageY) * sampleCount;

            //float2 sample = make_float2(0.5f, 0.5f);
            float2 sample = squareSample(&rand);

            float xs = imageX + sample.x;
            float ys = imageY + sample.y;

            float2 ss = make_float2(xs, ys);

            //float a = (ss.x / width) - 0.5;
            //float b = (ss.y / height) - 0.5;
            //float c = sqrt(1 - (a*a + b*b));
            
            float3 k = make_float3((ss.y/width)-0.5, -((ss.x/height)-0.5),1.0);
            k = k / length(k);

            //float px = a * cuConstRendererParams.c_left.x + b * cuConstRendererParams.c_up.x + c * cuConstRendererParams.c_lookAt.x;
            //float py = a * cuConstRendererParams.c_left.y + b * cuConstRendererParams.c_up.y + c * cuConstRendererParams.c_lookAt.y;
            //float pz = a * cuConstRendererParams.c_left.z + b * cuConstRendererParams.c_up.z + c * cuConstRendererParams.c_lookAt.z;

            float3 dir = k.x * cuConstRendererParams.c_left + k.y * cuConstRendererParams.c_up + k.z * cuConstRendererParams.c_lookAt;
            //float3 dir = 
            //printf("x:%d, y:%d : %f %f %f\n", imageX, imageY, px, py, pz);

            CuRay *r = &cuConstRendererParams.queues1[destIndex];
            r->o = cuConstRendererParams.c_origin;
            r->d = dir;
            r->importance = make_float3(1, 1, 1);
            r->lightImportance = make_float3(0, 0, 0);
            r->light = make_float3(0, 0, 0);
            r->maxT = INFINITY; // TODO: Put inf.
            r->ss = ss;
            r->sid = i;
            r->id = destIndex;
            r->valid = true;
        }

        cuConstRendererParams.randomStates[0] = rand;

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
        

        CuRay *r = &cuConstRendererParams.queues1[iid];

        //if(iid > 30000 && iid < 30200) {
        //    printf("INTERSECTION: %d %d %d\n", iid, its->valid, r->id);
        //}

        if(!its->valid){ 
            r->valid = false;
            return;
        }
        
        curandState *rand = &cuConstRendererParams.randomStates[iid];
        float2 sample = squareSample(rand);
        // Generate the sample.
        float sampleX = (sample.x - 0.5);
        float sampleY = (sample.y - 0.5);

        
        float3 lpt = e->position + sampleX * e->dim_x + sampleY * e->dim_y;
        float3 d =  lpt - its->pt;
        float cosTheta = dot(d, e->direction);
        float sqDist = dot(d,d);
        float dist = sqrt(sqDist);
        r->d = d / dist;
        r->o = its->pt;
        float distToLight = dist;
        float pdf = sqDist / (e->area * abs(cosTheta));
        float fpdf = abs(dot(its->n, r->d))/ pdf;
        r->lightImportance = its->importance * make_float3(fpdf, fpdf, fpdf) * e->radiance;
        r->maxT = distToLight;
        r->importance = its->importance;
        r->sid = its->sid;
        r->light = its->light;
        r->id = its->id;

        // copied from its so that ray can easily duplicate value.
        r->n = its->n;
        r->wi = its->wi;
        r->t = its->t;
        r->valid = true;

        /*if(r->id > 30000 && r->id < 30200 && its->valid) {
            printf("INTERSECTION\n");
            printf("d: %f %f %f\n", r->d.x, r->d.y, r->d.z);
            printf("o: %f %f %f\n", r->o.x, r->o.y, r->o.z);
            printf("lightImportance: %f %f %f\n", r->lightImportance.x, r->lightImportance.y, r->lightImportance.z);
            printf("maxT: %f\n", r->maxT);
            printf("ID: %d\n", r->id);
            printf("SID: %d\n", r->sid);
            //printf("importance: %f %f %f\n", r->importance.x, r->importance.y, r->importance.z);
            //printf("its importance: %f %f %f\n", its->importance.x, its->importance.y, its->importance.z);
            printf("ID: %d SS: %f %f o: %f %f %f lpt: %f %f %f pt: %f %f %f t: %f d: %f %f %f its n: %f %f %f r->d: %f %f %f light: %f %f %f NEW: %d\n", r->id, r->ss.x, r->ss.y, r->o.x, r->o.y, r->o.z, lpt.x, lpt.y, lpt.z, its->pt.x, its->pt.y, its->pt.z, its->t, r->d.x, r->d.y, r->d.z, its->n.x, its->n.y, its->n.z, r->d.x, r->d.y, r->d.z, r->light.x, r->light.y, r->light.z, its->is_new);
            //printf("its n: %f %f %f\n", its->n.x, its->n.y, its->n.z);
        }*/
    }

    __global__ void kernelPrintLevelLists(int level, int total) {
        for(int i = 0; i < total; i++) {
            int nidx = cuConstRendererParams.levelIndices[level * LEVEL_INDEX_SIZE + i];
            printf("%d->%d\n", nidx, cuConstRendererParams.qCounts[nidx]);
        }
    }

    __global__ void kernelClearIntersections( ) {
        int iid = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = 0; i < MAX_INTERSECTIONS; i++) {
            CuIntersection *its = &cuConstRendererParams.multiIntersections[iid * MAX_INTERSECTIONS + i];
            its->valid = false;
        }

        cuConstRendererParams.intersections[iid].valid = false;
        cuConstRendererParams.minT[iid] = INFINITY;
        cuConstRendererParams.intersectionTokens[iid] = 0;
    }

    //__global__ void kernelClearIntersectionApparatus( ) {
    //    int iid = blockIdx.x * blockDim.x + threadIdx.x;
    //    cuConstRendererParams.minT[iid] = INFINITY;
    //    cuConstRendererParams.intersectionTokens[iid] = 0; 
    //}

    __global__ void kernelMergeIntersections( ) {
        int iid = blockIdx.x * blockDim.x + threadIdx.x;

        float t = INFINITY;
        CuIntersection best;
        for(int i = 0; i < MAX_INTERSECTIONS; i++) {
            CuIntersection *its = &cuConstRendererParams.multiIntersections[iid * MAX_INTERSECTIONS + i];
            //if(!its->valid) continue;
            if( its->valid && its->sort_t < t ) {
                //printf("VALID ITS: %d %f\n", iid, its->t);
                t = its->sort_t;
                best = *its;
            }
        }

        if(t != INFINITY)
            cuConstRendererParams.intersections[iid] = best;
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
        CuRay *r = &cuConstRendererParams.queues1[0];
        if(!its->is_new) {
            // This intersection slot is stale. Ignore
            return;
        }

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
            r->id = its->id;
        } else if(bsdf->fn == 1){
            // TODO: Implement specular stuff.
            r->importance = its->importance * dot(r->d, its->n); // TODO: Compute with BSDF.
            r->light = its->light;
            r->lightImportance = make_float3(0, 0, 0);
            r->maxT = INFINITY;
            r->sid = its->sid;
            r->id = its->id;
        }

        // Reset new bit so that we don't confuse intersections at a later stage.
        its->is_new = 0;
    }

    __global__ void kernelUpdateSSImage( ) {
        // For each element in intersection list.
        // Update the its.ss pixels using a reconstruction filter into
        // imageData.

        int width = cuConstRendererParams.imageWidth;
        int height = cuConstRendererParams.imageHeight;
        int sampleCount = cuConstRendererParams.sampleCount;

        int iid = blockIdx.x * blockDim.x + threadIdx.x;
        float4 *fx = &cuConstRendererParams.ssImageData[iid];
        CuIntersection *its = &cuConstRendererParams.intersections[iid];
            
        if(its->valid) {

            int x = static_cast<int>(its->ss.x);
            int y = static_cast<int>(its->ss.y);

            int sid = its->sid;

            //float4 *fx = &cuConstRendererParams.imageData[((y * height + x) * sampleCount + sid)];
            *fx = make_float4(its->light.x, its->light.y, its->light.z, 1.0);
            //*fx = make_float4(its->t / 5.0f, its->t / 5.0f, its->t / 5.0f, 1.0);
        } else {
            *fx = make_float4(0.0, 0.0, 0.0, 1.0);
        }
        
        
    }

    // Box Filter.
    // Soon change to Gaussian.
    __global__ void kernelReconstructImage( ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int scount = cuConstRendererParams.sampleCount;

        float4 color = make_float4(0, 0, 0, 0);
        for(int i = 0; i < scount; i++) {
            float4 localColor = cuConstRendererParams.ssImageData[idx * scount + i];
            color += localColor / scount;
        }
        
        //printf("%d->%f %f %f %f\n", idx, color.x, color.y, color.z, color.w);
        cuConstRendererParams.imageData[idx] = color;
    }

    __global__ void kernelResetCounts( ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int width = cuConstRendererParams.imageWidth;
        int height = cuConstRendererParams.imageHeight;
        int sampleCount = cuConstRendererParams.sampleCount;
        
        int totalRays = width * height * sampleCount;

        if(idx == 0)
            cuConstRendererParams.qCounts[idx] = totalRays;
        else
            cuConstRendererParams.qCounts[idx] = 0;
        

    }

    // Intersection functions
    // Performs ray intersect on a single node.
    __device__ void rayIntersectSingle(int snode, int index, CuRay* inputQueue, CuRay* outputQueue) {

        // Mapping: Each block takes a subset of rays.

        // Each thread takes one ray.

        // Combined load of the data for the snode to shared memory.

        // Test all 16 child node BBoxes against every ray.

        // If hit:
        // (TODO) Add to the queue for that outlet.

        // If miss:
        // Leave it. 
        //printf("%d, %d\n", snode, index);
        bool is_leaf = (cuConstRendererParams.bvhSubTrees[snode].range != 0);
        int sampleCount = cuConstRendererParams.sampleCount;
        int imageWidth = cuConstRendererParams.imageWidth;
        int imageHeight = cuConstRendererParams.imageHeight;
        int maxRayCount = imageWidth * imageHeight * sampleCount;
        int rayCount = min(maxRayCount, cuConstRendererParams.qCounts[snode]);

        __shared__ int tindex;
        //printf("%d, %d\n", is_leaf, index);

        //return;



        //if(index >= rayCount) {
        //    return;
        //}

        //if(threadIdx.x == 0) {
        //    printf("SNODE Read: %d->%d\n", snode, snode * maxRayCount);
        //}

        //CuRay *r = &cuConstRendererParams.queues[maxRayCount * snode + index];
        //CuRay *raylist = &cuConstRendererParams.queues[maxRayCount * snode];
       

        __shared__ CuBVHSubTree subtree;
        //bool is_leaf = (cuConstRendererParams.bvhSubTrees[snode].range != 0);
        //printf("%d, %d\n", is_leaf, index);
        //if(r->valid && index < 200 && index < rayCount)
        //    printf("STREE: %d, %d: %d, %d valid: %d\n", is_leaf, index, rayCount, maxRayCount, r->valid);

        //if(is_leaf) {
        //printf("LEAFFF: %d, %d: %d, %d\n", is_leaf, index, rayCount, maxRayCount); 
        //}
        //return;
        // Combined load.
        int subindex = threadIdx.x;
        if(subindex < 1) {
            subtree.start = cuConstRendererParams.bvhSubTrees[snode].start;
            subtree.range = cuConstRendererParams.bvhSubTrees[snode].range;
            subtree.wOffset = cuConstRendererParams.bvhSubTrees[snode].wOffset; // Write offset. Given by kernelScanCounts();
            subtree.rOffset = cuConstRendererParams.bvhSubTrees[snode].rOffset; // Read offset. Set by previous kernelRayIntersectLevel();
        }


        //if(index == 0)
        //    printf("AFTER STREE: %d, %d\n", is_leaf, index);

        // TODO: Test of RAYS_PER_BLOCK being a power of 2.

        if(subindex < TREE_WIDTH) {
            subtree.outlets[subindex] = cuConstRendererParams.bvhSubTrees[snode].outlets[subindex];
            subtree.minl[subindex + 0] = cuConstRendererParams.bvhSubTrees[snode].minl[subindex + 0];
            subtree.maxl[subindex + 0] = cuConstRendererParams.bvhSubTrees[snode].maxl[subindex + 0];
            
            
            if(subtree.outlets[subindex] != (uint64_t) -1){
                //if(subtree.outlets[subindex] > 600)

                cuConstRendererParams.bvhSubTrees[subtree.outlets[subindex]].rOffset = subtree.wOffset + subindex * rayCount; // Read offset for next level is based on write offset for this level.
            }
        } 
        

        __syncthreads();
        
        int rOffset = (snode == 0) ? 0 : subtree.rOffset;// + subindex * rayCount;
        int wOffset = subtree.wOffset;
        //outputOffsets[linearIndex] = ;
        
        CuRay *r = &inputQueue[rOffset + index];
        CuRay *raylist = &inputQueue[rOffset];
        
        //if(r->id == 528) {
        //if(snode == 294 && index < rayCount) {
        //    printf("%d: FOUND ID %d at %d+%d\n", snode, r->id, rOffset, index);
       // }
        
        //if(index == 0)
        //    printf("STREE: node: %d, %d, %d: %d, %d, %d, %d\n", snode, is_leaf, index, rayCount, maxRayCount, rOffset, wOffset);

        if(!is_leaf) {

            __shared__ uint _outlets[TREE_WIDTH * RAYS_PER_BLOCK];
            __shared__  uint _c_outlets[TREE_WIDTH * RAYS_PER_BLOCK];
            //__shared__ uint _compacter[TREE_WIDTH * RAYS_PER_BLOCK];
            __shared__ uint _c_qid[2 * TREE_WIDTH * RAYS_PER_BLOCK];
            //__shared__ uint _scratch[TREE_WIDTH * RAYS_PER_BLOCK];
            //if(index == 0)
            //    printf("STREE: %d, %d\n", is_leaf, index);
            //int subindex = index % (RAYS_PER_BLOCK);

            float minT = cuConstRendererParams.minT[r->id];
            /*if(index == 2100) {
              printf("SubTree: Start->%lu\n", subtree.start);
              printf("SubTree: Range->%lu\n", subtree.range);
              for(int i = 0; i < TREE_WIDTH; i++)
              printf("SubTree: Outlets[%d]->%lu\n", i, subtree.outlets[i]);
              for(int i = 0; i < TREE_WIDTH; i++)
              printf("SubTree: Min[%d]->%f %f %f\n", i, subtree.minl[i].x, subtree.minl[i].y, subtree.minl[i].z);
              for(int i = 0; i < TREE_WIDTH; i++)
              printf("SubTree: Max[%d]->%f %f %f\n", i, subtree.maxl[i].x, subtree.maxl[i].y, subtree.maxl[i].z);

            //printf("SUBTREE_TEST: %d\n",cuConstRendererParams.bvhSubTrees[0].outlets[0]);
                printf("\n");
            }*/

            for(int i = 0; i < TREE_WIDTH; i++) {
                _outlets[i * RAYS_PER_BLOCK + subindex] = 0; 
            }
            //return;
            __syncthreads();

            if(index < rayCount && r->valid) {

                for(int i = 0; i < TREE_WIDTH; i++) {
                    // Intersect the rays here.
                    //if(r->id == 33024) {
                        //printf("OUTLET[%d]=%lu\n", i, subtree.outlets[i]);
                    //}
                    if(subtree.outlets[i] == (uint64_t)-1) continue;

                    float t = intersectBBox(r->o, r->d, subtree.minl[i], subtree.maxl[i]);
                    
                    if( t >= 0 && t <= minT ) {
                        // If intersected, place a mark.
                        _outlets[i * RAYS_PER_BLOCK + subindex] = 1;
                        //printf("RAY %d,%d,%d: %f\n", i, RAYS_PER_BLOCK, subindex, t);
                    } else {
                        _outlets[i * RAYS_PER_BLOCK + subindex] = 0;
                    }

                }

           }

                //return;
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
                //if(index == 2100)
                //    printf("EXSCAN: %d, %d, %d, %d\n", RAYS_PER_BLOCK, subindex, TREE_WIDTH, blockDim.x);

                //if(subindex > RAYS_PER_BLOCK) {
                //    printf("EXSCAN: %d, %d, %d, %d\n", RAYS_PER_BLOCK, subindex, TREE_WIDTH, blockDim.x); 
                //}

                //return;
                //sharedMemExclusiveScan(thread, _assignments, _compacter[sliceIdx * size], &scratch[sliceIdx * size], size);
                for(int i = 0; i < TREE_WIDTH; i++) {
                    //if(subtree.outlets[i] == (uint64_t)-1) continue;
                    
                    sharedMemExclusiveScan(subindex, &_outlets[i * RAYS_PER_BLOCK], &_c_outlets[i * RAYS_PER_BLOCK], &_c_qid[i * 2 * RAYS_PER_BLOCK], RAYS_PER_BLOCK);
                }
                __syncthreads();

                /*if(subindex == 0) {
                    if(_c_outlets[RAYS_PER_BLOCK - 1] % 4 != 0) {
                        printf("BLOCK: %d Request: %d IDX: %d RC: %d\n", blockIdx.x, _c_outlets[RAYS_PER_BLOCK - 1], index, rayCount);
                        for(int i = 0; i < RAYS_PER_BLOCK; i+=4)
                            printf("%d:%d%d%d%d ", i, _outlets[i], _outlets[i+1], _outlets[i+2], _outlets[i+3]);
                        printf("\n");
                        //for(int i = 0; i < RAYS_PER_BLOCK; i++)
                        //    printf("%d:%d ", i, _c_outlets[i]);
                        //printf("\n");
                    } 

                }*/
                //return;
                // Rearrange.
                for(int i = 0; i < TREE_WIDTH; i++) {
                    if(subindex >= RAYS_PER_BLOCK) 
                        continue;

                    uint k0 = _c_outlets[i * RAYS_PER_BLOCK + subindex + 0];
                    uint k1 = _c_outlets[i * RAYS_PER_BLOCK + subindex + 1]; 
                    //printf("%d,%d->%d,%d - %d\n", i, subindex, k0, k1, _c_outlets[(i+1) * RAYS_PER_BLOCK - 1]);
                    if(subindex != RAYS_PER_BLOCK - 1) {
                        if((k0 + 1) == k1) {
                            _c_qid[i * RAYS_PER_BLOCK + k0] = index;
                            //printf("%d->%d\n", k0, index);
                        }
                    } else {
                        if(_outlets[i * RAYS_PER_BLOCK + subindex])
                            _c_qid[i * RAYS_PER_BLOCK + k0] = index;
                    }
                }

                __syncthreads();
                for(int i = 0; i < TREE_WIDTH; i++) {
                    uint64_t target = subtree.outlets[i];

                    __syncthreads();
                    // Atomic grab.
                    //if (raylist[rayid].id == 1536) {
                    int numRays = _c_outlets[(i+1) * RAYS_PER_BLOCK - 1] + _outlets[(i+1) * RAYS_PER_BLOCK - 1];
                    if(subindex == 0 && (target != (uint64_t)-1)){
                        tindex = atomicAdd(&cuConstRendererParams.qCounts[target], numRays);
                    }
                    __syncthreads();

                    if((target != (uint64_t)-1) && subindex < numRays) { 
                        int rayid = _c_qid[i * RAYS_PER_BLOCK + subindex];
                        //cuConstRendererParams.queues[maxRayCount * target + tindex + subindex] = raylist[rayid];
                        //if(wOffset + i * rayCount + tindex + subindex > 2097152) { printf("Illegal access: %d = %d+%d+%d+%d \n", wOffset + i * rayCount + tindex + subindex, wOffset, i * rayCount, tindex, subindex); }
                        outputQueue[(wOffset + i * rayCount) + tindex + subindex] = raylist[rayid];
                    }

                    //if( subindex == 0 && numRays != 0 ) {
                    //  printf("SNODE: %d TINDEX--------------(B:%dx%d) %d+%d (tot:%d) %lu->%d\n", snode, blockIdx.x, blockDim.x, subindex, tindex, numRays, target, wOffset + i * rayCount);
                    //}
                    //}

                    /*if(raylist[_c_qid[i * RAYS_PER_BLOCK + subindex]].id == 65664 && (target != (uint64_t)-1) && (subindex < numRays) ){
                    //subtree.
                    //CuBVHSubTree subtree = cuConstRendererParams.bvhSubTrees[snode];
                    int rayid = _c_qid[i * RAYS_PER_BLOCK + subindex];
                    float t = intersectBBox(raylist[rayid].o, raylist[rayid].d, subtree.minl[i], subtree.maxl[i]);
                    printf("Node: %d %lu->%d+%d,ID:%d\n", snode, target, cuConstRendererParams.qCounts[target], numRays, raylist[rayid].id);
                    printf("Target: %lu->%d Raycount: %d\n", target, wOffset + i * rayCount, rayCount);
                    printf("index:%d-i:%d \nSource Node: %d Node:%lu->Loc:%d+%d, RAYID: %d/%d \n MAX_INDICES: %d\nID:%d\n t:%f \nO:%f %f %f \n D:%f %f %f \nmin: %f %f %f\nmax: %f %f %f\n\n\n", index, i, snode, target, wOffset + i * rayCount, tindex + subindex, rayid, rayCount, numRays, raylist[rayid].id, t, raylist[rayid].o.x, raylist[rayid].o.y, raylist[rayid].o.z, raylist[rayid].d.x, raylist[rayid].d.y, raylist[rayid].d.z, subtree.minl[i].x, subtree.minl[i].y, subtree.minl[i].z, subtree.maxl[i].x, subtree.maxl[i].y, subtree.maxl[i].z);


                    printf("tindex: %lu, %d, %lu, %f\n", maxRayCount * target, tindex + subindex, maxRayCount * target + tindex + subindex, t);
                    //printf("Ntest: %f\n\n", t);
                    }*/

                    //__syncthreads();


                }

            } else {

                __shared__ CuTriangle _triangles[MAX_TRIANGLES];
                int num_triangles = cuConstRendererParams.bvhSubTrees[snode].range;

                //return;
                // Copy to shared memory

                //int subindex = threadIdx.x;

                if(subindex < num_triangles) {
                    _triangles[subindex] = cuConstRendererParams.triangles[cuConstRendererParams.bvhSubTrees[snode].start + subindex];
                }

                //return;

                __syncthreads();

                float t = INFINITY;

                CuTriangle tri;
                // Perform triangle intersect.
                if(index < rayCount && r->valid) {
                    for(int i = 0; i < num_triangles; i++) {
                        int start = cuConstRendererParams.bvhSubTrees[snode].start;
                        //if(subindex == 0 && (start + i > 1733) && (start + i < 1745)){
                        float thist = intersectRayTriangle(_triangles[i].a, _triangles[i].b, _triangles[i].c, r->o, r->d);
                        if(thist < t && thist >= 0){
                            //if(thist >= 0){
                            t = thist;
                            tri = _triangles[i];
                            //}
                        }

                    }
                }


                //int x = static_cast<int>(r->ss.x);
                //int y = static_cast<int>(r->ss.y);

                //int sid = r->sid;

                int imageWidth = cuConstRendererParams.imageWidth;
                int imageHeight = cuConstRendererParams.imageHeight;
                //int iid = ((y * imageHeight + x) * sampleCount + sid);

                int iid = r->id;


                if(iid >= imageWidth * imageHeight * SAMPLES_PER_PIXEL) 
                    printf("IID exceeds total samples.%d\n", iid);

                //float4 *fx = &imageData[((y * height + x) * sampleCount + sid)];
                //CuIntersection *ac_its = &cuConstRendererParams.intersections[((y * imageHeight+ x) * sampleCount + sid)];
                float tmin = cuConstRendererParams.minT[iid];
                uint *tokens = &cuConstRendererParams.intersectionTokens[iid];
                CuIntersection its;

                //if(atomicMin(tmin, t) != t) {
                if(t == INFINITY || tmin < t) {
                    return;
                }

                cuConstRendererParams.minT[iid] = t;

                // Take a token.
                int token = atomicAdd(tokens, 1);
                bool direct_light = !(r->lightImportance == make_float3(0.0, 0.0, 0.0));
                if(!direct_light) {
                    // Overwrite the intersection.
                    its.t = t;
                    its.sort_t = t;
                    its.pt = (r->o + r->d * t);// + r->n * 1e-4;
                    //its->lightImportance = r->lightImportance;
                    its.light = r->light;
                    its.importance = r->importance;

                    //float3 n = normalize(cross(tri.a - tri.b, tri.b - tri.c));
                    //its->n = ((dot(n, r->d) < 0) ? -1 : 1) * n;

                    // Compute barycentric coordinates.
                    float total = length(cross(tri.a - tri.b, tri.b - tri.c));
                    //float bC = 0.0;
                    //float bA = 0.0;
                    //float bB = 0.0;

                    float bC = length(cross(tri.a - its.pt, tri.b - its.pt)) / total;
                    float bA = length(cross(tri.b - its.pt, tri.c - its.pt)) / total;
                    float bB = length(cross(tri.c - its.pt, tri.a - its.pt)) / total;
                    its.n = bA * tri.n0 + bB * tri.n1 + bC * tri.n2;
                    its.n = (its.n) * (dot(its.n, r->d) < 0 ? 1 : -1);
                    
                    its.pt += its.n * 1e-4;

                    // Make 2 more axes.
                    float3 ax = normalize(cross(make_float3(0.1, 0.1, 1), its.n));
                    float3 ay = normalize(cross(ax, its.n));
                    its.wi = normalize(make_float3(dot(ax, r->d), dot(ay, r->d), dot(its.n, r->d)));

                    its.ss = r->ss;
                    its.sid = r->sid;
                    its.id = r->id;
                    its.bsdf = tri.bsdf;
                    its.is_new = 1;
                    its.valid = true;
                } else {
                    // If direct light estimate, then only estimate the light at this point.
                    

                    its.n = r->n;
                    its.wi = r->wi;
                    its.ss = r->ss;
                    its.sid = r->sid;
                    its.id = r->id;
                    its.importance = r->importance;
                    its.pt = r->o;//make_float3(r->maxT, r->maxT, r->maxT);//TODO: CHANGED CHANGE THIS BACK TODO TODO TODO TODO
                    its.t = r->t; //TODO: CHANGED CHANGE THIS BACK TODO TODO TODO TODO
                    its.sort_t = t;
                    its.bsdf = tri.bsdf;
                    its.light = r->light + ((t > r->maxT - 1e-4) ? (r->lightImportance) : make_float3(0.0)); // TODO: Make update.
                    its.is_new = 2;
                    its.valid = true;
                    //if(r->id > 30000 && r->id < 30200 && r->valid) {
                    //    printf("Intersection ID: %d %d\n", its.id, its.is_new);
                    //}
                    
                }

                // write this intersection out.
                cuConstRendererParams.multiIntersections[iid * MAX_INTERSECTIONS + token] = its;
            }


            }

            __global__ void kernelSetupRandomSeeds(){
            	int idx = blockIdx.x * blockDim.x + threadIdx.x;
            	curand_init(seed, idx, 0, &cuConstRendererParams.randomStates[idx]);
            }

            __global__ void kernelRayIntersectSingle(int snode) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                //if(index == 0) {
                //    printf("INDEX: %d\n\n", index);
                //}
                rayIntersectSingle(snode, index, cuConstRendererParams.queues1, cuConstRendererParams.queues2);
            }

            __global__ void kernelPrintQueueCounts() {
                //printf("%u->%u\n", threadIdx.x, cuConstRendererParams.qCounts[threadIdx.x]);
            }

            __global__ void kernelScanCounts(int level) {
                int idx = threadIdx.x;
                __shared__ uint inputCounts[512];
                __shared__ uint outputCounts[512];
                __shared__ uint spare[1024];
                
                int nodeIdx = cuConstRendererParams.levelIndices[level * LEVEL_INDEX_SIZE + idx];
                inputCounts[idx] = cuConstRendererParams.qCounts[nodeIdx];
                __syncthreads();
                sharedMemExclusiveScan(idx, &inputCounts[0], &outputCounts[0], &spare[0], 512);
                __syncthreads();

                cuConstRendererParams.bvhSubTrees[nodeIdx].wOffset = outputCounts[idx] * TREE_WIDTH; // Provide space for all branches (worst-case)
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
                //int levelIndex = (blockIdx.x * blockDim.x) >> QUEUE_LENGTH_LOG2;
                int imageWidth = cuConstRendererParams.imageWidth;
                int imageHeight = cuConstRendererParams.imageHeight;
                int sampleCount = cuConstRendererParams.sampleCount;

                int levelIndex = (blockIdx.x * blockDim.x) / (imageWidth * imageHeight * sampleCount);

                int nodeIndex = cuConstRendererParams.levelIndices[level * LEVEL_INDEX_SIZE + levelIndex];
                int rayIndex = (blockIdx.x * blockDim.x + threadIdx.x) % (imageWidth * imageHeight * sampleCount);
                

                //if(rayIndex == 0){
                //    printf("At %d\n", nodeIndex);
                //}
                //if(level < 4) { 
                int rayCount = cuConstRendererParams.qCounts[nodeIndex];

                __shared__ bool active;

                if(threadIdx.x == 0) {
                    active = (rayIndex <= rayCount);
                }
                
                __syncthreads();
                
                if (active) {
                    if(level % 2 == 0) {
                        rayIntersectSingle(nodeIndex, rayIndex, cuConstRendererParams.queues1, cuConstRendererParams.queues2);
                    } else {
                        rayIntersectSingle(nodeIndex, rayIndex, cuConstRendererParams.queues2, cuConstRendererParams.queues1); 
                    }
                }
                //}
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
                deviceRays1 = NULL;
                deviceRays2 = NULL;
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
                    cudaFree(deviceRays1);
                    cudaFree(deviceRays2);
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
                        case Collada::Instance::CAMERA: {
                                                            c = static_cast<Collada::CameraInfo *>(instance);
                                                            c_pos = (transform * Vector4D(c_pos, 1)).to3D();
                                                            c_dir = (transform * Vector4D(c->view_dir, 1)).to3D().unit();
                                                            std::cout << "Camera parameters: " << std::endl;
                                                            this->c_lookAt = -c_dir;
                                                            this->c_origin = c_pos + Vector3D(0, 0.5, 0);
                                                            Vector3D acup(0.0f, 1.0f, 0.0f);
                                                            this->c_left = cross(acup, c_dir).unit();
                                                            this->c_up = cross(this->c_left, c_dir).unit();
                                                            std::cout << "lookAt: " << this->c_lookAt << std::endl;
                                                            std::cout << "origin: " << this->c_origin << std::endl;
                                                            std::cout << "left: " << this->c_left << std::endl;
                                                            std::cout << "up: " << this->c_up << std::endl;

                                                            init_camera(*c, transform);
                                                            break;
                                                        }
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
                printf("Loading scene: %s\n", sceneFilePath.c_str()); 
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
                if(scene->lights.size() != 1) {
                    std::cout << "Error: Too many lights: " << scene->lights.size() << ". Can only handle one for now.\n";
                    exit(0);
                }

                CuEmitter e;
                auto l = scene->lights[0];
                auto al = reinterpret_cast<StaticScene::AreaLight*>(l);

                std::cout << "AreaLight: " << al->position << " ,\n" << al->direction << " ,\n" << al->dim_x << " ,\n" << al->dim_y << std::endl;
                std::cout << al->radiance << std::endl;

                e.position = v2f3(al->position);
                e.direction = v2f3(al->direction);
                e.dim_x = v2f3(al->dim_x);
                e.dim_y = v2f3(al->dim_y);
                e.radiance = make_float3(al->radiance.r, al->radiance.g, al->radiance.b);
                e.area = length(e.dim_x) * length(e.dim_y);
                std::cout << "Area: " << e.area;
                //std::vector<CuEmitter> emitters;
                emitters.push_back(e);


                auto bvh = new StaticScene::BVHAccel(primitives);
                
                std::cout << "Primitives loaded: " << primitives.size() << std::endl;
                for(auto prim : bvh->getSortedPrimitives()) {
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

                    //std::cout << "Primitive: " << v0 << " " << v1 << " " << v2 << std::endl;
                }
                
                int scount;
                //this->subtrees = bvh->compressedTree();
                auto tmp_subtrees = bvh->compactedTree(); // Tree compaction system to make the tree smaller.

                std::vector<StaticScene::C_BVHSubTree> *tree = new std::vector<StaticScene::C_BVHSubTree>();
                //std::vector<int> *levelCounts = new std::vector<int>();

                this->levelIndices = (int*) malloc(sizeof(int) * LEVEL_INDEX_SIZE * MAX_LEVELS);
                tmp_subtrees->compress(tree, this->levelIndices, LEVEL_INDEX_SIZE, &levelCounts, 0, MAX_LEVELS); // Compressed subtree.

                int curr = 0;
                for(int t = 0; t < tree->size(); t ++) {
                    auto entry = tree->at(t);
                    //if(entry.range != 0) continue;
                    //std::cout << "BVHSubTree entry " << curr++ << std::endl; 

                    //std::cout << "start: " << entry.start << std::endl;
                    //std::cout << "range: " << entry.range << std::endl;

                    CuBVHSubTree cutree;
                    cutree.start = entry.start;
                    cutree.range = entry.range;

                    for(int i = 0; i < TREE_WIDTH; i++){
                        //std::cout << "min " << i << ": " << entry.min[i] << std::endl;
                        //std::cout << "max " << i << ": " << entry.max[i] << std::endl;
                        //std::cout << "outlet " << i << ": " << entry.outlets[i] << std::endl;

                        cutree.minl[i] = v2f3(entry.min[i]);
                        cutree.maxl[i] = v2f3(entry.max[i]);
                        cutree.outlets[i] = entry.outlets[i];
                    }
                    this->subtrees.push_back(cutree);
                }

                std::cout << "\nLevel Profile\n";
                for(int t = 0; t < levelCounts.size(); t++) {
                    std::cout << t << "->" << levelCounts.at(t) << std::endl;
                }
                std::cout << std::endl;
                for(int i = 0; i < levelCounts.size(); i++) {
                    int sz = levelCounts.at(i);
                    for(int j = 0; j < sz; j++) {
                        printf("%d,",levelIndices[LEVEL_INDEX_SIZE * i + j]);
                    }
                    printf("\n");
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
                int queueSize = numRays * TREE_WIDTH * 2;
                
                std::cout << "Queue Size: " << queueSize << std::endl;
                std::cout << "\nDevice Allocation \n";
                std::cout << "BSDFS: " << bsdfs.size() << std::endl;
                std::cout << "Emitters: " << emitters.size() << std::endl;
                std::cout << "Triangles: " << triangles.size() << std::endl;
                std::cout << "BVHSubTrees: " << subtrees.size() << std::endl;
                
                std::cout << "BVH SIZE: " << sizeof(CuBVHSubTree) * subtrees.size() << std::endl;

                size_t total = 0;
                total += sizeof(CuBSDF) * bsdfs.size();
                total += sizeof(CuEmitter) * emitters.size();
                total += sizeof(CuTriangle) * triangles.size();
                total += sizeof(CuBVHSubTree) * subtrees.size();
                total += sizeof(CuRay) * queueSize;
                total += sizeof(CuIntersection) * numRays;
                total += sizeof(int) * LEVEL_INDEX_SIZE * MAX_LEVELS; 
                total += sizeof(float) * 4 * image->width * image->height * SAMPLES_PER_PIXEL;
                total += sizeof(float) * 4 * image->width * image->height;
                total += sizeof(uint) * subtrees.size();
                total += sizeof(float) * numRays;
                total += sizeof(uint) * numRays;
                total += sizeof(CuIntersection) * MAX_INTERSECTIONS * numRays;
                
                std::cout << "Total memory allocation: " << total / 1000000 << " MB" << std::endl;
                std::cout << "Device rays: " << sizeof(CuRay) * queueSize / 1000000 << " MB" << std::endl;

                auto ok = cudaMalloc(&deviceBSDFs, sizeof(CuBSDF) * bsdfs.size());
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceEmitters, sizeof(CuEmitter) * emitters.size());
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceTriangles, sizeof(CuTriangle) * triangles.size());
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceBVHSubTrees, sizeof(CuBVHSubTree) * subtrees.size());
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceRays1, sizeof(CuRay) * queueSize);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceRays2, sizeof(CuRay) * queueSize);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceIntersections, sizeof(CuIntersection) * numRays);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceLevelIndices, sizeof(int) * LEVEL_INDEX_SIZE * MAX_LEVELS); 
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceSSImageData, sizeof(float) * 4 * image->width * image->height * SAMPLES_PER_PIXEL);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceImageData, sizeof(float) * 4 * image->width * image->height);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceQueueCounts, sizeof(uint) * subtrees.size());
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceMinT, sizeof(float) * numRays);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceIntersectionTokens, sizeof(uint) * numRays);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMalloc(&deviceMultiIntersections, sizeof(CuIntersection) * MAX_INTERSECTIONS * numRays);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}

                //cudaMalloc(&deviceQueueOffsets1, sizeof(uint) * subtrees.size());
                //cudaMalloc(&deviceQueueOffsets2, sizeof(uint) * subtrees.size());

                ok = cudaMalloc(&deviceRandomStates, sizeof(curandState) * numRays);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}

                //int* hcounts = reinterpret_cast<int*>(calloc((image->width >> KWIDTH) * (image->height >> KWIDTH), sizeof(int)));

                //cudaMemcpy(counts, hcounts, sizeof(int) * (image->width >> KWIDTH) * (image->height >> KWIDTH), cudaMemcpyHostToDevice);
                //cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
                //cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
                //cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
                //cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);
                std::vector<uint> qcounts;
                qcounts.push_back(numRays);
                for(int i = 1; i < subtrees.size(); i++ ){
                    qcounts.push_back(0);
                }


                ok = cudaMemcpy(deviceBSDFs, &bsdfs[0], sizeof(CuBSDF) * bsdfs.size(), cudaMemcpyHostToDevice);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMemcpy(deviceEmitters, &emitters[0], sizeof(CuEmitter) * emitters.size(), cudaMemcpyHostToDevice);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMemcpy(deviceTriangles, &triangles[0], sizeof(CuTriangle) * triangles.size(), cudaMemcpyHostToDevice);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMemcpy(deviceBVHSubTrees, &subtrees[0], sizeof(CuBVHSubTree) * subtrees.size(), cudaMemcpyHostToDevice);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMemcpy(deviceQueueCounts, &qcounts[0], sizeof(uint) * subtrees.size(), cudaMemcpyHostToDevice);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}
                ok = cudaMemcpy(deviceLevelIndices, levelIndices, sizeof(int) * LEVEL_INDEX_SIZE * MAX_LEVELS, cudaMemcpyHostToDevice);
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}

                printf("CuTest: %d\n", subtrees[0].outlets[0]);

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
                params.queues1 = deviceRays1;
                params.queues2 = deviceRays2;
                params.intersections = deviceIntersections;
                params.ssImageData = (float4*)deviceSSImageData;
                params.imageData = (float4*)deviceImageData;
                params.qCounts = deviceQueueCounts;
                params.levelIndices = deviceLevelIndices;
                params.c_lookAt = v2f3(c_lookAt);
                params.c_up = v2f3(c_up);
                params.c_origin = v2f3(c_origin);
                params.c_left = v2f3(c_left);
                params.sampleCount = SAMPLES_PER_PIXEL;
                params.minT = deviceMinT;
                params.multiIntersections = deviceMultiIntersections;
                params.intersectionTokens = deviceIntersectionTokens;
                params.randomStates = deviceRandomStates;

                ok = cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
                if(ok != cudaSuccess) {printf("Couldn't allocate memory\n");exit(1);}

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

            void CudaRenderer::clearIntersections() {
                int numRays = image->height * image->width * SAMPLES_PER_PIXEL;
                int threadCount = 1024;

                dim3 blockDim(threadCount, 1);
                dim3 gridDim(numRays / threadCount, 1);

                kernelClearIntersections<<<gridDim, blockDim>>>();
            }
            
            void CudaRenderer::render() {

                //printf("Started rendering %d\n", batchSize);fflush(stdout);
                // 256 threads per block is a healthy number
                //dim3 blockDim(NUM_THREADS, 1);
                //dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
                //dim3 gridDim(numCircles);

                //dim3 blockDim(1024);
                //dim3 gridDim(1024 * (NUM_CIRCLES_PER_BLOCK >> 10));
                dim3 primaryRaysBlockDim(32, 32);
                dim3 primaryRaysGridDim(image->width >> 5, image->height >> 5);

                int blocksPerNode = (image->width * image->height * SAMPLES_PER_PIXEL) / RAYS_PER_BLOCK;
                dim3 rayIntersectBlockDim(RAYS_PER_BLOCK, 1);
                dim3 rayIntersectGridDim(blocksPerNode, 1);
                
                int iidBlocksPerNode = (image->width * image->height * SAMPLES_PER_PIXEL) / 1024;
                dim3 intersectionBlockDim(1024, 1);
                dim3 intersectionGridDim(iidBlocksPerNode, 1);

                int imageBlocksPerNode = (image->width * image->height) / 1024;
                dim3 imageBlockDim(1024, 1);
                dim3 imageGridDim(imageBlocksPerNode, 1);
                
                int a = 0;
                for(int i = 0; i < levelCounts.size(); i++)
                    a += levelCounts[i];
                
                //dim3 queueCountsBlockDim(400, 1);
                //dim3 queueCountsGridDim(1, 1);
                
                kernelSetupRandomSeeds<<<intersectionGridDim, intersectionBlockDim>>>();       
                
                cudaDeviceSynchronize();
                //double start = CycleTimer::currentSeconds();
                kernelPrimaryRays<<<primaryRaysGridDim, primaryRaysBlockDim>>>();

                cudaDeviceSynchronize();
                // TODO: Concurrent execution.
                clearIntersections();

                cudaDeviceSynchronize();
                
                printf("Reset counts: %d\n", a);
                kernelResetCounts<<<1, a>>>();

                cudaDeviceSynchronize();
                
                kernelScanCounts<<<1,levelCounts[0]>>>(0);

                cudaDeviceSynchronize();

                kernelRayIntersectSingle<<<rayIntersectGridDim, rayIntersectBlockDim>>>(0);

                cudaDeviceSynchronize();
                    
                // Compute level indices.
                for(int level = 1; level < levelCounts.size(); level ++) {
                    //if(level != levelCounts.size() - 1) {
                    printf("kernelPrintLevelLists\n");
                    kernelPrintLevelLists<<<1,1>>>(level, levelCounts[level]);
                    cudaDeviceSynchronize();
                    //}
                    // for(int level = 1; level < 2; level++) {
                    int totalCount = image->height * image->width * SAMPLES_PER_PIXEL * levelCounts[level];
                    int numBlocks = totalCount / RAYS_PER_BLOCK;
                    //int numBlocks = totalCount / RAYS_PER_BLOCK;
                    printf("kernelIntersectLevel: %d, %d, %d, %d\n", totalCount, numBlocks, levelCounts[level], SAMPLES_PER_PIXEL * image->height * image->width); 
                    
                    
                    kernelScanCounts<<<1,levelCounts[level]>>>(level);
                    
                    cudaDeviceSynchronize();

                    dim3 rayIntersectLevelBlockDim(RAYS_PER_BLOCK, 1);
                    dim3 rayIntersectLevelGridDim(numBlocks, 1);
                    kernelRayIntersectLevel<<<numBlocks, RAYS_PER_BLOCK>>>(level);
                    cudaDeviceSynchronize();

                }

                //cudaDeviceSynchronize();

                //kernelPrintQueueCounts<<<queueCountsGridDim, queueCountsBlockDim>>>();

                //cudaDeviceSynchronize();
                
                //printf("kernelMergeIntersections\n");
                kernelMergeIntersections<<<intersectionGridDim, intersectionBlockDim>>>();
                
                cudaDeviceSynchronize();
                
                //printf("kernelUpdateSSImage\n");

                kernelDirectLightRays<<<intersectionGridDim, intersectionBlockDim>>>();

                cudaDeviceSynchronize();
                
                clearIntersections();

                
                cudaDeviceSynchronize();
                      
                // BOUNCE TWO (DIRECT LIGHT) 
                
                printf("Reset counts: %d\n", a);
                kernelResetCounts<<<1, a>>>();

                cudaDeviceSynchronize();
                
                kernelScanCounts<<<1,levelCounts[0]>>>(0);

                cudaDeviceSynchronize();

                kernelRayIntersectSingle<<<rayIntersectGridDim, rayIntersectBlockDim>>>(0);

                cudaDeviceSynchronize();

                // Compute level indices.
                for(int level = 1; level < levelCounts.size(); level ++) {
                    //if(level != levelCounts.size() - 1) {
                    printf("kernelPrintLevelLists DLIGHT %d\n", level);
                    kernelPrintLevelLists<<<1,1>>>(level, levelCounts[level]);
                    cudaDeviceSynchronize();
                    //}
                    // for(int level = 1; level < 2; level++) {
                    int totalCount = image->height * image->width * SAMPLES_PER_PIXEL * levelCounts[level];
                    int numBlocks = totalCount / RAYS_PER_BLOCK;
                    //int numBlocks = totalCount / RAYS_PER_BLOCK;
                    printf("kernelIntersectLevel: %d, %d, %d, %d\n", totalCount, numBlocks, levelCounts[level], SAMPLES_PER_PIXEL * image->height * image->width); 
                    
                    kernelScanCounts<<<1,levelCounts[level]>>>(level);
                    
                    cudaDeviceSynchronize();

                    dim3 rayIntersectLevelBlockDim(RAYS_PER_BLOCK, 1);
                    dim3 rayIntersectLevelGridDim(numBlocks, 1);
                    kernelRayIntersectLevel<<<numBlocks, RAYS_PER_BLOCK>>>(level);
                    cudaDeviceSynchronize();
                }


                //cudaDeviceSynchronize();

                //kernelPrintQueueCounts<<<queueCountsGridDim, queueCountsBlockDim>>>();

                //cudaDeviceSynchronize();
                
                //printf("kernelMergeIntersections\n");
                kernelMergeIntersections<<<intersectionGridDim, intersectionBlockDim>>>();
                
                cudaDeviceSynchronize();
                
                kernelUpdateSSImage<<<intersectionGridDim, intersectionBlockDim>>>();
                
                cudaDeviceSynchronize();
                
                //printf("kernelReconstructImage\n");
                kernelReconstructImage<<<imageGridDim, imageBlockDim>>>();
                
                cudaDeviceSynchronize();
                
                //kernelDirectLightRays<<<intersectionGridDim, intersectionBlockDim>>>();

                //cudaDeviceSynchronize();



                //cudaDeviceSynchronize();

                
                }

            }
