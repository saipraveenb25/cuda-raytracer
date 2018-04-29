#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CMU462 {
namespace StaticScene {


	bool cmpX(const Primitive* a, const Primitive* b) {
		return a->get_bbox().centroid().x < b->get_bbox().centroid().x;
	}
	bool cmpY(const Primitive* a, const Primitive* b) {
		return a->get_bbox().centroid().y < b->get_bbox().centroid().y;
	}
	bool cmpZ(const Primitive* a, const Primitive* b) {
		return a->get_bbox().centroid().z < b->get_bbox().centroid().z;
	}

	bool cmpdX(double d, const Primitive* b) {
		return d < b->get_bbox().centroid().x;
	}
	bool cmpdY(double d, const Primitive* b) {
		return d < b->get_bbox().centroid().y;
	}
	bool cmpdZ(double d, const Primitive* b) {
		return d < b->get_bbox().centroid().z;
	}

	bool cmpdX2(const Primitive* b, double d) {
		return b->get_bbox().centroid().x < d;
	}
	bool cmpdY2(const Primitive* b, double d) {
		return b->get_bbox().centroid().y < d;
	}
	bool cmpdZ2(const Primitive* b, double d) {
		return b->get_bbox().centroid().z < d;
	}

	BVHNode* splitBVHNode(std::vector<Primitive *> &_primitives,
												//const std::vector<Primitive *> &_primitives1,
												//const std::vector<Primitive *> &_primitives2,	
		size_t max_leaf_size, size_t start, size_t end, BBox _bb, int dim) {

		auto thisnode = new BVHNode(_bb, start, (end - start));
		if (end - start <= max_leaf_size) {
			return thisnode;
		}

		double total_sa = _bb.surface_area();
		if (total_sa < 1e-15)
			return thisnode;

		float current_cost = 2 * (end - start);
		float bestcost = current_cost;
		int besti = 0;
		float bestk = 0;
		BBox boxl;
		BBox boxr;
		for (int i = 0; i < 3; i++) {
			switch (i) {
			case 0:
				std::sort(_primitives.begin() + start, _primitives.begin() + end, cmpX);
				break;
			case 1:
				std::sort(_primitives.begin() + start, _primitives.begin() + end, cmpY);
				break;
			case 2:
				std::sort(_primitives.begin() + start, _primitives.begin() + end, cmpZ);
				break;
			}

			std::vector<BBox> ltorbboxs;
			std::vector<BBox> rtolbboxs;
			BBox b1;
			BBox b2;

			double startval;
			double endval;
			switch (i)
			{
			case 0:
				startval = _primitives[start]->get_bbox().centroid().x;
				endval = _primitives[end - 1]->get_bbox().centroid().x;
				break;
			case 1:
				startval = _primitives[start]->get_bbox().centroid().y;
				endval = _primitives[end - 1]->get_bbox().centroid().y;
				break;
			case 2:
				startval = _primitives[start]->get_bbox().centroid().z;
				endval = _primitives[end - 1]->get_bbox().centroid().z;
				break;
			default:
				break;
			}

			//std::cout << startval << " " << endval << std::endl;
			int numparts = 12;

			int lastidx = start;
			std::vector<int> counts;
			std::vector<int> indices;
			// Compute partition boxes.
			for (long part = 1; part <= numparts; part++) {
				// Compute the partition requirement here.
				double divider = startval + part * ((endval - startval) / (numparts+1));
				
				// Get the index.
				int idx = 1;
				switch (i) {
				case 0:
					idx = std::upper_bound(_primitives.begin() + start, _primitives.begin() + end, divider, cmpdX) - _primitives.begin();
					break;
				case 1:
					idx = std::upper_bound(_primitives.begin() + start, _primitives.begin() + end, divider, cmpdY) - _primitives.begin();
					break;
				case 2:
					idx = std::upper_bound(_primitives.begin() + start, _primitives.begin() + end, divider, cmpdZ) - _primitives.begin();
					break;
				}
				//std::cout << idx << std::endl;
				for (int j = lastidx; j < idx; j++) {
					b1.expand(_primitives[j]->get_bbox());
				}
				
				counts.push_back(idx - lastidx);
				indices.push_back(idx);
				lastidx = idx;
				ltorbboxs.push_back(b1);
			}

			lastidx = end;
			// Compute partition boxes.
			for (long part = 1; part <= numparts; part++) {
				// Compute the partition requirement here.
				double divider = endval - part * ((endval - startval) / (numparts + 1));

				// Get the index.
				int idx = 1;
				switch (i) {
				case 0:
					idx = std::lower_bound(_primitives.begin() + start, _primitives.begin() + end, divider, cmpdX2) - _primitives.begin();
					break;
				case 1:
					idx = std::lower_bound(_primitives.begin() + start, _primitives.begin() + end, divider, cmpdY2) - _primitives.begin();
					break;
				case 2:
					idx = std::lower_bound(_primitives.begin() + start, _primitives.begin() + end, divider, cmpdZ2) - _primitives.begin();
					break;
				}

				for (int j = lastidx - 1; j >= idx; j--) {
					b2.expand(_primitives[j]->get_bbox());
				}

				lastidx = idx;
				rtolbboxs.push_back(b2);
			}
			
			double mincost = current_cost;
			size_t mink = 1;
			BBox minboxl;
			BBox minboxr;
			//for (size_t k = start; k < end; k++) {
			for(size_t k = 0; k < numparts; k++){
				//size_t n1 = k - start;
				//size_t n2 = (end - k) - 1;
				int count = indices[k] - start;
				int count2 = (end - start) - count;
				double sa1 = ltorbboxs[k].surface_area();
				double sa2 = rtolbboxs[numparts - k - 1].surface_area();

				double cost = 5 + (sa1 / total_sa) * count * 2 + (sa2 / total_sa) * count2 * 2;
				/*std::cout << "k:" << k << " Cost: " << cost << " sa1:" << sa1 
									<< " sa2:" << sa2 
									<< " count: " << count
									<< " count2: " << count2 << std::endl;*/

				if (mincost > cost) {
					mincost = cost;
					mink = indices[k];
					minboxl = ltorbboxs[k];
					minboxr = rtolbboxs[numparts - k - 1];
				}

			}
			if (mincost == current_cost) {
				mink = indices[1];
				minboxl = ltorbboxs[1];
				minboxr = rtolbboxs[numparts - 2];
			}

			if (mincost < bestcost) {
				bestcost = mincost;
				bestk = mink;
				besti = i;
				boxl = minboxl;
				boxr = minboxr;
			}
		}
		//std::vector<Primitive* > prims(_primitives.begin(), _primitives.end());
		
		if (bestcost == current_cost) {
			// Don't bother splitting this node. It's clearly going to make it worse.
			return thisnode;
		}

		switch (besti) {
		case 0:
			std::sort(_primitives.begin() + start, _primitives.begin() + end, cmpX);
			break;
		case 1:
			std::sort(_primitives.begin() + start, _primitives.begin() + end, cmpY);
			break;
		case 2:
			std::sort(_primitives.begin() + start, _primitives.begin() + end, cmpZ);
			break;
		}

		thisnode->l = splitBVHNode(_primitives, max_leaf_size, start, bestk, boxl, (dim + 1)%3);
		thisnode->r = splitBVHNode(_primitives, max_leaf_size, bestk, end, boxr, (dim + 1)%3);

		return thisnode;
	}


#define TREE_BRANCHES 16
#define DEPTH 4


int BVHSubTree::compress(std::vector<C_BVHSubTree>& tree, int* levelLists, int levelStride, std::vector<int>& levelCounts, int depth, int max_depth) {

    C_BVHSubTree _ctree;
    
    int idx = tree.size();
    tree.push_back(ctree);
    
    C_BVHSubTree *ctree = &tree[idx];
    
    levelLists[depth * levelStride + (levelCounts[depth]++)] = idx;

    //int old_offset = offset;
    //offset += sizeof(C_BVHSubTree);
    ctree->range = this->range;
    ctree->start = this->start;
    if(depth > max_depth) {
        printf("Depth exceeds max depth\n");
        exit(0);
    }

    for(int i = 0; i < TREE_BRANCHES; i++) {
        if(this->outlets[i] != 0) {
            offset = this->outlets[i]->compress(tree, levelLists, levelStride, levelCounts, depth+1, max_depth);
            ctree->outlets[i] = offset;
        } else {
            ctree->outlets[i] = (uint64_t)-1;
        }
        
    }
    
    /*while( st.size() != 0 ){
        auto tree = st.pop();
        levelLists[tree.first * stride + (levelCounts[tree.first]++)] = ;
    }*/
    
    //tree[old_offset] = ctree;
    return idx;
}

BVHSubTree* BVHNode::compactTree() {

    // Do an iterative depth-first search.
    bool done = false;
    std::stack<std::pair<int,BVHNode*> > st;
    
    BVHSubTree* subtree = new BVHSubTree();

    for(int i = 0; i < TREE_BRANCES; i++) {
        subtree->outlets[i] = NULL;
    }

    if(this->l == NULL && this->r == NULL) {
        subtree->range = this->range;
        subtree->start = this->start;
        return subtree;
    }

    int curr = 0;
    st.push(std::make_pair(0,this));
    while(st.size() != 0) {
        std::pair<int,BVHNode*> dn = st.pop();
        int depth = dn->left;
        BVHNode* n = dn->right;
        if(depth == DEPTH - 1) {
            int temp = curr;
            curr++;
            subtree->outlets[temp] = n->compactTree();
            subtree->minl[temp] = n->minl;
            subtree->maxl[temp] = n->maxl;
        }
        
        if(n->l != NULL)
            st.push(std::make_pair(d+1,n->l));
        if(n->r != NULL)
            st.push(std::make_pair(d+1,n->r));
        
        if(n->l == NULL && n->r == NULL && depth != DEPTH - 1) {
            int temp = curr;
            curr++;
            subtree->outlets[temp] = n->compactTree();
            subtree->minl[temp] = n->minl;
            subtree->maxl[temp] = n->maxl;
        }
    }

    return subtree;
}

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {
  this->primitives = _primitives;

  // TODO (PathTracer):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bb;
  for (size_t i = 0; i < primitives.size(); ++i) {
    bb.expand(primitives[i]->get_bbox());
  }

	//std::vector<Primitive *> _primitives1 = _primitives;
	//std::vector<Primitive *> _primitives2 = _primitives;
	//std::vector<Primitive *> _primitives3 = _primitives;

	std::sort(primitives.begin(), primitives.end(), cmpX);
	//std::sort(_primitives2.begin(), _primitives2.end(), cmpY);
	//std::sort(_primitives3.begin(), _primitives3.end(), cmpZ);

	root = splitBVHNode(primitives, max_leaf_size, 0, primitives.size(), bb, 0);
	
	
}

#define MAX_SUBTREES 40000
std::vector<C_BVHSubTree> BVHAccel::compressedTree() {
    //std::vector<C_BVHSubTree> buffer;// = new BVHSubTree[MAX_SUBTREES];
    //BVHSubTree* stree = root->compactTree();
    //return stree->compress(buffer);
}
BVHSubTree* BVHAccel::compactedTree() {
    return root->compactTree();
}

BVHAccel::~BVHAccel() {
  // TODO (PathTracer):
  // Implement a proper destructor for your BVH accelerator aggregate

}

BBox BVHAccel::get_bbox() const { return root->bb; }
/*
int BVHNode::compress(void* cmem, int coffset) {
    C_BVHNode thisnode(this);
    
    int oldoffset = coffset;

    coffset += sizeof(C_BVHNode);

    thisnode.l = coffset;
    thisnode.compress(cmem, coffset);
    if(this->l != NULL)
        coffset += this->l->compress(cmem, coffset);
    
    thisnode.r = coffset;
    if(this->r != NULL)
        coffset += this->r->compress(cmem, coffset)
    
    thisnode.compress(cmem, oldoffset);

    return coffset - oldoffset;
}
*/
bool BVHAccel::intersect(const Ray &ray) const {
  // TODO (PathTracer):
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.

  bool hit = false;
  for (size_t p = 0; p < primitives.size(); ++p) {
    if (primitives[p]->intersect(ray)) hit = true;
  } 

  return hit;
}

bool intersectBBox(BBox bbox, const Ray &ray, Intersection *isect) {
	double t0, t1;
	auto b = bbox.intersect(ray, t0, t1);
	isect->t = t0;
	return b;
}

bool nodeIntersect(BVHNode* node, const Ray &ray,
	Intersection *isect, const std::vector<Primitive*>& _primitives) {

	//std::cout << "Intersecting Node: " << node->start << " " << node->range << std::endl;

	// If this is a leaf, intersect it
	if (node->isLeaf()) {
		//std::cout << "LEAF" << std::endl;
		
		bool hit = false;
		float mint = 1e10;
		Intersection minint;
		
		for (size_t p = 0; p < node->range; ++p) {
			Intersection its;
			if (_primitives[p + node->start]->intersect(ray, &its)) {
				
				hit = true;
				//std::cout << "ITS.T " << its.t << std::endl;
				if (mint > its.t) {
					mint = its.t;
					minint = its;
				}

			}
		}

		if (hit) *isect = minint;

		return hit;
	}

	// Test left bbox.
	Intersection left;
	bool lhit = intersectBBox(node->l->bb, ray, &left);
	Intersection right;
	bool rhit = intersectBBox(node->r->bb, ray, &right);


	// TODO: Fix a max val
	if (!lhit) left.t = INF_D;
	if (!rhit) right.t = INF_D;

	//std::cout << "Lefthit: " << lhit << ", " << left.t << std::endl;
	//std::cout << "Righthit: " << rhit << ", " << right.t << std::endl;

	if (!lhit && !rhit) return false;

	BVHNode* nclose = (left.t < right.t) ? node->l : node->r;
	BVHNode* nfar = (left.t < right.t) ? node->r : node->l;

	double tclose = (left.t < right.t) ? left.t : right.t;
	double tfar = (left.t < right.t) ? right.t : left.t;

	bool hclose = (left.t < right.t) ? lhit : rhit;
	bool hfar = (left.t < right.t) ? rhit : lhit;

	Intersection closer;
	//std::cout << node->start << ":" << node->range << " Checking close " << std::endl;
	bool bclose = nodeIntersect(nclose, ray, &closer, _primitives);
	if (!bclose) closer.t = INF_D;
	//std::cout << node->start << ":" << node->range << " Got close " << bclose << ", " << closer.t << std::endl;

	if (!bclose || (closer.t > tfar)) {
		Intersection far;
		//std::cout << node->start << ":" << node->range << " Checking far " << std::endl;
		bool bfar = hfar ? (nodeIntersect(nfar, ray, &far, _primitives)) : false;
		if (!bfar) far.t = INF_D;
		//std::cout << node->start << ":" << node->range << " Got far " << bfar << ", " << far.t << std::endl;

		if (!bfar && !bclose) return false;

		*isect = (far.t < closer.t) ? far : closer;
		return true;
	}

	if (bclose)
	{
		*isect = closer;
		return true;
	}

	// No hit on either of the boxes.
	return false;
}


bool BVHAccel::intersect(const Ray &ray, Intersection *isect) const {
  // TODO (PathTracer):
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.

	//std::cout << "BVHAccel: Rendering: "<< primitives.size() << std::endl;
  // Old code. 
	/*bool hit = false;
  for (size_t p = 0; p < primitives.size(); ++p) {
    if (primitives[p]->intersect(ray, isect)) hit = true;
  }
	
	return hit;*/
	return nodeIntersect(this->root, ray, isect, this->primitives);
}

}  // namespace StaticScene
}  // namespace CMU462
