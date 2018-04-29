#include "triangle.h"

#include "CMU462/CMU462.h"
#include "GL/glew.h"

namespace CMU462 {
namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, vector<size_t>& v) : mesh(mesh), v(v) {}
Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3)
    : mesh(mesh), v1(v1), v2(v2), v3(v3) {}

BBox Triangle::get_bbox() const {
  // TODO (PathTracer):
  // compute the bounding box of the triangle

	auto &psns = mesh->positions;
	Vector3D p1 = psns[v1];
	Vector3D p2 = psns[v2];
	Vector3D p3 = psns[v3];

	double maxX = (p1.x > p2.x) ? p1.x : p2.x;
	double maxY = (p1.y > p2.y) ? p1.y : p2.y;
	double maxZ = (p1.z > p2.z) ? p1.z : p2.z;
	maxX = (maxX > p3.x) ? maxX : p3.x;
	maxY = (maxY > p3.y) ? maxY : p3.y;
	maxZ = (maxZ > p3.z) ? maxZ : p3.z;

	double minX = (p1.x < p2.x) ? p1.x : p2.x;
	double minY = (p1.y < p2.y) ? p1.y : p2.y;
	double minZ = (p1.z < p2.z) ? p1.z : p2.z;
	minX = (minX < p3.x) ? minX : p3.x;
	minY = (minY < p3.y) ? minY : p3.y;
	minZ = (minZ < p3.z) ? minZ : p3.z;

	// Minor padding to make sure there's no 'z-fighting'
	// for axis aligned triangles.
#define PADDING 1e-3
	minX -= PADDING;
	maxX += PADDING;
	minY -= PADDING;
	maxY += PADDING;
	minZ -= PADDING;
	maxZ += PADDING;

  return BBox(minX, minY, minZ, maxX, maxY, maxZ);
}

bool Triangle::intersect(const Ray& r) const {
	
	auto &psns = mesh->positions;
	Vector3D p1 = psns[v1];
	Vector3D p2 = psns[v2];
	Vector3D p3 = psns[v3];

	/*auto n = cross(p1 - p2, p1 - p3);
	n.normalize();

	auto c = dot(n, p1);

	// Calculate distance to plane.
	auto dtoplane = -(dot(r.o, n) - c);
	if (dtoplane < 0)
		return false;

	// Compute angle between ray direction and the normal.
	float dnangle = dot(r.d, n);

	// Check for corner case where the ray is almost parallel.
	if (abs(dnangle) < 1e-5)
		return false;

	// Find point on plane.
	auto pt = (dtoplane / dnangle) * r.d + r.o;

	// Find the triangle intersection.
	auto x1 = p1 - pt;
	auto x2 = p2 - pt;
	auto x3 = p3 - pt;

	// Now find the cross products.
	auto x12 = cross(x1, x2);
	auto x23 = cross(x2, x3);
	auto x31 = cross(x3, x1);

	if (dot(x12, x23) >= 0 && dot(x23, x31) >= 0 && dot(x31, x12) >= 0)
		return true;

  return false;*/

	auto s = r.o - p1;
	auto e1 = p2 - p1;
	auto e2 = p3 - p1;

	auto t1 = cross(e1, r.d);
	auto t2 = cross(s, e2);

	auto den = 1.0 / (dot(t1, e2));

	auto u = dot(-t2, r.d) * den;
	auto v = dot(t1, s) * den;
	auto t = dot(-t2, r.d) * den;

	if (abs(den) < 1e-5) {
		return false;
	}

	if (u < 0 || v < 0 ||
		u > 1 || v > 1 ||
		t < r.min_t || t > r.max_t ||
		(u + v < 0) || (u + v > 1)) {

		return false;
	}

	return true;
}

bool Triangle::intersect(const Ray& r, Intersection* isect) const {
  // TODO (PathTracer):
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly

	auto &psns = mesh->positions;
	Vector3D p1 = psns[v1];
	Vector3D p2 = psns[v2];
	Vector3D p3 = psns[v3];

	/*auto n = cross(p1 - p3, p1 - p2);
	n.normalize();

	auto c = dot(n, p1);

	// Calculate distance to plane.
	auto dtoplane = -(dot(r.o, n) - c);

	// Compute angle between ray direction and the normal.
	float dnangle = dot(r.d, n);

	// Check for corner case where the ray is almost parallel.
	if (abs(dnangle) < 1e-5)
		return false;
	
	if ((dtoplane / dnangle) < r.min_t && (dtoplane / dnangle) > r.max_t)
		return false;

	// Find point on plane.
	auto pt = (dtoplane / dnangle) * r.d + r.o;

	// Find the triangle intersection.
	auto x1 = p1 - pt;
	auto x2 = p2 - pt;
	auto x3 = p3 - pt;

	// Now find the cross products.
	auto x12 = cross(x1, x2);
	auto x23 = cross(x2, x3);
	auto x31 = cross(x3, x1);

	if (dot(x12, x23) >= 0 && dot(x23, x31) >= 0 && dot(x31, x12) >= 0) {
		isect->t = (dtoplane / dnangle);
		isect->primitive = this;
		isect->bsdf = this->get_bsdf();
		isect->n = n * (dtoplane > 0 ? 1 : -1);
		return true;
	}

  return false;*/

	auto s = r.o - p1;
	auto e1 = p2 - p1;
	auto e2 = p3 - p1;

	auto t1 = cross(e1, r.d);
	auto t2 = cross(s, e2);

	auto den = 1.0 / (dot(t1, e2));
	
	auto u = dot(-t2, r.d) * den;
	auto v = dot(t1, s) * den;
	auto t = dot(-t2, e1) * den;

	if (abs(den) > 1e+10) {
		return false;
	}

	if (u < 0 || v < 0 || 
			u > 1 || v > 1 || 
			t < r.min_t || t > r.max_t ||
			(u + v < 0) || (u + v > 1) ) {

		return false;
	}

	Vector3D normal1 = mesh->normals[v1];
	Vector3D normal2 = mesh->normals[v2];
	Vector3D normal3 = mesh->normals[v3];
	auto n = cross(e1, e2);
	isect->n = u * normal2 + v * normal3 + (1 - u - v) * normal1;
	isect->n = isect->n * (dot(s, isect->n) > 0 ? 1 : -1);

	isect->n.normalize();

	isect->t = t;
	isect->primitive = this;
	isect->bsdf = get_bsdf();

	return true;
}

void Triangle::draw(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_TRIANGLES);
  glVertex3d(mesh->positions[v1].x, mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x, mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x, mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

void Triangle::positions(Vector3D &t0, Vector3D &t1, Vector3D &t2) {
    t0 = mesh->positions[v1];
    t1 = mesh->positions[v2];
    t2 = mesh->positions[v3];
}

void Triangle::normals(Vector3D &t0, Vector3D &t1, Vector3D &t2) {
    t0 = mesh->normals[v1];
    t1 = mesh->normals[v2];
    t2 = mesh->normals[v3];
}

void Triangle::drawOutline(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_LOOP);
  glVertex3d(mesh->positions[v1].x, mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x, mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x, mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

}  // namespace StaticScene
}  // namespace CMU462
