/*
 * Simple 3D Raytracer Outputting a PPM file
 * Optimized for modern Intel CPUs (e.g., i9) with OpenMP multithreading.
 *
 * Compile with:
 *    gcc -O3 -march=native -fopenmp -o raytrace raytrace.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Image dimensions */
#define WIDTH 800
#define HEIGHT 600

/* Maximum ray recursion depth */
#define MAX_DEPTH 3

/* Structure for 3D vectors */
typedef struct {
    double x, y, z;
} Vec3;

/* Inline vector functions for performance */
static inline Vec3 vec_add(const Vec3 a, const Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 vec_sub(const Vec3 a, const Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 vec_scale(const Vec3 v, const double s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

static inline double vec_dot(const Vec3 a, const Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline Vec3 vec_normalize(const Vec3 v) {
    double len = sqrt(vec_dot(v, v));
    return (len > 0) ? vec_scale(v, 1.0/len) : v;
}

static inline Vec3 vec_mult(const Vec3 a, const Vec3 b) {
    return (Vec3){a.x * b.x, a.y * b.y, a.z * b.z};
}

/* Ray structure */
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

/* Sphere structure */
typedef struct {
    Vec3 center;
    double radius;
    Vec3 color;       // Base color of the sphere
    double reflectivity;
} Sphere;

/* Scene: We create an array of spheres. */
#define NUM_SPHERES 3
Sphere spheres[NUM_SPHERES] = {
    // A red sphere
    { { 0.0, -0.5, -3.0 }, 0.5, {1.0, 0.0, 0.0}, 0.3 },
    // A blue sphere
    { { -1.0, 0.0, -4.0 }, 1.0, {0.0, 0.0, 1.0}, 0.5 },
    // A green sphere (large floor)
    { { 0.0, -5001.0, 0.0 }, 5000.0, {0.2, 0.8, 0.2}, 0.0 }
};

/* Function to compute ray-sphere intersection.
   Returns the distance along the ray to the intersection, or -1 if no hit.
*/
double intersect_sphere(const Ray ray, const Sphere sphere) {
    Vec3 oc = vec_sub(ray.origin, sphere.center);
    double a = vec_dot(ray.direction, ray.direction);
    double b = 2.0 * vec_dot(oc, ray.direction);
    double c = vec_dot(oc, oc) - sphere.radius * sphere.radius;
    double discriminant = b*b - 4*a*c;
    if (discriminant < 0)
        return -1.0;
    double sqrt_disc = sqrt(discriminant);
    double t1 = (-b - sqrt_disc) / (2.0 * a);
    double t2 = (-b + sqrt_disc) / (2.0 * a);
    if (t1 > 1e-4) return t1;
    if (t2 > 1e-4) return t2;
    return -1.0;
}

/* Trace function: returns the color seen along the ray.
   Implements simple diffuse shading and recursive reflections.
*/
Vec3 trace(const Ray ray, int depth) {
    double t_min = 1e20;
    int hit_index = -1;
    for (int i = 0; i < NUM_SPHERES; i++) {
        double t = intersect_sphere(ray, spheres[i]);
        if (t > 0 && t < t_min) {
            t_min = t;
            hit_index = i;
        }
    }

    if (hit_index == -1) {
        // Background color (gradient)
        double t = 0.5 * (ray.direction.y + 1.0);
        return vec_add(vec_scale((Vec3){0.5, 0.7, 1.0}, t), vec_scale((Vec3){1.0, 1.0, 1.0}, 1.0 - t));
    }

    Sphere hitSphere = spheres[hit_index];
    Vec3 hitPoint = vec_add(ray.origin, vec_scale(ray.direction, t_min));
    Vec3 normal = vec_normalize(vec_sub(hitPoint, hitSphere.center));

    // Simple diffuse shading from a light source positioned at (5, 5, -10)
    Vec3 lightPos = {5.0, 5.0, -10.0};
    Vec3 lightDir = vec_normalize(vec_sub(lightPos, hitPoint));
    double diffuse = fmax(0.0, vec_dot(normal, lightDir));

    Vec3 localColor = vec_scale(hitSphere.color, diffuse);

    // Reflection (if depth allows and sphere is reflective)
    if (depth < MAX_DEPTH && hitSphere.reflectivity > 0.0) {
        // Reflect the ray direction
        Vec3 reflectDir = vec_sub(ray.direction, vec_scale(normal, 2.0 * vec_dot(ray.direction, normal)));
        reflectDir = vec_normalize(reflectDir);
        Ray reflectRay = { hitPoint, reflectDir };
        Vec3 reflectColor = trace(reflectRay, depth + 1);
        // Mix the reflection and local color
        localColor = vec_add(vec_scale(localColor, 1.0 - hitSphere.reflectivity),
                             vec_scale(reflectColor, hitSphere.reflectivity));
    }

    return localColor;
}

int main(void) {
    const int image_width = WIDTH;
    const int image_height = HEIGHT;
    FILE *fp = fopen("gpt-output.ppm", "wb");
    if (!fp) {
        perror("Failed to open output file");
        return EXIT_FAILURE;
    }
    
    // Write the PPM header (P6 binary format)
    fprintf(fp, "P6\n%d %d\n255\n", image_width, image_height);
    
    // Camera setup
    Vec3 camera_origin = {0.0, 0.0, 0.0};
    double viewport_height = 2.0;
    double viewport_width = (double)image_width / image_height * viewport_height;
    double focal_length = 1.0;
    
    Vec3 horizontal = { viewport_width, 0.0, 0.0 };
    Vec3 vertical   = { 0.0, viewport_height, 0.0 };
    Vec3 lower_left_corner = vec_sub(vec_sub(vec_sub(camera_origin,
                                                     vec_scale(horizontal, 0.5)),
                                                     vec_scale(vertical, 0.5)),
                                                     (Vec3){0.0, 0.0, focal_length});
    
    // Allocate a buffer for the image data
    unsigned char *image = malloc(3 * image_width * image_height);
    if (!image) {
        perror("Failed to allocate image buffer");
        return EXIT_FAILURE;
    }
    
    // Render loop: use OpenMP to parallelize over rows.
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            // Compute u,v for the pixel (center of pixel sample)
            double u = (i + 0.5) / image_width;
            double v = ((image_height - j - 0.5)) / image_height;
            
            // Compute ray direction for current pixel
            Vec3 direction = vec_add(lower_left_corner,
                             vec_add(vec_scale(horizontal, u),
                                     vec_scale(vertical, v)));
            direction = vec_sub(direction, camera_origin);
            direction = vec_normalize(direction);
            
            Ray ray = { camera_origin, direction };
            Vec3 pixel_color = trace(ray, 0);
            
            // Gamma correction (gamma = 2)
            pixel_color.x = sqrt(pixel_color.x);
            pixel_color.y = sqrt(pixel_color.y);
            pixel_color.z = sqrt(pixel_color.z);
            
            // Clamp and convert to [0,255]
            int ir = (int)(fmin(1.0, pixel_color.x) * 255.99);
            int ig = (int)(fmin(1.0, pixel_color.y) * 255.99);
            int ib = (int)(fmin(1.0, pixel_color.z) * 255.99);
            
            int index = (j * image_width + i) * 3;
            image[index    ] = (unsigned char) ir;
            image[index + 1] = (unsigned char) ig;
            image[index + 2] = (unsigned char) ib;
        }
    }
    
    // Write the image data to file
    fwrite(image, 3, image_width * image_height, fp);
    fclose(fp);
    free(image);
    
    printf("Rendered image saved to output.ppm\n");
    return EXIT_SUCCESS;
}
