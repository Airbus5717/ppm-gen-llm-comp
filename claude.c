#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 center;
    float radius;
    Vec3 color;
    float specular;
    float reflective;
} Sphere;

typedef struct {
    Vec3 position;
    Vec3 direction;
} Ray;

typedef struct {
    Vec3 position;
    float intensity;
} Light;

#define WIDTH 1024
#define HEIGHT 768
#define MAX_DEPTH 5
#define SPHERE_COUNT 4
#define LIGHT_COUNT 2

Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 vec3_mul(Vec3 v, float t) {
    return (Vec3){v.x * t, v.y * t, v.z * t};
}

float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_normalize(Vec3 v) {
    float len = sqrtf(vec3_dot(v, v));
    return vec3_mul(v, 1.0f / len);
}

Vec3 vec3_reflect(Vec3 v, Vec3 n) {
    return vec3_sub(v, vec3_mul(n, 2.0f * vec3_dot(v, n)));
}

float intersect_sphere(Ray ray, Sphere sphere) {
    Vec3 oc = vec3_sub(ray.position, sphere.center);
    float a = vec3_dot(ray.direction, ray.direction);
    float b = 2.0f * vec3_dot(oc, ray.direction);
    float c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) return -1;
    
    float t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return t > 0 ? t : -1;
}

Vec3 trace(Ray ray, Sphere* spheres, Light* lights, int depth) {
    if (depth >= MAX_DEPTH) return (Vec3){0, 0, 0};
    
    float closest_t = INFINITY;
    Sphere* closest_sphere = NULL;
    
    // Find closest intersection
    for (int i = 0; i < SPHERE_COUNT; i++) {
        float t = intersect_sphere(ray, spheres[i]);
        if (t > 0 && t < closest_t) {
            closest_t = t;
            closest_sphere = &spheres[i];
        }
    }
    
    if (!closest_sphere) return (Vec3){0.2f, 0.3f, 0.5f}; // Background color
    
    Vec3 intersection = vec3_add(ray.position, vec3_mul(ray.direction, closest_t));
    Vec3 normal = vec3_normalize(vec3_sub(intersection, closest_sphere->center));
    
    Vec3 color = {0, 0, 0};
    float total_light = 0;
    
    // Calculate lighting
    #pragma omp simd
    for (int i = 0; i < LIGHT_COUNT; i++) {
        Vec3 light_dir = vec3_normalize(vec3_sub(lights[i].position, intersection));
        float diffuse = fmaxf(0, vec3_dot(normal, light_dir));
        
        // Check for shadows
        Ray shadow_ray = {intersection, light_dir};
        int in_shadow = 0;
        
        for (int j = 0; j < SPHERE_COUNT; j++) {
            float t = intersect_sphere(shadow_ray, spheres[j]);
            if (t > 0.001f) {
                in_shadow = 1;
                break;
            }
        }
        
        if (!in_shadow) {
            total_light += diffuse * lights[i].intensity;
            
            // Specular reflection
            if (closest_sphere->specular > 0) {
                Vec3 reflected = vec3_reflect(vec3_mul(light_dir, -1), normal);
                float specular = powf(fmaxf(0, vec3_dot(reflected, vec3_mul(ray.direction, -1))), 
                                    closest_sphere->specular);
                total_light += specular * lights[i].intensity * 0.5f;
            }
        }
    }
    
    color = vec3_mul(closest_sphere->color, total_light);
    
    // Calculate reflection
    if (closest_sphere->reflective > 0 && depth < MAX_DEPTH) {
        Vec3 reflected_dir = vec3_reflect(ray.direction, normal);
        Ray reflected_ray = {intersection, reflected_dir};
        Vec3 reflected_color = trace(reflected_ray, spheres, lights, depth + 1);
        color = vec3_add(vec3_mul(color, 1 - closest_sphere->reflective),
                        vec3_mul(reflected_color, closest_sphere->reflective));
    }
    
    // Clamp colors
    color.x = fminf(1.0f, fmaxf(0.0f, color.x));
    color.y = fminf(1.0f, fmaxf(0.0f, color.y));
    color.z = fminf(1.0f, fmaxf(0.0f, color.z));
    
    return color;
}

int main() {
    // Scene setup
    Sphere spheres[SPHERE_COUNT] = {
        {{0, -1, 3}, 1, {1, 0, 0}, 500, 0.2f},     // Red sphere
        {{2, 0, 4}, 1, {0, 1, 0}, 500, 0.3f},      // Green sphere
        {{-2, 0, 4}, 1, {0, 0, 1}, 10, 0.4f},      // Blue sphere
        {{0, -5001, 0}, 5000, {1, 1, 0}, 1000, 0.5f} // Yellow ground
    };
    
    Light lights[LIGHT_COUNT] = {
        {{2, 1, 0}, 0.6f},      // Point light 1
        {{-2, 1, -1}, 0.3f}     // Point light 2
    };
    
    // Output file setup
    FILE* fp = fopen("claude-output.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    unsigned char *pixels = malloc(WIDTH * HEIGHT * 3);
    
    // Set number of threads for OpenMP
    omp_set_num_threads(16); // Adjust based on i9 core count
    
    // Main rendering loop
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float fov = M_PI / 3;
            float aspect_ratio = (float)WIDTH / HEIGHT;
            float px = (2 * ((x + 0.5f) / WIDTH) - 1) * tanf(fov / 2) * aspect_ratio;
            float py = (1 - 2 * ((y + 0.5f) / HEIGHT)) * tanf(fov / 2);
            
            Ray ray = {{0, 0, 0}, vec3_normalize((Vec3){px, py, 1})};
            Vec3 color = trace(ray, spheres, lights, 0);
            
            int index = (y * WIDTH + x) * 3;
            pixels[index] = (unsigned char)(color.x * 255);
            pixels[index + 1] = (unsigned char)(color.y * 255);
            pixels[index + 2] = (unsigned char)(color.z * 255);
        }
    }
    
    fwrite(pixels, 1, WIDTH * HEIGHT * 3, fp);
    fclose(fp);
    free(pixels);
    
    return 0;
}
