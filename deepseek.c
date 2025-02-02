#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    float x, y, z;
} vec3;

static inline vec3 vec3_add(vec3 a, vec3 b) {
    return (vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline vec3 vec3_sub(vec3 a, vec3 b) {
    return (vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline vec3 vec3_mul(vec3 a, float s) {
    return (vec3){a.x * s, a.y * s, a.z * s};
}

static inline float vec3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline vec3 vec3_cross(vec3 a, vec3 b) {
    return (vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline float vec3_length(vec3 a) {
    return sqrtf(vec3_dot(a, a));
}

static inline vec3 vec3_normalize(vec3 a) {
    return vec3_mul(a, 1.0f / vec3_length(a));
}

typedef struct {
    vec3 origin;
    vec3 dir;
} Ray;

typedef struct {
    vec3 center;
    float radius;
    vec3 color;
    float reflectivity;
    float specular;
} Sphere;

vec3 light_dir;

float sphere_intersect(const Sphere *s, const Ray *r) {
    vec3 oc = vec3_sub(r->origin, s->center);
    float a = vec3_dot(r->dir, r->dir);
    float b = 2.0f * vec3_dot(oc, r->dir);
    float c = vec3_dot(oc, oc) - s->radius*s->radius;
    float disc = b*b - 4*a*c;
    
    if (disc < 0) return 0.0f;
    float sqrt_disc = sqrtf(disc);
    float t = (-b - sqrt_disc)/(2*a);
    if (t > 0.001f) return t;
    t = (-b + sqrt_disc)/(2*a);
    return t > 0.001f ? t : 0.0f;
}

vec3 trace(const Ray *ray, const Sphere *spheres, int num_spheres, int depth) {
    if (depth > 3) return (vec3){0.0f, 0.0f, 0.0f};

    float closest_t = 0.0f;
    const Sphere *hit_sphere = NULL;
    vec3 hit_point, hit_normal;

    // Intersection detection
    for (int i = 0; i < num_spheres; ++i) {
        float t = sphere_intersect(&spheres[i], ray);
        if (t > 0.0f && (closest_t == 0.0f || t < closest_t)) {
            closest_t = t;
            hit_sphere = &spheres[i];
        }
    }

    if (!hit_sphere) {
        // Sky gradient
        float t = 0.5f*(ray->dir.y + 1.0f);
        return (vec3){1.0f - t + 0.5f*t, 1.0f - t + 0.7f*t, 1.0f - t + 1.0f*t};
    }

    // Calculate surface properties
    hit_point = vec3_add(ray->origin, vec3_mul(ray->dir, closest_t));
    hit_normal = vec3_normalize(vec3_sub(hit_point, hit_sphere->center));

    // Shadow calculation
    vec3 shadow_orig = vec3_add(hit_point, vec3_mul(hit_normal, 0.001f));
    Ray shadow_ray = {shadow_orig, light_dir};
    int in_shadow = 0;
    for (int i = 0; i < num_spheres; ++i) {
        if (sphere_intersect(&spheres[i], &shadow_ray) > 0.001f) {
            in_shadow = 1;
            break;
        }
    }

    // Lighting calculation
    float diffuse = fmaxf(vec3_dot(hit_normal, light_dir), 0.0f);
    vec3 view_dir = vec3_normalize(vec3_sub(ray->origin, hit_point));
    vec3 reflect_dir = vec3_sub(ray->dir, vec3_mul(hit_normal, 2.0f*vec3_dot(ray->dir, hit_normal)));
    float specular = powf(fmaxf(vec3_dot(view_dir, reflect_dir), 0.0f), hit_sphere->specular);

    vec3 color = vec3_mul(hit_sphere->color, 0.1f + diffuse*0.9f*(1-in_shadow));
    color = vec3_add(color, vec3_mul((vec3){1.0f, 1.0f, 1.0f}, specular*0.5f));

    // Reflection
    if (hit_sphere->reflectivity > 0.0f) {
        Ray reflect_ray = {vec3_add(hit_point, vec3_mul(hit_normal, 0.001f)), 
                          vec3_normalize(reflect_dir)};
        vec3 reflected = trace(&reflect_ray, spheres, num_spheres, depth+1);
        color = vec3_add(vec3_mul(color, 1.0f - hit_sphere->reflectivity),
                        vec3_mul(reflected, hit_sphere->reflectivity));
    }

    return color;
}

int main() {
    const int width = 1920;
    const int height = 1080;
    const float aspect = (float)width/height;
    unsigned char *img = malloc(width*height*3);

    // Camera setup
    vec3 cam_pos = {3.0f, 2.0f, 4.0f};
    vec3 look_at = {0.0f, 0.5f, 0.0f};
    vec3 up = {0.0f, 1.0f, 0.0f};
    light_dir = vec3_normalize((vec3){0.4f, -1.0f, 0.4f});

    // Scene setup
    Sphere spheres[] = {
        {{0.0f, -1000.5f, 0.0f}, 1000.0f, {0.8f, 0.8f, 0.8f}, 0.2f, 50.0f},
        {{0.0f, 0.5f, 0.0f}, 0.5f, {0.8f, 0.3f, 0.2f}, 0.5f, 100.0f},
        {{-1.2f, 0.3f, 0.8f}, 0.3f, {0.2f, 0.8f, 0.3f}, 0.3f, 50.0f},
        {{1.0f, 0.4f, -0.5f}, 0.4f, {0.3f, 0.2f, 0.8f}, 0.3f, 50.0f}
    };
    const int num_spheres = sizeof(spheres)/sizeof(Sphere);

    // Camera vectors
    vec3 w = vec3_normalize(vec3_sub(cam_pos, look_at));
    vec3 u = vec3_normalize(vec3_cross(up, w));
    vec3 v = vec3_cross(w, u);
    float fov = 45.0f * M_PI/180.0f;
    float h = tanf(fov/2);

    // Parallel rendering
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float px = (2*(x + 0.5f)/width - 1) * aspect * h;
            float py = (1 - 2*(y + 0.5f)/height) * h;
            
            vec3 ray_dir = vec3_normalize(vec3_add(
                vec3_add(vec3_mul(u, px), vec3_mul(v, py)),
                vec3_mul(w, -1.0f)
            ));

            Ray ray = {cam_pos, ray_dir};
            vec3 color = trace(&ray, spheres, num_spheres, 0);

            int idx = (y*width + x)*3;
            img[idx]   = (unsigned char)(fminf(color.x*255, 255));
            img[idx+1] = (unsigned char)(fminf(color.y*255, 255));
            img[idx+2] = (unsigned char)(fminf(color.z*255, 255));
        }
    }

    // Save PPM
    FILE *fp = fopen("deepseek-output.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(img, 1, width*height*3, fp);
    fclose(fp);
    free(img);

    return 0;
}
