#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Constants for image properties
#define WIDTH 800
#define HEIGHT 600
#define SAMPLES 100
#define MAX_DEPTH 50

// Sphere
typedef struct {
    float center[3];
    float radius;
    float color[3];
} Sphere;

// Ray structure
typedef struct {
    float origin[3];
    float direction[3];
} Ray;

// Hit record
typedef struct {
    float t;
    float point[3];
    float normal[3];
    int front_face;
} HitRecord;

// Camera
typedef struct {
    float origin[3];
    float lower_left_corner[3];
    float horizontal[3];
    float vertical[3];
} Camera;

void vec3_normalize(float v[3]) {
    float length = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] /= length; v[1] /= length; v[2] /= length;
}

float vec3_dot(const float v1[3], const float v2[3]) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void vec3_sub(float result[3], const float v1[3], const float v2[3]) {
    result[0] = v1[0] - v2[0];
    result[1] = v1[1] - v2[1];
    result[2] = v1[2] - v2[2];
}

float hit_sphere(const Sphere *sphere, const Ray *r, float t_min, float t_max) {
    float oc[3];
    vec3_sub(oc, r->origin, sphere->center);
    float a = vec3_dot(r->direction, r->direction);
    float half_b = vec3_dot(oc, r->direction);
    float c = vec3_dot(oc, oc) - sphere->radius * sphere->radius;
    float discriminant = half_b*half_b - a*c;
    
    if (discriminant < 0) return -1.0;
    float sqrtd = sqrt(discriminant);
    
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return -1.0;
    }
    return root;
}

int hit_world(const Sphere *world, int world_size, const Ray *r, float t_min, float t_max, HitRecord *rec) {
    HitRecord temp_rec;
    int hit_anything = 0;
    float closest_so_far = t_max;

    for(int i = 0; i < world_size; ++i) {
        float t = hit_sphere(&world[i], r, t_min, closest_so_far);
        if (t > 0.0) {
            hit_anything = 1;
            closest_so_far = t;
            temp_rec.t = t;
            vec3_sub(temp_rec.point, r->origin, r->direction);
            temp_rec.point[0] *= t;
            temp_rec.point[1] *= t;
            temp_rec.point[2] *= t;
            vec3_sub(temp_rec.normal, temp_rec.point, world[i].center);
            vec3_normalize(temp_rec.normal);
            temp_rec.front_face = vec3_dot(r->direction, temp_rec.normal) < 0;
            *rec = temp_rec;
        }
    }
    return hit_anything;
}

void ray_color(const Ray *r, const Sphere *world, int world_size, float out_color[3], int depth) {
    if (depth <= 0) {
        out_color[0] = out_color[1] = out_color[2] = 0;
        return;
    }

    HitRecord rec;
    if (hit_world(world, world_size, r, 0.001, INFINITY, &rec)) {
        for (int i = 0; i < world_size; i++) {
            if (rec.t == hit_sphere(&world[i], r, 0.001, INFINITY)) {
                out_color[0] = world[i].color[0];
                out_color[1] = world[i].color[1];
                out_color[2] = world[i].color[2];
                return;
            }
        }
    }

    float unit_direction[3] = {r->direction[0], r->direction[1], r->direction[2]};
    vec3_normalize(unit_direction);
    float t = 0.5*(unit_direction[1] + 1.0);
    out_color[0] = (1.0-t) + t*0.5; // Lerp from white to blue
    out_color[1] = (1.0-t) + t*0.7;
    out_color[2] = (1.0-t) + t*1.0;
}

int main() {
    FILE *f = fopen("grok-output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
    
    Camera cam = {
        {0, 0, 0}, 
        {-2.0, -1.5, -1.0}, 
        {4.0, 0, 0}, 
        {0, 3.0, 0}
    };

    Sphere world[] = {
        {{0, 0, -1}, 0.5, {0.1, 0.2, 0.5}}, // Blue sphere
        {{0, -100.5, -1}, 100, {1.0, 0.8, 0.8}} // Large ground sphere
    };
    int world_size = sizeof(world) / sizeof(Sphere);

    for (int j = HEIGHT-1; j >= 0; --j) {
        for (int i = 0; i < WIDTH; ++i) {
            float pixel_color[3] = {0, 0, 0};
            for (int s = 0; s < SAMPLES; ++s) {
                float u = (float)(i + drand48()) / (WIDTH-1);
                float v = (float)(j + drand48()) / (HEIGHT-1);
                Ray r = {
                    {cam.origin[0], cam.origin[1], cam.origin[2]},
                    {
                        cam.lower_left_corner[0] + u*cam.horizontal[0] + v*cam.vertical[0] - cam.origin[0],
                        cam.lower_left_corner[1] + u*cam.horizontal[1] + v*cam.vertical[1] - cam.origin[1],
                        cam.lower_left_corner[2] + u*cam.horizontal[2] + v*cam.vertical[2] - cam.origin[2]
                    }
                };
                float sample_color[3];
                ray_color(&r, world, world_size, sample_color, MAX_DEPTH);
                pixel_color[0] += sample_color[0];
                pixel_color[1] += sample_color[1];
                pixel_color[2] += sample_color[2];
            }
            // Gamma correction and scale to 0-255
            pixel_color[0] = sqrt(pixel_color[0] / SAMPLES) * 255.99;
            pixel_color[1] = sqrt(pixel_color[1] / SAMPLES) * 255.99;
            pixel_color[2] = sqrt(pixel_color[2] / SAMPLES) * 255.99;
            fprintf(f, "%d %d %d\n", (int)pixel_color[0], (int)pixel_color[1], (int)pixel_color[2]);
        }
    }
    fclose(f);
    return 0;
}
