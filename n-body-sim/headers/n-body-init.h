#include <CL/cl.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float randf_range(float a, float b) {
    return a + (b - a) * ((float)rand() / (float)RAND_MAX);
}

void *init_bodies_two_colliding_galaxies(cl_float8 *bodies, unsigned int body_count, unsigned int seed) {

    /* THIS WAS VIBE CODED FOR TESTING PURPOSE !!!
    since how to build a galaxy is not really in the scope on the studies related
    to this project (GPU parallel programming) I'll keep this until I'll code one myself. 
    have a great day xoxo   
    */
    srand(seed);

    if (body_count < 4) {
        return bodies;
    }

    const float G = 0.05f;

    /* galaxy centers */
    const float galaxy_sep = 80.0f;
    const float g1_cx = -galaxy_sep * 0.5f;
    const float g1_cy = 0.0f;
    const float g1_cz = 0.0f;

    const float g2_cx =  galaxy_sep * 0.5f;
    const float g2_cy = 0.0f;
    const float g2_cz = 0.0f;

    /* bulk velocities: moving toward each other */
    const float g1_vx =  1.2f;
    const float g1_vy =  0.15f;
    const float g1_vz =  0.0f;

    const float g2_vx = -1.2f;
    const float g2_vy = -0.15f;
    const float g2_vz =  0.0f;

    /* central masses */
    const float core_mass_1 = 8000.0f;
    const float core_mass_2 = 8000.0f;

    /* disk parameters */
    const float r_min = 8.0f;
    const float r_max = 28.0f;
    const float z_spread = 1.5f;

    /* split particles between the two galaxies */
    unsigned int n1 = body_count / 2;
    unsigned int n2 = body_count - n1;

    if (n1 < 2 || n2 < 2) {
        return bodies;
    }

    /* core 1 */
    bodies[0].s0 = 0.0f;
    bodies[0].s1 = g1_cx;
    bodies[0].s2 = g1_cy;
    bodies[0].s3 = g1_cz;
    bodies[0].s4 = g1_vx;
    bodies[0].s5 = g1_vy;
    bodies[0].s6 = g1_vz;
    bodies[0].s7 = core_mass_1;

    /* core 2 */
    bodies[n1].s0 = 0.0f;
    bodies[n1].s1 = g2_cx;
    bodies[n1].s2 = g2_cy;
    bodies[n1].s3 = g2_cz;
    bodies[n1].s4 = g2_vx;
    bodies[n1].s5 = g2_vy;
    bodies[n1].s6 = g2_vz;
    bodies[n1].s7 = core_mass_2;

    /* stars of galaxy 1 */
    for (unsigned int i = 1; i < n1; i++) {
        float theta = randf_range(0.0f, 2.0f * (float)M_PI);
        float r = randf_range(r_min, r_max);
        float z = randf_range(-z_spread, z_spread);

        float x = g1_cx + r * cosf(theta);
        float y = g1_cy + r * sinf(theta);
        float vz = randf_range(-0.05f, 0.05f);

        /* circular-orbit estimate around core 1 */
        float v_circ = sqrtf(G * core_mass_1 / r);

        /* tangential direction */
        float tx = -sinf(theta);
        float ty =  cosf(theta);

        /* small random noise helps make it look less artificial */
        float noise = 0.08f * v_circ;

        bodies[i].s0 = 0.0f;
        bodies[i].s1 = x;
        bodies[i].s2 = y;
        bodies[i].s3 = z;
        bodies[i].s4 = g1_vx + tx * v_circ + randf_range(-noise, noise);
        bodies[i].s5 = g1_vy + ty * v_circ + randf_range(-noise, noise);
        bodies[i].s6 = g1_vz + vz;
        bodies[i].s7 = 1.0f;
    }

    /* stars of galaxy 2 */
    for (unsigned int i = n1 + 1; i < body_count; i++) {
        float theta = randf_range(0.0f, 2.0f * (float)M_PI);
        float r = randf_range(r_min, r_max);
        float z = randf_range(-z_spread, z_spread);

        float x = g2_cx + r * cosf(theta);
        float y = g2_cy + r * sinf(theta);
        float vz = randf_range(-0.05f, 0.05f);

        float v_circ = sqrtf(G * core_mass_2 / r);

        /* opposite disk rotation for nicer collision visuals */
        float tx =  sinf(theta);
        float ty = -cosf(theta);

        float noise = 0.08f * v_circ;

        bodies[i].s0 = 0.0f;
        bodies[i].s1 = x;
        bodies[i].s2 = y;
        bodies[i].s3 = z;
        bodies[i].s4 = g2_vx + tx * v_circ + randf_range(-noise, noise);
        bodies[i].s5 = g2_vy + ty * v_circ + randf_range(-noise, noise);
        bodies[i].s6 = g2_vz + vz;
        bodies[i].s7 = 1.0f;
    }

    return bodies;
}


void *init_bodies_demo(cl_float8 *bodies, unsigned int body_count, float center_distance, unsigned int seed) {
    
    srand(seed);
    
    /*inits a galaxy-like particle system*/
    bodies[0].s0 = 0;
    bodies[0].s1 = 0;
    bodies[0].s2 = 0;
    bodies[0].s3 = 0;
    bodies[0].s4 = 0;
    bodies[0].s5 = 0;
    bodies[0].s6 = 0;
    bodies[0].s7 = 10000;

    for (int i = 1; i < body_count; i++) {  
        bodies[i].s0 = 0;
        bodies[i].s1 = cos(i) * (center_distance + rand() % 9);
        bodies[i].s2 = sin(i) * (center_distance + rand() % 9);
        bodies[i].s3 = rand() % 4 - 2;
        bodies[i].s4 = -sin(i) * center_distance / 3;
        bodies[i].s5 = cos(i) * center_distance / 3;
        bodies[i].s6 = 0;
        bodies[i].s7 = 1.0;
    }
    
    return bodies;
}