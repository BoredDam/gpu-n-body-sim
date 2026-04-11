#include <CL/cl.h>
#include <math.h>

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
        bodies[i].s4 = -sin(i) * 4.5;
        bodies[i].s5 = cos(i) * 4.5;
        bodies[i].s6 = 0;
        bodies[i].s7 = 1.0;
    }
    
    return bodies;
}