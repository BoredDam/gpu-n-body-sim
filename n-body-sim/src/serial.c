#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>


/*constants for array indexing*/
#define COMPONENTS 8
#define TIME 0
#define POS_X 1
#define POS_Y 2
#define POS_Z 3
#define VEL_X 4
#define VEL_Y 5
#define VEL_Z 6
#define MASS 7


#define DIMENSIONS 3
#define FORCE_X 0
#define FORCE_Y 1
#define FORCE_Z 2

/*simulation constants*/
#define DELTA_TIME 0.2
#define G 0.05

#define CENTER_DISTANCE 10


double *init_bodies_demo(const int count) {
    /*
    inits a galaxy-like particle system
    */
    double *bodies = calloc(count * COMPONENTS, sizeof(double));

    bodies[POS_X] = 0;
    bodies[POS_Y] = 0;
    bodies[POS_Z] = 0;
    bodies[VEL_X] = 0;
    bodies[VEL_Y] = 0;
    bodies[VEL_Z] = 0;
    bodies[MASS] = 10000;

    for (int i = 1; i < count; i++) {
        bodies[i * COMPONENTS + MASS] = 1.0;
        bodies[i * COMPONENTS + POS_X] = cos(i) * (CENTER_DISTANCE + rand() % 3 - 3);
        bodies[i * COMPONENTS + POS_Y] = sin(i) * (CENTER_DISTANCE + rand() % 3 - 3);
        bodies[i * COMPONENTS + POS_Z] = rand() % 8 - 4;
        bodies[i * COMPONENTS + VEL_X] = -sin(i) * 4.5;
        bodies[i * COMPONENTS + VEL_Y] = cos(i) * 4.5;
    }
    
    return bodies;
}

double *init_bodies_demo_1(const int count) {
    /*
    inits a galaxy-like particle system
    */
    double *bodies = calloc(count * COMPONENTS, sizeof(double));

    for (int i = 0; i < count; i++) {
        bodies[i * COMPONENTS + MASS] = 1.0 + rand() % 50;
        bodies[i * COMPONENTS + POS_X] = cos(rand() % 360) * (CENTER_DISTANCE + rand() % 100) ;
        bodies[i * COMPONENTS + POS_Y] = sin(rand() % 360) * (CENTER_DISTANCE + rand() % 100);
        bodies[i * COMPONENTS + POS_Z] = -sin(rand() % 360) * (CENTER_DISTANCE) + rand() % 100;
        bodies[i * COMPONENTS + VEL_X] = cos(rand() % 360) * 1;
        bodies[i * COMPONENTS + VEL_Y] = sin(rand() % 360) * 1;
    }
    
    return bodies;
}


double *init_force_buffer(const int count) {
    double *bodies = (double*) calloc(count, sizeof(double) * DIMENSIONS);
    return bodies;
}


void write_frame_on_disk(const int count, const double *bodies, const int time) {
    /* 
    this function is SHIT.
    it does work, but its just too much overhead...
    gotta find a smarter way to handle 
    output file creation :')
    */
    FILE *fptr;
    char file_name[512];
    sprintf(file_name, "./outputs/sim2/simulation_frame_%d.csv", time);
    fptr = fopen(file_name, "w+");

    fprintf(fptr, "x,y,z\n");
    for (int i = 0; i < count; i++) {
        fprintf(fptr, "%f,%f,%f\n", bodies[i * COMPONENTS + POS_X], bodies[i * COMPONENTS + POS_Y], bodies[i * COMPONENTS + POS_Z]);
    }
    fclose(fptr); 
}


void update_force(const int count, double *force_buffer, double *bodies) {

    double dist_x;
    double dist_y;
    double dist_z;
    double dist;
    double attraction;
    
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            if (i == j) continue;
            dist_x = bodies[j * COMPONENTS + POS_X] - bodies[i * COMPONENTS + POS_X];
            dist_y = bodies[j * COMPONENTS + POS_Y] - bodies[i * COMPONENTS + POS_Y];
            dist_z = bodies[j * COMPONENTS + POS_Z] - bodies[i * COMPONENTS + POS_Z];
            dist = fmax(sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z), 5);
            attraction = G * (bodies[i * COMPONENTS + MASS] * bodies[j * COMPONENTS + MASS]) / (dist * dist);
            force_buffer[i * DIMENSIONS + FORCE_X] += attraction * (dist_x / dist);
            force_buffer[i * DIMENSIONS + FORCE_Y] += attraction * (dist_y / dist);
            force_buffer[i * DIMENSIONS + FORCE_Z] += attraction * (dist_z / dist);
        }
    }
}


void reset_force(const int count, double *force_buffer) {
    memset(force_buffer, 0, sizeof(double) * DIMENSIONS * count);
}


void update_pos(const int count, double *bodies, const int time) {
    for (int i = 0; i < count; i++) {
        bodies[i * COMPONENTS + TIME] = time;
        bodies[i * COMPONENTS + POS_X] += bodies[i * COMPONENTS + VEL_X] * DELTA_TIME;
        bodies[i * COMPONENTS + POS_Y] += bodies[i * COMPONENTS + VEL_Y] * DELTA_TIME;
        bodies[i * COMPONENTS + POS_Z] += bodies[i * COMPONENTS + VEL_Z] * DELTA_TIME;
    }
}


void update_vel(const int count, double *force_buffer, double *bodies, const double dt) {
    for (int i = 0; i < count; i++) {
        bodies[i * COMPONENTS + VEL_X] += (force_buffer[i * DIMENSIONS + FORCE_X] / bodies[i * COMPONENTS + MASS]) * dt;
        bodies[i * COMPONENTS + VEL_Y] += (force_buffer[i * DIMENSIONS + FORCE_Y] / bodies[i * COMPONENTS + MASS]) * dt;
        bodies[i * COMPONENTS + VEL_Z] += (force_buffer[i * DIMENSIONS + FORCE_Z] / bodies[i * COMPONENTS + MASS]) * dt;
    }
}


void simulation(double *bodies, size_t body_count, size_t iterations) {

    double *force_buffer = init_force_buffer(body_count);

    /*leapfrog first step*/
    update_force(body_count, force_buffer, bodies);
    update_vel(body_count, force_buffer, bodies, DELTA_TIME / 2);
    reset_force(body_count, force_buffer);

    for (size_t t = 0; t < iterations; t++) {
        update_pos(body_count, bodies, t);
        update_force(body_count, force_buffer, bodies);
        update_vel(body_count, force_buffer, bodies, DELTA_TIME);
        reset_force(body_count, force_buffer);
        //write_frame_on_disk(body_count, bodies, t);
    }
    free(force_buffer);
}


int main(int argc, char *argv[]) {
    
    /*error handling*/
    if (argc < 3) {
        printf("correct usage: %s, [body count], [iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t body_count = atoi(argv[1]);
    double *bodies = init_bodies_demo_1(body_count);
    if (bodies == NULL) {
        printf("particle initialization failed\n");
        return EXIT_FAILURE;
    }

    size_t iterations = atoi(argv[2]);

    simulation(bodies, body_count, iterations);
    free(bodies);
    
    return EXIT_SUCCESS;
}