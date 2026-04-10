#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../headers/ocl_boiler.h"


#define CENTER_DISTANCE 10

void *init_bodies_demo(cl_float8 *bodies, unsigned int body_count) {
    
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
        bodies[i].s1 = cos(i) * (CENTER_DISTANCE + rand() % 3 - 3);
        bodies[i].s2 = sin(i) * (CENTER_DISTANCE + rand() % 3 - 3);
        bodies[i].s3 = rand() % 8 - 4;
        bodies[i].s4 = -sin(i) * 4.5;
        bodies[i].s5 = cos(i) * 4.5;
        bodies[i].s6 = 0;
        bodies[i].s7 = 1.0;
    }
    
    return bodies;
}


cl_event update_force_run_k(cl_command_queue que, cl_kernel k, cl_mem bodies, cl_mem forces, unsigned int body_count) {
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(k, arg_index, sizeof(bodies), &bodies);
    ocl_check(err,"clSetKernelArg 0");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(forces), &forces);
    ocl_check(err,"clSetKernelArg 1");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg 2");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


cl_event update_pos_run_k(cl_command_queue que, cl_kernel k, cl_mem bodies, cl_mem forces, unsigned int body_count) {
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(k, arg_index, sizeof(bodies), &bodies);
    ocl_check(err,"clSetKernelArg 0");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(forces), &forces);
    ocl_check(err,"clSetKernelArg 1");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg 2");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


int main(int argc, char *argv[]) {
    
    /*error handling*/
    if (argc < 3) {
        printf("correct usage: %s, [body count], [iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int body_count = atoi(argv[1]);
    if (body_count <= 0) {
        printf("body count must be at least 1\n");
        return EXIT_FAILURE;
    }

    cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("./n-body-sim/kernels/naive_nbody.ocl", ctx, d);

    cl_int err;
    cl_kernel update_force_k = clCreateKernel(prog, "update_force", &err);
    ocl_check(err, "clCreateKernel failed on update_force");

    cl_kernel update_pos_k = clCreateKernel(prog, "update_pos", &err);
    ocl_check(err, "clCreateKernel failed on update_pos");

    size_t body_buffer_size = sizeof(cl_float8) * body_count;
    void *bodies = malloc(body_buffer_size);
    if (!bodies) {
        return EXIT_FAILURE;
    }

    cl_mem body_vec = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, body_buffer_size, bodies, &err);
    ocl_check(err, "clCreateBuffer failed on body_buffer");

    size_t force_buffer_size = sizeof(cl_float3) * body_count;
    cl_mem forces = clCreateBuffer(ctx, CL_MEM_READ_WRITE, force_buffer_size, NULL, &err);
    ocl_check(err, "clCreateBuffer failed on body_buffer");

    /*sets force buffer to 0*/
    cl_event fill_buffer;
    size_t pattern[] = {0};
    err = clEnqueueFillBuffer(que, forces, pattern, 1, 0, force_buffer_size, 0, NULL, &fill_buffer);
    ocl_check(err, "clEnqueueFillBuffer failed on fill_buffer");

    clWaitForEvents(1, &fill_buffer);

    cl_event update_force_event;
    
    update_force_event = update_force(que, update_force_k, body_vec, forces, body_count);
}