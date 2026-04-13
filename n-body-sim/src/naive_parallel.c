#include "../headers/ocl_boiler.h"
#include "../headers/n-body-init.h"

#define DELTA_TIME 0.02f
#define CENTER_DISTANCE 10
#define SEED 42


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


cl_event update_pos_run_k(cl_command_queue que, cl_kernel k, cl_mem bodies, cl_mem forces, unsigned int body_count, cl_float delta_time) {
    cl_event event;
    const size_t gws[1]= { round_mul_up(body_count, 32) };
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(k, arg_index, sizeof(bodies), &bodies);
    ocl_check(err,"clSetKernelArg update_pos 0");
    arg_index++;
    
    err = clSetKernelArg(k, arg_index, sizeof(forces), &forces);
    ocl_check(err,"clSetKernelArg update_pos 1");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(body_count), &body_count);
    ocl_check(err,"clSetKernelArg update_pos 2");
    arg_index++;

    err = clSetKernelArg(k, arg_index, sizeof(delta_time), &delta_time);
    ocl_check(err,"clSetKernelArg update_pos 3");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


cl_event update_vel_run_k(cl_command_queue que, cl_kernel k, cl_mem bodies, cl_mem forces, unsigned int body_count, cl_float delta_time) {
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

    err = clSetKernelArg(k, arg_index, sizeof(delta_time), &delta_time);
    ocl_check(err,"clSetKernelArg 3");
    arg_index++;

    cl_int error = clEnqueueNDRangeKernel(que, k, 1, NULL, gws, NULL, 0, NULL, &event);
    ocl_check(error, "clEnqueueNDRangeKernel");

    return event;
}


void write_frame_on_disk(const int count, const cl_float8 *bodies, const int time) {
    /* 
    this function is SHIT.
    it does work, but its just too much overhead...
    gotta find a smarter way to handle 
    output file creation :')
    */
    FILE *fptr;
    char file_name[512];
    sprintf(file_name, "./outputs/sim4_test/simulation_frame_%d.csv", time);
    fptr = fopen(file_name, "w+");

    fprintf(fptr, "x,y,z\n");
    for (int i = 0; i < count; i++) {
        fprintf(fptr, "%f,%f,%f\n", bodies[i].s1, bodies[i].s2, bodies[i].s3);
    }
    fclose(fptr); 
}


int main(int argc, char *argv[]) {
    
    if (argc < 3) {
        printf("correct usage: %s, [body count], [iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int body_count = atoi(argv[1]);
    if (body_count <= 0) {
        printf("body count must be at least 1\n");
        return EXIT_FAILURE;
    }

    unsigned int iterations = atoi(argv[2]);
    if (iterations <= 0) {
        printf("iterations must be at least 1\n");
        return EXIT_FAILURE;
    }

    /*openCL shenanigans*/
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

    cl_kernel update_vel_k = clCreateKernel(prog, "update_vel", &err);
    ocl_check(err, "clCreateKernel failed on update_vel");

    size_t body_buffer_size = sizeof(cl_float8) * body_count;
    cl_float8 *bodies = malloc(body_buffer_size);
    if (!bodies) {
        return EXIT_FAILURE;
    }

    bodies = init_bodies_two_colliding_galaxies(bodies, body_count, SEED);
    cl_mem body_vec = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, body_buffer_size, bodies, &err);
    ocl_check(err, "clCreateBuffer failed on body_buffer");

    free(bodies);
    size_t force_buffer_size = sizeof(cl_float3) * body_count;
    cl_mem forces = clCreateBuffer(ctx, CL_MEM_READ_WRITE, force_buffer_size, NULL, &err);
    ocl_check(err, "clCreateBuffer failed on body_buffer");

    /*sets force buffer to 0*/
    cl_event fill_buffer;
    size_t pattern[] = {0};
    err = clEnqueueFillBuffer(que, forces, pattern, 1, 0, force_buffer_size, 0, NULL, &fill_buffer);
    ocl_check(err, "clEnqueueFillBuffer failed on fill_buffer");

    clWaitForEvents(1, &fill_buffer);

    cl_event update_force_event[iterations + 1], update_pos_event[iterations], update_vel_event[iterations + 1];

    update_force_event[0] = update_force_run_k(que, update_force_k, body_vec, forces, body_count);
    clWaitForEvents(1, update_force_event);

    update_vel_event[0] = update_vel_run_k(que, update_vel_k, body_vec, forces, body_count, (cl_float) DELTA_TIME / 2);
    clWaitForEvents(1, update_vel_event);

    cl_event enqueue_map_buffer_event;

    for (int i = 0; i < iterations; i++) {
        update_pos_event[i] = update_pos_run_k(que, update_pos_k, body_vec, forces, body_count, (cl_float) DELTA_TIME);
        clWaitForEvents(1, update_pos_event + i);

        update_force_event[i + 1] = update_force_run_k(que, update_force_k, body_vec, forces, body_count);
        clWaitForEvents(1, update_force_event + i + 1);

        update_vel_event[i + 1] = update_vel_run_k(que, update_vel_k, body_vec, forces, body_count, (cl_float) DELTA_TIME);
        clWaitForEvents(1, update_vel_event + i + 1);

        
        bodies = clEnqueueMapBuffer(que, body_vec, CL_TRUE, CL_MAP_READ, 0, body_buffer_size, 0, NULL, &enqueue_map_buffer_event, &err);
        ocl_check(err, "enqueueMapBufferEvent failed");

        write_frame_on_disk(body_count, bodies, i);

        cl_event enqueue_unmap_event;
        err = clEnqueueUnmapMemObject(que, body_vec, bodies, 0, NULL, &enqueue_unmap_event);
        ocl_check(err, "enqueueUnmapObject failed");
    }

    double time_force_ms, time_pos_ms, time_vel_ms, time_enqueue_map_ms;
    
    time_pos_ms = total_runtime_ms(update_pos_event[0], update_pos_event[iterations - 1]);
    time_vel_ms = total_runtime_ms(update_vel_event[0], update_vel_event[iterations]);
    time_force_ms = total_runtime_ms(update_force_event[0], update_force_event[iterations]);
    time_enqueue_map_ms = runtime_ms(enqueue_map_buffer_event);

    printf("TIMES:\n\nupdate_pos: %gms,\nupdate_vel: %gms,\nupdate_force: %gms,\nenqueue_map_buffer: %gms\n",
    time_pos_ms, time_vel_ms, time_force_ms, time_enqueue_map_ms);
    
    /*testing purpose
    for (int i = 0; i < 10; i++) {
        printf("%f %f %f\n", bodies[i].s1, bodies[i].s2, bodies[i].s3);
    }
    */
}