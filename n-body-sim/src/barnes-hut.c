#include "../headers/ocl_boiler.h"
#include "../headers/sim-utils.h"
#include "../headers/n-body-init.h"
#include <sys/stat.h>

#define DELTA_TIME 0.02f
#define CENTER_DISTANCE 10
#define MAX_PARTICLES_FOR_LEAF 8
#define MAX_TREE_DEPTH 32
#define NODE_PROPORTION 4
#define SEED 42





/*
data la natura molto complessa della struttura dati che vorremmo poter associare a ciascuno dei
nodi, preferiamo usare tanti vettori di tipi differenti.


nodo:
    centro di massa
    massa

    centro del cubo

    figlio0 ... figlio7

    sono una foglia?
    lista delle particelle nella foglia
    numero di particelle

*/


int main(int argc, char *argv[]) {
    
    if (argc < 4) {
        printf("correct usage: %s, [body count], [iterations], [simulation-name]\n", argv[0]);
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

    char *sim_name = argv[3];

    /*openCL shenanigans*/
    cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("./n-body-sim/kernels/naive_nbody.ocl", ctx, d);
    cl_int err;

    
    cl_float3 

    /*
    array di particelle di dimensione n
    buffer di appoggio di dimensione round_mul_up(n/2, 2)
    
    while (i < n) {
        allocate res_buffer

        find_max(pos, res_buffer, n)
        find_min(pos, res_buffer, n)
        i * 2
        
    }
    *max and min will then be in the global memory* 
    */


    char path_name[1024] = "./outputs/";
    strcat(path_name, sim_name);
    mkdir(path_name, S_IRWXU);
}