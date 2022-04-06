#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <sstream>
#include "consts.h"
#include "kernels.h"
#define m2d(x, i, j) (x)->values[i * (x)->ncol + j]
#define m1d(x, i) (x)->values[i]
#define _ReLU(x) (x > 0 ?  x : 0)


struct matrix {
    int nrow;
    int ncol;
    dtype *values;
};

int matmul(struct matrix *X, struct matrix *Y, struct matrix *Z) 
{
    int i, j, k;
    for(i = 0; i < X->nrow; i++)
        for(j = 0; j < Y->ncol; j++)
            for(k = 0; k < X->ncol; k++) {
                //printf(" [%d,%d] += [%d,%d] x [%d,%d]   (=%.3f x %.3f)\n", i, j, i, k, k, j, m2d(X, i, k), m2d(Y, k, j) );
                m2d(Z, i, j) = m2d(Z, i, j) + (m2d(X, i, k) * m2d(Y, k, j));
            }
    return 0;
}

int matadd(struct matrix *X, struct matrix *Y, struct matrix *Z)
{
    int i;
    //printf(" adding %d x %d\n", X->nrow, X->ncol);
    for (i = 0; i < X->nrow * X->ncol; i++) {
        Z->values[i] = X->values[i] + Y->values[i];
    }
}

void print_matrix(struct matrix *X)
{
    int i, j;

    for(i=0; i<X->nrow; i++)
    {
        printf("\n\t");
        for(j=0; j<X->ncol;j++)
        {
            printf("%f\t", m2d(X, i, j));
        }
    }
    printf("\n");
}


void ReLU(struct matrix *X)
{
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        X->values[i] = _ReLU(X->values[i]);
    }
}

float forward_pass(struct matrix *input){
    float output;
    dtype o1[10] = {0};
    dtype o2[10] = {0};

    struct matrix W1 = {NR_FEAT, 10, w1};
    struct matrix out1 = {1, 10, o1};
    struct matrix B1 = {1, 10, b1};
    struct matrix W2 = {10, 1, w2};
    struct matrix out2 = {1, 1, o2};
    struct matrix B2 = {1, 1, b2};

    matmul(input, &W1, &out1);
    matadd(&out1, &B1, &out1);
    ReLU(&out1);
    matmul(&out1, &W2, &out2);
    matadd(&out2, &B2, &out2);
    output = m1d(&out2, 0);
    /* printf("output: %f\n", output); */
    /* return output > 0.5 ? 1 : 0; */
    return output;
}


int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Need argument: <inputs filename csv>\n");
        exit(1);
    }

    dtype mval[NR_FEAT];
    //struct matrix input = {1, NR_FEAT, mval};
    
    char line[300];
    int correct = 0, total = 0;
    int py_correct = 0, discrep = 0;

    FILE* f = fopen(argv[1], "r");
    fgets(line, 300, f); // Header
    int n = atoi(line);
    printf("Reading %d inputs\n", n);

    matrix inputs[n];
    for (int j = 0 ; j < n ; j++) {
        inputs[j].values = new dtype[NR_FEAT];
        inputs[j].nrow = 1;
        inputs[j].ncol = NR_FEAT;
    }

    for (int j = 0 ; j < n ; j++) {
        fgets(line, 300, f);
        int i;
        char *token, *string, *tofree;

        tofree = string = strdup(line);
        for (i = 0; i < NR_FEAT; i++) {
            token = strsep(&string, ",");
            float num = strtof(token, NULL);
            inputs[j].values[i] = num;
            /* printf("%f ", num); */
        }
        int label = atoi(strsep(&string, ","));
        int py_pred = atoi(strsep(&string, ","));
        /* float py_output = strtof(strsep(&string, ","), NULL); */
        free(tofree);
    }

    printf("Input read, inferencing..\n");
    gpu_setup(1);
    gpu_setup_inputs(inputs[0].values, 1);
    float rgpu = gpu_inference();
    float rcpu = forward_pass(inputs);

    printf("Inferences on CPU and GPU: %.5f, %.5f\n", rcpu, rgpu);

    //warmup
    for (int j = 0 ; j < 1 ; j++) {
        float output = forward_pass(inputs+j);
    }

    std::stringstream csv;
    csv << ", inf_total, inf_transfer_total, inf_avg, inf_transfer_avg\n";

    /*
     *  CPU timing
     */
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int j = 0 ; j < n ; j++) {
        float output = forward_pass(inputs+j);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "CPU time for " << n << " inferences: " << total_time << "ns. Average per inference:" << total_time/n << "ns." << std::endl;

    csv << "cpu," << total_time << "," << total_time << "," << total_time/n << "," << total_time/n << std::endl;

    /*
     *  GPU naive timing
     */
    uint32_t gpu_total(0);
    uint32_t gpu_all_total(0);
    for (int j = 0 ; j < n ; j++) {
        std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
        gpu_setup_inputs(inputs[j].values, 1);
        std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
        float output = gpu_inference();
        std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
        gpu_get_result(1);
        std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

        gpu_all_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu_all - begin_gpu_all).count();
        gpu_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - begin_gpu).count();
    }
    std::cout << "GPU time for " << n << " sequential inferences: " << gpu_total << "ns. Average per inference:" << gpu_total/n << "ns." << std::endl;
    gpu_clean();
    
    csv << "GPU naive" << "," << gpu_total << "," <<  gpu_all_total << "," << gpu_total/n << "," << gpu_all_total/n << "," << std::endl;

    /*
     *  GPU batched timing
     */

    int batch_sizes[] = {64, 128, 256, 512};


    for (int &N_INPUTS_BATCH : batch_sizes) {
        gpu_setup(N_INPUTS_BATCH);
        uint32_t gpubatch_total(0);
        uint32_t gpubatch_all_total(0);

        //flatten inputs
        float* linear_inputs = new float[NR_FEAT*n];
        for (int j = 0 ; j < n ; j++) {
            for (int i = 0; i < NR_FEAT; i++) {
                linear_inputs[j*NR_FEAT + i] = inputs[j].values[i];
            }
        }

        //warmup
        for (int j = 0 ; j < n/N_INPUTS_BATCH ; j++) {
            gpu_setup_inputs(linear_inputs+j*N_INPUTS_BATCH, N_INPUTS_BATCH);
            gpu_inference_many(N_INPUTS_BATCH);
        }

        //for each batch, measure
        for (int j = 0 ; j < n/N_INPUTS_BATCH ; j++) {
            std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
            gpu_setup_inputs(linear_inputs+j*N_INPUTS_BATCH, N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
            gpu_inference_many(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
            gpu_get_result(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

            gpubatch_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - begin_gpu).count();
            gpubatch_all_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu_all - begin_gpu_all).count();

        }
        std::cout << "Batched GPU time for " << n << " inferences (batch size " << N_INPUTS_BATCH << "): " << gpubatch_total << "ns. Average per inference:" << gpubatch_total/n << "ns." << std::endl;
        std::cout << "Including data transfers: " << gpubatch_all_total << "ns. Average per inference:" << gpubatch_all_total/n << "ns." << std::endl;
        
        csv << "GPU batch" << N_INPUTS_BATCH << "," << gpubatch_total << "," <<  gpubatch_all_total << "," << gpubatch_total/n << "," << gpubatch_all_total/n << "," << std::endl;
        gpu_clean();
    }

    std::cout << "CSV:\n" << csv.str();


    return 0;
}
