#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <sstream>
#include "kernels.h"



int main(int argc, char** argv)
{
    int n = 1024;

    std::stringstream csv;
    csv << ", inf_total, inf_avg, inf_transfer_total, inf_transfer_avg\n";

    /*
     *  GPU naive timing
     */

    setup_naive();

    uint32_t gpu_total(0);
    uint32_t gpu_all_total(0);
    for (int j = 0 ; j < n ; j++) {
        std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
        copy_inputs_naive();
        std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
        infer_naive();
        std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
        get_result_naive();
        std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

        gpu_all_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu_all - begin_gpu_all).count();
        gpu_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - begin_gpu).count();
    }
    std::cout << "GPU time for " << n << " sequential inferences: " << gpu_total << "ns. Average per inference:" << gpu_total/n << "ns." << std::endl;
    clean_naive();
    csv << "GPU naive" << "," << gpu_total << "," << gpu_total/n << "," << gpu_all_total << "," << gpu_all_total/n << "," << std::endl;



    /*
     *  GPU batched timing
     */

    int batch_sizes[] = {16, 64, 128, 256, 512};
    for (int &N_INPUTS_BATCH : batch_sizes) {
        setup_batch(N_INPUTS_BATCH);
        uint32_t gpubatch_total(0);
        uint32_t gpubatch_all_total(0);

        // //warmup
        // for (int j = 0 ; j < n/N_INPUTS_BATCH ; j++) {
        //     gpu_setup_inputs(linear_inputs+j*N_INPUTS_BATCH, N_INPUTS_BATCH);
        //     gpu_inference_many(N_INPUTS_BATCH);
        // }

        //for each batch, measure
        for (int j = 0 ; j < n/N_INPUTS_BATCH ; j++) {
            std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
            //copy_inputs_batch(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
            //infer_batch(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
            //get_result_batch(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

            gpubatch_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - begin_gpu).count();
            gpubatch_all_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu_all - begin_gpu_all).count();

        }
        std::cout << "Batched GPU time for " << n << " inferences (batch size " << N_INPUTS_BATCH << "): " << gpubatch_total << "ns. Average per inference:" << gpubatch_total/n << "ns." << std::endl;
        std::cout << "Including data transfers: " << gpubatch_all_total << "ns. Average per inference:" << gpubatch_all_total/n << "ns." << std::endl;
        
        csv << "GPU batch" << N_INPUTS_BATCH << "," << gpubatch_total << "," << gpubatch_total/n << "," << gpubatch_all_total << "," << gpubatch_all_total/n << "," << std::endl;
        clean_batch();
    }


    std::cout << "CSV:\n" << csv.str();
    return 0;
}