#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <sstream>
#include "kernels.h"
#include <thread>
#include <cuda_runtime.h>


int main(int argc, char** argv)
{
    int n = 1024;

    std::stringstream csv;
    csv << ", inference, inference+transfer\n";
    long input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};

    for (int j = 0 ; j < 1 ; j++) {
            bool res = prediction_cpu(&input[0]);
        }


    /*
     *  CPU timing
     */
    int cpu_sizes[] = {8, 16, 32, 64, 128, 256, 512};

    for (int &N_INPUTS_BATCH : cpu_sizes) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int j = 0 ; j < N_INPUTS_BATCH ; j++) {
            bool res = prediction_cpu(&input[0]);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "CPU time for " << N_INPUTS_BATCH << " inferences: " << total_time << "us. Average per inference:" << total_time/n << "us." << std::endl;

        csv << "cpu" <<N_INPUTS_BATCH<<", " << total_time << "," << total_time << std::endl;
    }



    /*
     *  GPU naive timing
     */

    setup_naive();

    uint32_t gpu_total(0);
    uint32_t gpu_all_total(0);
    for (int j = 0 ; j < n ; j++) {
        std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
        copy_inputs_batch(1);
        std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
        infer_batch(1);
        std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
        get_result_batch(1);
        std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

        gpu_all_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_all - begin_gpu_all).count();
        gpu_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu).count();
    }

    std::cout << "GPU time for " << n << " sequential inferences: " << gpu_total << "us. Average per inference:" << gpu_total/n << "us." << std::endl;
    clean_batch();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));    
    csv << "GPU naive" << "," << gpu_total << "," <<  gpu_all_total << std::endl;


    /*
     *  GPU batched timing
     */

    int batch_sizes[] = {8, 16, 32, 64, 128, 256, 512};
    for (int &N_INPUTS_BATCH : batch_sizes) {
        setup_batch(N_INPUTS_BATCH);
        uint32_t gpubatch_total(0);
        uint32_t gpubatch_all_total(0);

        int WARMUP_RUNS = 2;
        for (int j = 0 ; j < WARMUP_RUNS ; j++) {
            copy_inputs_batch(N_INPUTS_BATCH);
            infer_batch(N_INPUTS_BATCH);
        }
        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::chrono::steady_clock::time_point begin_out = std::chrono::steady_clock::now();
        int RUNS = 10;
        for (int j = 0 ; j < RUNS ; j++) {
            std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
            copy_inputs_batch(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
            infer_batch(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
            get_result_batch(N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

            gpubatch_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu).count();
            gpubatch_all_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_all - begin_gpu_all).count();

            std::this_thread::sleep_for(std::chrono::milliseconds(20)); 
        }
        std::chrono::steady_clock::time_point end_out = std::chrono::steady_clock::now();

        //std::cout << "Batched GPU time for " << n << " inferences (batch size " << N_INPUTS_BATCH << "): " << gpubatch_total << "us. Average per inference:" << gpubatch_total/n << "us." << std::endl;
        //std::cout << "Including data trausfers: " << gpubatch_all_total << "us. Average per inference:" << gpubatch_all_total/n << "us." << std::endl;
        std::cout << "Avg OUTSIDE time for " << N_INPUTS_BATCH << " inferences: " << std::chrono::duration_cast<std::chrono::microseconds>(end_out - begin_out).count()/RUNS << " us\n";
        
        csv << "GPU batch" << N_INPUTS_BATCH << "," << gpubatch_total/RUNS << "," <<  gpubatch_all_total/RUNS << std::endl; 
        clean_batch();
    }


    std::cout << "CSV:\n" << csv.str();
    return 0;
}