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
    // int n = 1024;
    // int batch_size = 256;
    // int do_cpu = 1;

    // uint64_t batches_per_sample = 5;
    // int run_for = 41*1000; //s to ms

    // uint64_t last_inf = 0;
    // std::chrono::high_resolution_clock::time_point prev_ts = std::chrono::high_resolution_clock::now();
    // std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // if (!do_cpu)
    //     setup_batch(batch_size);

    // while (1) {
    //     std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    //     double elaps = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    //     if (elaps > run_for) break;

    //     for(int j = 0 ; j < batches_per_sample ; j++) {
    //         if (do_cpu) {
    //             long input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    //             for (int k = 0 ; k < batch_size ; k++) {
    //                 long input_copy[31];
    //                 memcpy (input_copy, input, sizeof(input));
    //                 bool res = prediction_cpu(&input_copy[0]);
    //             }
    //         }
    //         else {
    //             if (j == 0) {
    //                 copy_inputs_batch(batch_size);
    //             }
    //             infer_batch(batch_size);
    //             get_result_batch(batch_size);
    //         } 
    //         std::this_thread::sleep_for(std::chrono::microseconds(10));
    //     }
        
    //     uint64_t infd = last_inf + batches_per_sample * batch_size;
    //     now = std::chrono::high_resolution_clock::now();
    //     printf("%f,%f\n", std::chrono::duration_cast<std::chrono::duration<double>>(now - start).count(), 
    //         (double) (infd - last_inf) / (std::chrono::duration_cast<std::chrono::duration<double>>(now - prev_ts).count()));
        
    //     last_inf = infd;
    //     prev_ts = now;
    // }
   
    // if (!do_cpu) clean_batch();

        // setup_batch(N_INPUTS_BATCH);
        // std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
        // copy_inputs_batch(N_INPUTS_BATCH);
        // std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
        // infer_batch(N_INPUTS_BATCH);
        // std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
        // get_result_batch(N_INPUTS_BATCH);
        // std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();
        // clean_batch();
   






//previous code, dont erase

    int n = 1024;

    std::stringstream csv;
    csv << ", inference, inference+transfer\n";
    long input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    long input_1[31] = {9,9,9,0,9,1,1,9,9,9,9,0,9,9,1,9,9,1,9,9,0,9,1,9,1,9,0,9,9,0,9};
    long input_2[31] = {9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9};
    long input_3[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int j = 0 ; j < 1 ; j++) {
            long input_copy[31];
            memcpy (input_copy, input, sizeof(input));
            bool res = prediction_cpu(&input_copy[0]);
        }


    /*
     *  CPU timing
     */
    int cpu_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int RUNS = 11;

    for (int &N_INPUTS_BATCH : cpu_sizes) {
        uint32_t cpubatch_total(0);
        //warmups
        for(int i = 0 ; i < 2; i++) {
            for (int j = 0 ; j < N_INPUTS_BATCH ; j++) {
                long input_copy[31];
                memcpy (input_copy, input, sizeof(input));
                bool res = prediction_cpu(&input_copy[0]);
            }
        }

        for (int i = 0 ; i < RUNS ; i++) {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (int j = 0 ; j < N_INPUTS_BATCH ; j++) {
	            long input_copy[31];
                memcpy (input_copy, input, sizeof(input));
                bool res = prediction_cpu(&input_copy[0]);
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            cpubatch_total += total_time;
	        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "CPU time for " << N_INPUTS_BATCH << " inferences: " << cpubatch_total/RUNS << "us. Average per inference:" << cpubatch_total/n << "us." << std::endl;
        csv << "cpu" <<N_INPUTS_BATCH<<", " << cpubatch_total/RUNS << "," << cpubatch_total/RUNS << std::endl;
    }

    // Test output of the inputs
    long input_copy[31];
    std::cout << "Outputs for batch size 4 \n" ;
    memcpy (input_copy, input, sizeof(input));
    bool res = prediction_cpu(&input_copy[0]);
    std::cout << res << std::endl;

    memcpy (input_copy, input_1, sizeof(input));
    res = prediction_cpu(&input_copy[0]);
    std::cout << res << std::endl;

    memcpy (input_copy, input_2, sizeof(input));
    res = prediction_cpu(&input_copy[0]);
    std::cout << res << std::endl;

    memcpy (input_copy, input_3, sizeof(input));
    res = prediction_cpu(&input_copy[0]);
    std::cout << res << std::endl;

    // /*
    //  *  GPU naive timing
    //  */

    // setup_batch(1);
    // cudaDeviceSynchronize();

    // uint32_t gpu_total(0);
    // uint32_t gpu_all_total(0);
    // for (int j = 0 ; j < n ; j++) {
    //     std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
    //     copy_inputs_batch(1);
    //     std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
    //     infer_batch(1);
    //     std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
    //     get_result_batch(1);
    //     std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

    //     gpu_all_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_all - begin_gpu_all).count();
    //     gpu_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu).count();
    // }

    // std::cout << "GPU time for " << n << " sequential inferences: " << gpu_total << "us. Average per inference:" << gpu_total/n << "us." << std::endl;
    // clean_batch();
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));    
    // csv << "GPU naive" << "," << gpu_total << "," <<  gpu_all_total << std::endl;


    // /*
    //  *  GPU batched timing
    //  */

    // int batch_sizes[] = {8, 16, 32, 64, 128, 256, 512};
    // for (int &N_INPUTS_BATCH : batch_sizes) {
    //     setup_batch(N_INPUTS_BATCH);
    //     uint32_t gpubatch_total(0);
    //     uint32_t gpubatch_all_total(0);

    //     int WARMUP_RUNS = 2;
    //     for (int j = 0 ; j < WARMUP_RUNS ; j++) {
    //         copy_inputs_batch(N_INPUTS_BATCH);
    //         infer_batch(N_INPUTS_BATCH);
    //     }
    //     cudaDeviceSynchronize();
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));

    //     std::chrono::steady_clock::time_point begin_out = std::chrono::steady_clock::now();
    //     int RUNS = 10;
    //     for (int j = 0 ; j < RUNS ; j++) {
    //         std::chrono::steady_clock::time_point begin_gpu_all = std::chrono::steady_clock::now();
    //         copy_inputs_batch(N_INPUTS_BATCH);
    //         std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
    //         infer_batch(N_INPUTS_BATCH);
    //         std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
    //         get_result_batch(N_INPUTS_BATCH);
    //         std::chrono::steady_clock::time_point end_gpu_all = std::chrono::steady_clock::now();

    //         gpubatch_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - begin_gpu).count();
    //         gpubatch_all_total += std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_all - begin_gpu_all).count();

    //         std::this_thread::sleep_for(std::chrono::milliseconds(20)); 
    //     }
    //     std::chrono::steady_clock::time_point end_out = std::chrono::steady_clock::now();

    //     //std::cout << "Batched GPU time for " << n << " inferences (batch size " << N_INPUTS_BATCH << "): " << gpubatch_total << "us. Average per inference:" << gpubatch_total/n << "us." << std::endl;
    //     //std::cout << "Including data trausfers: " << gpubatch_all_total << "us. Average per inference:" << gpubatch_all_total/n << "us." << std::endl;
    //     std::cout << "Avg OUTSIDE time for " << N_INPUTS_BATCH << " inferences: " << std::chrono::duration_cast<std::chrono::microseconds>(end_out - begin_out).count()/RUNS << " us\n";
        
    //     csv << "GPU batch" << N_INPUTS_BATCH << "," << gpubatch_total/RUNS << "," <<  gpubatch_all_total/RUNS << std::endl; 
    //     clean_batch();
    // }


    // std::cout << "CSV:\n" << csv.str();
    // return 0;
}
