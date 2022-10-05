/* 
 * File:   Benchmark.h
 * Author: gilberto
 *
 * Created on June 1, 2015, 8:39 PM
 */

#ifndef BENCHMARK_H
#define	BENCHMARK_H

#include "proj_types.h"
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include "Timer.h"

class Benchmark {
public:
    
    static const ulong GiB = 1024*1024*1024;
    static const ulong MiB = 1024*1024;
    static const ulong KiB = 1024;
    
    enum BlockMagType{
        MagKiB, MagMiB, MagGiB
    };
    
    Benchmark();
    Benchmark(std::string mountPoint, ulong repeats,
            ulong size, BlockMagType type, ulong block_size, BlockMagType block_type);
    Benchmark(const Benchmark& orig);
    virtual ~Benchmark();
    void setEnv(std::string mountPoint, ulong repeats,
            ulong size, BlockMagType type, ulong block_size, BlockMagType block_type);
    void run();
    
    void getResults();
    void getPartialResults();
    
private:
    std::string mountPoint;
    ulong repeats;

    // File content
    char *fileContent;

    // File size
    ulong magSize;
    Benchmark::BlockMagType blockType;
    ulong sizeRepeats;
    ulong sizeRWInMiB;
    ulong sizeRWInKiB;

    ulong gibs;
    ulong mibs;
    ulong kibs;

    // Block size
    ulong block_magSize;
    Benchmark::BlockMagType block_blockType;
    ulong block_sizeRepeats;
    ulong block_sizeRWInMiB;
    ulong block_sizeRWInKiB;

    ulong block_gibs;
    ulong block_mibs;
    ulong block_kibs;

    // Timer
    double totalTime;
    Timer timer;

    // Results
    // Write sequential
    double throughputWriteSequential;
    double execTimeWriteSequential;
    double averageWriteSequential;
    double defaulDevWriteSequential;
    
    // WriteRandom;
    double throughputWriteRandom;
    double execTimeWriteRandom;
    double averageWriteRandom;
    double defaulDevWriteRandom;

    // ReadSequential;
    double throughputReadSequential;
    double execTimeReadSequential;
    double averageReadSequential;
    double defaulDevReadSequential;

    // ReadRandom;
    double throughputReadRandom;
    double execTimeReadRandom;
    double averageReadRandom;
    double defaulDevReadRandom;
    
    
    // ResultPartial;
    // Write sequentiaPartial;
    double throughputWriteSequentialPartial;
    double execTimeWriteSequentialPartial;
    double averageWriteSequentialPartial;
    double defaulDevWriteSequentialPartial;
    // WriteRandomPartial;
    double throughputWriteRandomPartial;
    double execTimeWriteRandomPartial;
    double averageWriteRandomPartial;
    double defaulDevWriteRandomPartial;
    // ReadSequentialPartial;
    double throughputReadSequentialPartial;
    double execTimeReadSequentialPartial;
    double averageReadSequentialPartial;
    double defaulDevReadSequentialPartial;
    // ReadRandomPartial;
    double throughputReadRandomPartial;
    double execTimeReadRandomPartial;
    double averageReadRandomPartial;
    double defaulDevReadRandomPartial;

    std::string testFilePath;

    void reset();
    void setTestFilePath();
    void setMagTestSize();

    // Drop cache
    void dropCache();

    // Read and write
    void writeSequential();
    void readSequential();
    void writeRandom();
    void readRandom();

    void writeSequential_c();
    void writeSequential_c_warmup();
    void readSequential_c();
    void readSequential_c_warmup();

    bool envIsAlreadySet;

    // random access callback to seek to 0 if file pointer go further than the file total size.
    // static inline void lastIndexEventForFileWriter(ios::event ev, ios_base& stream, int index);
};

#endif	/* BENCHMARK_H */

