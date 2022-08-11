/* 
 * File:   Benchmark.cpp
 * Author: gilberto
 * 
 * Created on June 1, 2015, 8:39 PM
 */

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Benchmark.h"
#include "Timer.h"

using namespace std;


/**
 * Default constructor
 */
Benchmark::Benchmark() {

    this->fileContent = NULL;
    this->mountPoint = "";
    this->repeats = 0;
    this->sizeRepeats = 0;
    this->block_sizeRepeats = 0;

    envIsAlreadySet = false;

    this->throughputWriteSequential = 0;
    this->execTimeWriteSequential = 0;
    this->averageWriteSequential = 0;
    this->defaulDevWriteSequential = 0;

    this->throughputWriteRandom = 0;
    this->execTimeWriteRandom = 0;
    this->averageWriteRandom = 0;
    this->defaulDevWriteRandom = 0;

    this->throughputReadSequential = 0;
    this->execTimeReadSequential = 0;
    this->averageReadSequential = 0;
    this->defaulDevReadSequential = 0;

    this->throughputReadRandom = 0;
    this->execTimeReadRandom = 0;
    this->averageReadRandom = 0;
    this->defaulDevReadRandom = 0;
}

/**
 * Constructor setting the environment
 * @param mountPoint
 * @param repeats
 * @param size
 */
Benchmark::Benchmark(std::string mountPoint, ulong repeats,
        ulong size, BlockMagType type, ulong block_size, BlockMagType block_type)
    : Benchmark()
{
    this->mountPoint = mountPoint;
    this->repeats = repeats;
    this->sizeRepeats = size;
    this->blockType = type;
    this->block_sizeRepeats = block_size;
    this->block_blockType = block_type;

    envIsAlreadySet = true;
}

/**
 * Copy contructor
 * @param orig
 */
Benchmark::Benchmark(const Benchmark& orig) : Benchmark() {
    this->reset();
    this->mountPoint = orig.mountPoint;
    this->repeats = orig.repeats;
    this->sizeRepeats = orig.sizeRepeats;
    this->blockType = orig.blockType;
    this->block_sizeRepeats = orig.block_sizeRepeats;
    this->block_blockType = orig.block_blockType;
}

/**
 * Destructor ( no dym allocation, no deallocation)
 */
Benchmark::~Benchmark() {
    if (this->fileContent)
        free(this->fileContent);
}

/**
 * Set the environment to be able to run the benchmark.
 * Use in case of use of default constructor.
 * 
 * @param mountPoint
 * @param repeats
 * @param size
 */
void Benchmark::setEnv(std::string mountPoint, ulong repeats,
        ulong size, BlockMagType type, ulong block_size, BlockMagType block_type)
{
    this->reset();

    this->mountPoint = mountPoint;
    this->repeats = repeats;
    this->sizeRepeats = size;
    this->blockType = type;
    this->block_sizeRepeats = block_size;
    this->block_blockType = block_type;

    envIsAlreadySet = true;
}

/**
 * Run the benchmark only if the environment
 * is already set.
 */
void Benchmark::run(){

    this->reset();
    this->setMagTestSize();
    this->setTestFilePath();

    if(this->envIsAlreadySet){
        sizeRWInMiB = (this->sizeRepeats * this->magSize) / Benchmark::MiB;
        sizeRWInKiB = (this->sizeRepeats * this->magSize) / Benchmark::KiB;

        block_sizeRWInMiB = (this->block_sizeRepeats * this->block_magSize) / Benchmark::MiB;
        block_sizeRWInKiB = (this->block_sizeRepeats * this->block_magSize) / Benchmark::KiB;

        // Prepare data
        if (!this->fileContent) {
            this->fileContent = new char[this->sizeRepeats * this->magSize];

            srand(time(NULL));
            for (char *fp = this->fileContent;
                    (char *)fp < (this->fileContent + this->sizeRepeats * this->magSize);
                    fp++) {
                *fp = (rand() % 26) + 'a';
            }
        }

        for(ulong repeatIdx = 0; repeatIdx < this->repeats; repeatIdx++) {
            this->dropCache();
            this->writeSequential_c();
            this->dropCache();
            this->readSequential_c();

            //this->writeSequential();
            //this->readSequential();
            //this->writeRandom();
            //this->readRandom();

            cout << endl << "Repeat: " << repeatIdx + 1 << "/" << this->repeats << endl;
            this->getPartialResults();
            cout << endl;
        }
        this->getResults();
        printf("Time for all testes: %.10f seconds\n", this->totalTime);
    }
}

/**
 * Print results produced by benchmark
 * @return result
 */
void Benchmark::getResults(){

    char title[] = "%20s\t%20s\t%20s\t%20s\t%20s\n";
    char format[] = "%20s\t%20.10f\t%20.10f\t%20.10f\t%20.10f\n";
    
    cout << "Final average results:" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------" << endl;
    printf(title, "Operation Type", "Throughput MiB/s", "Average in seconds", "Default Deviation", "Execution time");    
    printf(format, "Read Sequential", throughputReadSequential/repeats, averageReadSequential/repeats, defaulDevReadSequential/repeats, execTimeReadSequential/repeats);
    printf(format, "Read Random", throughputReadRandom/repeats, averageReadRandom/repeats, defaulDevReadRandom/repeats, execTimeReadRandom/repeats);
    printf(format, "Write Sequential", throughputWriteSequential/repeats, averageWriteSequential/repeats, defaulDevWriteSequential/repeats, execTimeWriteSequential/repeats);
    printf(format, "Write Random", throughputWriteRandom/repeats, averageWriteRandom/repeats, defaulDevWriteRandom/repeats, execTimeWriteRandom/repeats);
    cout << endl;
}

/**
 * Print partial results
 */
void Benchmark::getPartialResults(){
    char title[] = "%20s\t%20s\t%20s\t%20s\t%20s\n";
    char format[] = "%20s\t%20.10f\t%20.10f\t%20.10f\t%20.10f\n";
    char sep[] = "----------------------------------------------------------------------------------------------------------------------------\n";
    printf(title, "Operation Type", "Throughput MiB/s", "Average in seconds", "Default Deviation", "Execution time");
    printf("%s", sep);
    printf(format, "Read Sequential", throughputReadSequentialPartial, averageReadSequentialPartial, defaulDevReadSequentialPartial, execTimeReadSequentialPartial);
    printf(format, "Read Random", throughputReadRandomPartial, averageReadRandomPartial, defaulDevReadRandomPartial, execTimeReadRandomPartial);
    printf(format, "Write Sequential", throughputWriteSequentialPartial, averageWriteSequentialPartial, defaulDevWriteSequentialPartial, execTimeWriteSequentialPartial);
    printf(format, "Write Random", throughputWriteRandomPartial, averageWriteRandomPartial, defaulDevWriteRandomPartial, execTimeWriteRandomPartial);
    printf("%s", sep);    
}

/**
 * Set throughput and total time to zero.
 */
void Benchmark::reset(){    
    totalTime = 0;    
}

/**
 * Set the the file path to write the test.
 */
void Benchmark::setTestFilePath(){
    this->testFilePath = this->mountPoint + "/testfile";
}

/**
 * Get selected magnitude and set max size.
 */
void Benchmark::setMagTestSize(){
    // File size
    if(blockType == Benchmark::MagGiB){
        this->magSize = Benchmark::GiB;
        this->gibs = this->sizeRepeats;
        this->mibs = 1024;
        this->kibs = 1024;
    }
    else if(blockType == Benchmark::MagMiB){
        this->magSize = Benchmark::MiB;
        this->gibs = 1;
        this->mibs = this->sizeRepeats;
        this->kibs = 1024;
    }
    else if(blockType == Benchmark::MagKiB){
        this->magSize = Benchmark::KiB;
        this->gibs = 1;
        this->mibs = 1;
        this->kibs = this->sizeRepeats;
    }
    else {
        this->magSize = Benchmark::KiB;
        this->gibs = 1;
        this->mibs = 1;
        this->kibs = this->sizeRepeats;
    }

    // Block size
    if(block_blockType == Benchmark::MagGiB){
        this->block_magSize = Benchmark::GiB;
        this->block_gibs = this->block_sizeRepeats;
        this->block_mibs = 1024;
        this->block_kibs = 1024;
    }
    else if(block_blockType == Benchmark::MagMiB){
        this->block_magSize = Benchmark::MiB;
        this->block_gibs = 1;
        this->block_mibs = this->block_sizeRepeats;
        this->block_kibs = 1024;
    }
    else {
        this->block_magSize = Benchmark::KiB;
        this->block_gibs = 1;
        this->block_mibs = 1;
        this->block_kibs = this->block_sizeRepeats;
    }
}

void Benchmark::dropCache()
{
    FILE *fp = fopen("/proc/sys/vm/drop_caches", "w");
    fprintf(fp, "3");
    fclose(fp);
    //system("./drop_cache");
}

/**
 * Logic to write sequential and register with c write
 */
void Benchmark::writeSequential_c()
{
    // file handler
    int file;

    // open the file
    file = open( this->testFilePath.c_str(), O_CREAT | O_WRONLY | O_TRUNC,
            S_IRUSR | S_IWUSR | S_IWOTH | S_IROTH);

    // size to use
    ulong sizeToUse = sizeRWInMiB ? sizeRWInMiB : 1;
    ulong amoutToDivideBy = sizeToUse == 1 ? 1024 / sizeRWInKiB : 1;

    // Set timer set amount of blocks
    timer.clear();
    timer.setSetSize((ulong)ceil(0.1 * sizeRWInKiB / block_sizeRWInKiB));

    // Write this->fileContent to file in blocks
    ulong left_size = sizeRWInKiB;
    ulong write_size;
    char *p = this->fileContent;

    while (left_size > 0) {
        write_size = (left_size >= block_sizeRWInKiB) ?
            (block_sizeRWInKiB << 10) : (left_size << 10);

        timer.start();
        write(file, p, write_size);
        timer.stop();

        p += write_size;
        left_size -= (write_size >> 10);
    }

    // Get time of this action
    this->totalTime += timer.totalTime();
    cout << this->totalTime << endl;

    // Write sequentiaPartial;
    // Throughput correction in case of less than 1 MiB and more than 1 KiB
    throughputWriteSequentialPartial = sizeToUse / timer.totalTime() / amoutToDivideBy;
    execTimeWriteSequentialPartial   = timer.totalTime();
    averageWriteSequentialPartial    = timer.averageTime();
    defaulDevWriteSequentialPartial  = timer.defaultDeviation();

    // Capture data
    this->averageWriteSequential += averageWriteSequentialPartial;
    this->defaulDevWriteSequential += defaulDevWriteSequentialPartial;    
    this->throughputWriteSequential += throughputWriteSequentialPartial;
    this->execTimeWriteSequential += execTimeWriteSequentialPartial;


    // close file
    fsync(file);
    close(file);

    // clear cronometer
    timer.clear();
}

/**
 * Read from file byte by byte sequentially with c read
 */
void Benchmark::readSequential_c()
{
    // file handler
    int file;

    // open the file    
    file = open(this->testFilePath.c_str(), O_RDONLY, NULL);

    // size to use
    ulong sizeToUse = sizeRWInMiB ? sizeRWInMiB : 1;
    ulong amoutToDivideBy = sizeToUse == 1 ? 1024 / sizeRWInKiB : 1;

    // Set timer set amount of blocks
    timer.setSetSize((ulong)ceil(0.1 * sizeRWInKiB / block_sizeRWInKiB));

    // Write this->fileContent to file in blocks
    ulong left_size = sizeRWInKiB;
    ulong read_size;
    char *p = this->fileContent;

    while (left_size > 0) {
        read_size = (left_size >= block_sizeRWInKiB) ?
            (block_sizeRWInKiB << 10) : (left_size << 10);

        timer.start();
        read_size = read(file, p, read_size);
        timer.stop();

        if (read_size <= 0) {
            cout << "Error: Read from file " << this->testFilePath << endl;
            break;
        }
        p += read_size;
        left_size -= (read_size >> 10);
    }

    // Get time of this action
    this->totalTime += timer.totalTime();

    // ReadSequentialPartial;
    // Throughput correction in case of less than 1 MiB and more than 1 KiB
    throughputReadSequentialPartial = sizeToUse / timer.totalTime() / amoutToDivideBy;
    execTimeReadSequentialPartial   = timer.totalTime();
    averageReadSequentialPartial    = timer.averageTime();
    defaulDevReadSequentialPartial  = timer.defaultDeviation();

    // capture data       
    this->throughputReadSequential += throughputReadSequentialPartial;
    this->averageReadSequential += averageReadSequentialPartial;
    this->defaulDevReadSequential += defaulDevReadSequentialPartial;
    this->execTimeReadSequential += execTimeReadSequentialPartial;    


    // close file 
    close(file);

    // clear cronometer
    timer.clear();
}

/**
 * Logic to write sequential and register with ofstream
 */
void Benchmark::writeSequential()
{
    // file handler
    ofstream file;

    // open the file
    file.open( this->testFilePath.c_str(), ios::binary | ios::trunc);

    // size to use
    ulong sizeToUse = sizeRWInMiB ? sizeRWInMiB : 1;
    ulong amoutToDivideBy = sizeToUse == 1 ? 1024 / sizeRWInKiB : 1;

#ifndef LEGACY

    file.rdbuf()->pubsetbuf(0, 0);

    // Set timer set amount of blocks
    timer.setSetSize((ulong)ceil(0.1 * sizeRWInKiB / block_sizeRWInKiB));

    // Write this->fileContent to file in blocks
    ulong left_size = sizeRWInKiB;
    ulong write_size;
    char *p = this->fileContent;

    while (left_size > 0) {
        write_size = (left_size >= block_sizeRWInKiB) ?
            (block_sizeRWInKiB << 10) : (left_size << 10);

        timer.start();
        file.write(p, write_size);
        file.flush();
        timer.stop();

        p += write_size;
        left_size -= (write_size >> 10);
    }

#else

    // Set timer set amount KiBs
    timer.setSetSize(sizeToUse);

    // Write to file using bytes as index
    for(ulong gIdx = 0; gIdx < gibs; gIdx++){
        for(ulong mIdx = 0; mIdx < mibs; mIdx++){

            // Start timer to each MiB measure
            timer.start();
            for(ulong kIdx = 0; kIdx < kibs; kIdx++){
                for(ulong bIdx = 0; bIdx < 1024; bIdx++){
                    file.put(0);
                }
            }
            // stop
            timer.stop();
        }
    }

#endif

    // Get time of this action
    this->totalTime += timer.totalTime();   

    // Write sequentiaPartial;
    // Throughput correction in case of less than 1 MiB and more than 1 KiB
    throughputWriteSequentialPartial = sizeToUse / timer.totalTime() / amoutToDivideBy;
    execTimeWriteSequentialPartial   = timer.totalTime();
    averageWriteSequentialPartial    = timer.averageTime();
    defaulDevWriteSequentialPartial  = timer.defaultDeviation();

    // Capture data
    this->averageWriteSequential += averageWriteSequentialPartial;
    this->defaulDevWriteSequential += defaulDevWriteSequentialPartial;    
    this->throughputWriteSequential += throughputWriteSequentialPartial;
    this->execTimeWriteSequential += execTimeWriteSequentialPartial;


    // close file
    file.close();
    sync();

    // clear cronometer
    timer.clear();
}

/**
 * Read from file byte by byte sequentially with ifstream
 */
void Benchmark::readSequential(){

    // file handler
    ifstream file;

    // open the file    
    file.open( this->testFilePath.c_str(), ios::binary);

    // size to use
    ulong sizeToUse = sizeRWInMiB ? sizeRWInMiB : 1;
    ulong amoutToDivideBy = sizeToUse == 1 ? 1024 / sizeRWInKiB : 1;

#ifndef LEGACY

    file.rdbuf()->pubsetbuf(0, 0);

    // Set timer set amount of blocks
    timer.setSetSize((ulong)ceil(0.1 * sizeRWInKiB / block_sizeRWInKiB));

    // Write this->fileContent to file in blocks
    ulong left_size = sizeRWInKiB;
    ulong read_size;
    char *p = this->fileContent;

    while (left_size > 0) {
        read_size = (left_size >= block_sizeRWInKiB) ?
            (block_sizeRWInKiB << 10) : (left_size << 10);

        timer.start();
        file.read(p, read_size);
        file.sync();
        timer.stop();

        p += read_size;
        left_size -= (read_size >> 10);
    }

#else

    // Set timer set amount KiBs
    timer.setSetSize(sizeToUse);

    // Read to file using bytes as index
    for(ulong gIdx = 0; gIdx < gibs; gIdx++){
        for(ulong mIdx = 0; mIdx < mibs; mIdx++){

            // Start measure for each MiB
            timer.start();
            for(ulong kIdx = 0; kIdx < kibs; kIdx++){
                for(ulong bIdx = 0; bIdx < 1024; bIdx++){
                    file.get();
                }
            }

            // Stop measure
            timer.stop();
        }
    }

#endif

    // Get time of this action
    this->totalTime += timer.totalTime();

    // ReadSequentialPartial;
    // Throughput correction in case of less than 1 MiB and more than 1 KiB
    throughputReadSequentialPartial = sizeToUse / timer.totalTime() / amoutToDivideBy;
    execTimeReadSequentialPartial   = timer.totalTime();
    averageReadSequentialPartial    = timer.averageTime();
    defaulDevReadSequentialPartial  = timer.defaultDeviation();
    
    // capture data       
    this->throughputReadSequential += throughputReadSequentialPartial;
    this->averageReadSequential += averageReadSequentialPartial;
    this->defaulDevReadSequential += defaulDevReadSequentialPartial;
    this->execTimeReadSequential += execTimeReadSequentialPartial;    

    
    // close file 
    file.close();
    
    // clear cronometer
    timer.clear();
}

/**
 * Write to file randomly
 */
void Benchmark::writeRandom(){
    
    // Start 
    srand( time(NULL));
    
    // File handler
    ofstream file;
    
    ulong seek_tmp = 1;
    
    // open the file    
    file.open(this->testFilePath.c_str(), ios::binary);
    
    // size to use
    ulong sizeToUse = sizeRWInMiB ? sizeRWInMiB : 1;
    ulong amoutToDivideBy = sizeToUse == 1 ? 1024 / sizeRWInKiB : 1;
    
    // Get the byte total size
    ulong totalSize = this->sizeRepeats * this->magSize;
    
    // Write to file using bytes as index with file seeks
    // for each KiB
    
    // TODO: Get the pointer to the of file position. NOTE: it's faster than using <file handler>.tellg(), because of a needless of jump and copy/return new allocated memory
    //file.register_callback(lastIndexEventForFileWriter, totalSize);
    
    // Set timer set amount MiBs
    timer.setSetSize(sizeToUse);
    
    for(ulong gIdx = 0, maxFpos = 0; gIdx < gibs; gIdx++){
        for(ulong mIdx = 0; mIdx < mibs; mIdx++){
            
             // Start measure for each MiB
            timer.start();
            
            for(ulong kIdx = 0; kIdx < kibs; kIdx++){
                
                // random seek position for each KiB
                seek_tmp = (rand() % totalSize);
                file.seekp(seek_tmp, ios::beg);
                maxFpos = seek_tmp;
                
                for(ulong bIdx = 0; bIdx < 1024; bIdx++, maxFpos++){    
                    // Position is write, but it is writing 1024 bytes
                    // if position is 800 byte index, then it will write
                    // 800 + 1024 bytes.
                    
                    // Go back to the first byte if seek position is
                    // wider than the file total size.
                    if(maxFpos >= totalSize){
                        maxFpos = file.tellp();
                        file.seekp(0, ios::beg);
                    }
                    file.put(0);                    
                }
                
            }
            
            // Stop measure
            timer.stop();       
        }        
    }
    

    // Get time of this action
    this->totalTime += timer.totalTime();
        
    // WriteRandomPartial;
    // Throughput correction in case of less than 1 MiB and more than 1 KiB
    throughputWriteRandomPartial = sizeToUse / timer.totalTime() / amoutToDivideBy;
    execTimeWriteRandomPartial   = timer.totalTime();
    averageWriteRandomPartial    = timer.averageTime();
    defaulDevWriteRandomPartial  = timer.defaultDeviation();
    
    // capture results
    this->throughputWriteRandom += throughputWriteRandomPartial;
    this->averageWriteRandom += averageWriteRandomPartial;
    this->execTimeWriteRandom += execTimeWriteRandomPartial;
    this->defaulDevWriteRandom += defaulDevWriteRandomPartial;   

    
    // close file
    file.close();
    
    // clear cronometer
    timer.clear();
}

/**
 * Read from file randomly.
 */
void Benchmark::readRandom(){
        
    // Start 
    srand( time(NULL));
    
    // File handler
    ifstream file;
    ulong seek_tmp = 1;
    
    // size to use
    ulong sizeToUse = sizeRWInMiB ? sizeRWInMiB : 1;
    ulong amoutToDivideBy = sizeToUse == 1 ? 1024 / sizeRWInKiB : 1;
    
    // open the file    
    file.open(this->testFilePath.c_str(), ios::binary);
    
    // Get the byte total size
    ulong totalSize = this->sizeRepeats * this->magSize;
        
    // Set timer set amount MiBs
    timer.setSetSize(sizeToUse);
    
    // Read from file using bytes as index with file seeks
    for(ulong gIdx = 0, maxFpos; gIdx < gibs; gIdx++){
        for(ulong mIdx = 0; mIdx < mibs; mIdx++){
            
            // Start measure for each MiB
            timer.start();
            
            for(ulong kIdx = 0; kIdx < kibs; kIdx++){
                
                // random seek position for each KiB
                seek_tmp = (rand() % totalSize);
                file.seekg(seek_tmp, ios::beg);
                maxFpos = seek_tmp;
                
                for(ulong bIdx = 0; bIdx < 1024; bIdx++, maxFpos++){    
                    // Position is read, but it is read 1024 bytes
                    // if position is 800 byte index, then it will write
                    // 800 + 1024 bytes.
                    
                    // Go back to the first byte if seek position is
                    // wider than the file total size.
                    if(maxFpos >= totalSize){
                        maxFpos = file.tellg();
                        file.seekg(0, ios::beg);
                    }
                    file.get();                    
                }                
            }
            
            // Stop measure
            timer.stop();   
        }        
    }
        
    // Get time of this action
    this->totalTime += timer.totalTime();
        
    // Read Random Partial;
    // Throughput correction in case of less than 1 MiB and more than 1 KiB
    throughputReadRandomPartial  = sizeToUse / timer.totalTime() / amoutToDivideBy;
    averageReadRandomPartial     = timer.averageTime();
    execTimeReadRandomPartial    = timer.totalTime();
    defaulDevReadRandomPartial   = timer.defaultDeviation();
    
    // Capture results
    this->throughputReadRandom  += throughputReadRandomPartial;
    this->averageReadRandom     += averageReadRandomPartial;
    this->execTimeReadRandom    += execTimeReadRandomPartial;
    this->defaulDevReadRandom   += defaulDevReadRandomPartial;


    
    // close file
    file.close();
    
    // clear cronometer
    timer.clear();
}
