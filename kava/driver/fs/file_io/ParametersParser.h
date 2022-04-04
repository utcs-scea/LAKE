/* 
 * File:   ParametersParser.h
 * Author: gilberto
 *
 * Created on June 16, 2015, 9:06 PM
 */

#ifndef PARAMETERSPARSER_H
#define	PARAMETERSPARSER_H

#include "Benchmark.h"
#include <iostream>
#include <string.h>

class ParametersParser{
    std::string mountPoint;
    int repeats;
    int timesMag;
    Benchmark::BlockMagType magType;
    int blockTimesMag;
    Benchmark::BlockMagType blockMagType;

public:

    ParametersParser(int argc, char** argv){
        this->parse(argc, argv);
    }

    /**
     * Parse the arguments passed
     * @param argc
     * @param argv
     */
    void parse(int argc, char** argv)
    {
        if(argc < 5) {
            std::string usage = "";
            usage = "<mount point> <repeats> <file_size (magnitude)> <block_size>";
            std::cout << "Usage: " << argv[0] << usage << std::endl;
            exit(0);
        }

        this->mountPoint = (const char*)argv[1];
        this->repeats = atoi(argv[2]);

        // Defaults
        this->magType = Benchmark::MagKiB;
        this->timesMag = 0;
        this->blockMagType = Benchmark::MagKiB;
        this->blockTimesMag = 0;

        // parse magnitude
        char* lastParamenter = argv[3];
        int lastParamenterSize = strlen(lastParamenter);

        for(int idx = 0; idx < lastParamenterSize; idx++) {

            char c = lastParamenter[idx];

            if ( (c == 'k' || c == 'K') && idx != 0) {
                this->magType = Benchmark::MagKiB;
            }
            else if ( (c == 'm' || c == 'M') && idx != 0) {
                this->magType = Benchmark::MagMiB;
            }
            else if ( (c == 'g' || c == 'G') && idx != 0) {
                this->magType = Benchmark::MagGiB;
            }
            else if ( isdigit(c)) {
                this->timesMag *= 10;
                this->timesMag += (c -'0');
            }
            else {
                // Exit on failure of parsing the 3th parameter
                cout << "Error on 3th parameter. Format accepted: (count)[k|K|m|M|g|G]" << endl;
                cout << "E.g.: " << argv[0] << " ~/ 1 1G 4K" << endl;
                exit(EXIT_FAILURE);
            }
        }

        // Exit on failure if the file size is wider than 8GiB
        if (this->magType == Benchmark::MagGiB && this->timesMag > 8){
            cout << "Maximum permitted file size is 8 GiBs" << endl;
            exit(EXIT_FAILURE);
        }

        // parse block_size
        lastParamenter = argv[4];
        lastParamenterSize = strlen(lastParamenter);

        for(int idx = 0; idx < lastParamenterSize; idx++) {

            char c = lastParamenter[idx];

            if ( (c == 'k' || c == 'K') && idx != 0) {
                this->blockMagType = Benchmark::MagKiB;
            }
            else if ( (c == 'm' || c == 'M') && idx != 0) {
                this->blockMagType = Benchmark::MagMiB;
            }
            else if ( (c == 'g' || c == 'G') && idx != 0) {
                this->blockMagType = Benchmark::MagGiB;
            }
            else if ( isdigit(c)) {
                this->blockTimesMag *= 10;
                this->blockTimesMag += (c -'0');
            }
            else {
                // Exit on failure of parsing the 3th parameter
                cout << "Error on 4th parameter. Format accepted: (count)[k|K|m|M|g|G]" << endl;
                cout << "E.G.: " << argv[0] << " ~/ 1 1G 4K" << endl;
                exit(EXIT_FAILURE);
            }
        }

        // Exit on failure if the file size is wider than 8GiB
        if (this->blockMagType == Benchmark::MagGiB && this->blockTimesMag > 1){
            cout << "Maximum permitted block size is 1 GiBs" << endl;
            exit(EXIT_FAILURE);
        }
    }
    
    /**
     * Getter repeats
     * @return repeats
     */
    int getRepeats(){
        return this->repeats;
    }
    
    /**
     * Getter times the magnitude repeats
     * @return timesMag
     */
    int getTimesMag(){
        return this->timesMag;
    }

    int getBlockTimesMag(){
        return this->blockTimesMag;
    }

    /**
     * Getter to the mount point
     * @return mountPoint
     */
    std::string getMountPoint(){
        return this->mountPoint;
    }
    
    /**
     * Getter to the magnitude type
     * @return magType
     */
    Benchmark::BlockMagType getMagType(){
        return this->magType;
    }

    Benchmark::BlockMagType getBlockMagType(){
        return this->blockMagType;
    }
};


#endif	/* PARAMETERSPARSER_H */

