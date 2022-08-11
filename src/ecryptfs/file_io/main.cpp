/*
 * File:   main.cpp
 * Author: gilberto
 *
 * Created on June 1, 2015, 8:39 PM
 */


#include <iostream>
#include <string.h>
#include "Benchmark.h"
#include "ParametersParser.h"

using namespace std;

/*
 * Get the paramenters, parse and use it to set
 * Benchmark environment.
 */
int main(int argc, char** argv)
{
    ParametersParser pp(argc, argv);
    string mp = pp.getMountPoint();
    int repeats = pp.getRepeats();
    int timesMag = pp.getTimesMag();
    Benchmark::BlockMagType magType = pp.getMagType();
    int block_timesMag = pp.getBlockTimesMag();
    Benchmark::BlockMagType block_magType = pp.getBlockMagType();

    Benchmark bnk;
    bnk.setEnv(mp, repeats, timesMag, magType, block_timesMag, block_magType);
    bnk.run();

    return 0;
}
