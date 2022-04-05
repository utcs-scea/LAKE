#include "Timer.h"

/**
 * Default Constructor.
 * Set values to default.
 */
Timer::Timer(){
    this->lastDuration = 0;
    this->acumulator = 0;
    this->size = 1;
}

/**
 * Register the initial time to subtract on stop.
 */
void Timer::start(){
    gettimeofday(&startTime, NULL);
}

/**
 * Register current time and subtract from start
 * time to get the difference. Set to the last 
 * duration field the last time differnce. Sum
 * to the accumulator the partial result to make
 * calculation later. Push the individual result
 * to a partial array for future calculation too.
 */
void Timer::stop(){

    timeval endTime;
    long seconds, useconds;
    double duration;

    gettimeofday(&endTime, NULL);

    seconds  = endTime.tv_sec  - startTime.tv_sec;
    useconds = endTime.tv_usec - startTime.tv_usec;

    duration = seconds + useconds/1000000.0;
    this->indResult.push_back(duration);
    this->lastDuration = duration;
    this->acumulator += this->lastDuration;
}

/**
 * Get the last duration
 * @return last duration
 */
double Timer::getDuration(){
    return this->lastDuration;
}

/**
 * Reset variables and clear time array.
 */
void Timer::clear(){
    this->acumulator = 0;
    this->lastDuration = 0;
    this->indResult.clear();
    this->size = 1;
}

/**
 * Get the total time of the individual
 * operations getting accumulator value.
 * @return 
 */
double Timer::totalTime(){
    return this->acumulator;
}

/**
 * Set the amount of individual results
 * to average, variance and default 
 * deviation calculations.
 * @param size
 */
void Timer::setSetSize(ulong size){    
    if (size == 0) size = 1;
    this->indResult.reserve(size);
    this->size = size;
}

/**
 * Calculate the average time.
 * @return average time
 */
long double Timer::averageTime(){    
    return this->acumulator / size;
}

/**
 * Calculates the variance
 * @return variance
 */
double Timer::variance(){
    double average = this->averageTime();
    double sumSamples = 0;
    
    for(ulong aIdx = 0; aIdx < size; aIdx++){
        
        double diffQuad = pow((indResult[aIdx] - average), 2);
        sumSamples += diffQuad;
    }
    
    return sumSamples / size;    
}

/**
 * Calculates the default deviation.
 * @return default deviation
 */
double Timer::defaultDeviation(){
    long double res = sqrtl( this->variance());
    return res ? res : 0;
}
