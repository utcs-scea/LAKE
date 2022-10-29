#ifndef CPU_KML_H
#define CPU_KML_H


void cpu_predict_readahead_class(int batch_size);
void cleanup(void);
void setup_cpu(void);
void setup_input(int batch_size);

#endif