#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2
#define FEAT_31


void clean_naive();
void infer_naive();
void copy_inputs_naive();
void setup_naive();
bool get_result_naive();


void clean_batch();
void infer_batch(int batch_size);
void copy_inputs_batch(int batch_size);
void setup_batch(int batch_size);
bool get_result_batch(int batch_size);

bool prediction_cpu(long *input_vec_i);