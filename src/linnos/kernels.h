#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

void gpu_setup(int n_inputs);
void gpu_clean();
void gpu_setup_inputs(float* inputs, int n);
float gpu_inference();
float gpu_inference_many(int n_inputs);
float gpu_get_result(int n_inputs);