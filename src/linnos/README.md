
###
  The pdf at the root if this repo is more up to date
###



# Enabling linnos

Everything is done through kernel_hook. that folder has scripts to enable different
types of linnos (fake, cpu, gpu, batchtest).

To run the baseline, the hook MUST be disabled. If you see a bunch of errors 95, it is not disabled.
Run `sudo ./disable_linnos.sh` to disable it.


Changing weights requires changing the main.c file in kernel_hook and requires recompiling
For example, for enabling only on the first device the weights in  `weights_header/weights_15s_1m_100us.trace/header/w_15s_1m_100us.trace_nvme0n1.h`
```
//#include "weights_header/weights_15s_256k_50us.trace/header/w_15s_256k_50us.trace_nvme0n1.h"
#include "weights_header/weights_15s_1m_100us.trace/header/w_15s_1m_100us.trace_nvme0n1.h"

static const char *devices[] = {
    //"/dev/vdb",
	"/dev/nvme0n1",
	//"/dev/nvme1n1",
	//"/dev/nvme2n1",
	0
};

static long *weights[][4] = {
	//{weight_0_T_sde, weight_1_T_sde, bias_0_sde, bias_1_sde}
	{weight_0_T_nvme0n1, weight_1_T_nvme0n1, bias_0_nvme0n1, bias_1_nvme0n1},
	///{weight_0_T_nvme1n1, weight_1_T_nvme1n1, bias_0_nvme1n1, bias_1_nvme1n1},
	///{weight_0_T_nvme2n1, weight_1_T_nvme2n1, bias_0_nvme2n1, bias_1_nvme2n1},
};
```




