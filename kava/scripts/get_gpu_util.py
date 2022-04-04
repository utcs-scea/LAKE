import nvidia_smi
import time

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

start = time.time()
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
end = time.time()
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
print(f'query used {end - start} seconds')
