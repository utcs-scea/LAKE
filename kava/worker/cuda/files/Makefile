CU := $(wildcard *.cu)
NAME := $(CU:.cu=)

all: $(NAME)

$(NAME): % : %.cu
	nvcc $< --ptx -gencode arch=compute_30,code=compute_30

clean:
	rm *.ptx
