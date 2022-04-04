#include "srad.h"

// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY CORRUPTION

// srad kernel
__global__ void srad2(int d_lambda, 
										int d_Nr, 
										int d_Nc, 
										long d_Ne, 
										int *d_iN, 
										int *d_iS, 
										int *d_jE, 
										int *d_jW,
										int *d_dN, 
										int *d_dS, 
										int *d_dE, 
										int *d_dW, 
										int *d_c, 
										int *d_I){

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_cN,d_cS,d_cW,d_cE;
	fp d_D;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;												// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;											// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															// make sure that only threads matching jobs run

		// diffusion coefficent
		d_cN = d_c[ei] / 1000.0;														// north diffusion coefficient
		d_cS = d_c[d_iS[row] + d_Nr*col] / 1000.0;										// south diffusion coefficient
		d_cW = d_c[ei] / 1000.0;														// west diffusion coefficient
		d_cE = d_c[row + d_Nr * d_jE[col]] / 1000.0;									// east diffusion coefficient

		// divergence (equ 58)
		d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];// divergence
        d_D /= 1000.0;

		// image update (equ 61) (every element of IMAGE)
		d_I[ei] = d_I[ei] + 0.25/d_lambda*d_D * 1000;								// updates image (based on input time step and divergence)

	}
}
