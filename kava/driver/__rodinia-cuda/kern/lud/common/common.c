#include <linux/fs.h>
#include <linux/random.h>

#include "common.h"

void stopwatch_start(stopwatch *sw){
    if (sw == NULL)
        return;

    memset(&sw->begin, 0, sizeof(struct timeval));
    memset(&sw->end  , 0, sizeof(struct timeval));

    getnstimeofday(&sw->begin);
}

void stopwatch_stop(stopwatch *sw){
    if (sw == NULL)
        return;

    getnstimeofday(&sw->end);
}

long
get_interval_by_nsec(stopwatch *sw){
    if (sw == NULL)
        return 0;
    return ((sw->end.tv_sec-sw->begin.tv_sec) * 1000000000 + (sw->end.tv_nsec-sw->begin.tv_nsec));
}

long
get_interval_by_usec(stopwatch *sw){
    if (sw == NULL)
        return 0;
    return ((sw->end.tv_sec-sw->begin.tv_sec)*1000000+(sw->end.tv_nsec-sw->begin.tv_nsec)/1000);
}

func_ret_t 
create_matrix_from_file(int **mp, const char* filename, int *size_p){
    int i, j, size;
    int *m;
    struct file *fp = NULL;
    loff_t pos = 0;

	fp = filp_open(filename, O_RDONLY, 0);
	if (!fp) {
		pr_err("Error Reading input file %s\n", filename);
        return RET_FAILURE;
	}

    kernel_read(fp, (char *)&size, sizeof(int), &pos);

    m = (int *) vmalloc(sizeof(int)*size*size);
    if (m == NULL) {
        filp_close(fp, NULL);
        return RET_FAILURE;
    }

    for (i=0; i < size; i++) {
        for (j=0; j < size; j++) {
            kernel_read(fp, (char *)(m+i*size+j), sizeof(int), &pos);
        }
    }

    filp_close(fp, NULL);

    *size_p = size;
    *mp = m;

    return RET_SUCCESS;
}

func_ret_t
create_matrix_from_random(int **mp, int size){
  int *l, *u, *m;
  int i,j,k;
  int rand_num;

  l = (int *)vmalloc(size*size*sizeof(int));
  if ( l == NULL)
    return RET_FAILURE;

  u = (int *)vmalloc(size*size*sizeof(int));
  if ( u == NULL) {
      vfree(l);
      return RET_FAILURE;
  }

  m = *mp;
  m = (int *) vmalloc(sizeof(int)*size*size);

  for (i = 0; i < size; i++) {
      for (j = 0; j < size; j++) {
          if (i > j) {
              get_random_bytes(&rand_num, sizeof(rand_num));
              l[i * size + j] = rand_num % 1000 + 1000;
          } else if (i == j) {
              l[i * size + j] = 1;
          } else {
              l[i * size + j] = 0;
          }
      }
  }

  for (j = 0; j < size; j++) {
      for (i = 0; i < size; i++) {
          if (i > j) {
              u[j * size + i] = 0;
          } else {
              get_random_bytes(&rand_num, sizeof(rand_num));
              u[j * size + i] = rand_num % 1000 + 1000;
          }
      }
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          for (k=0; k <= MIN(i,j); k++)
            m[i*size+j] = l[i*size+k] * u[j*size+k];
      }
  }

  vfree(l);
  vfree(u);

  *mp = m;

  return RET_SUCCESS;
}

static int exp_x_10(int x) {
    // lamda = -0.001
    // return 10*exp(lamda*x)
    int tmp = 1000 * 10, sum = 1000 * 10;
    int i;
    for (i = 1; i <= 8; i++) {
        tmp = tmp * (x / 1000) / i;
        if ((i & 1) == 1)
            sum -= tmp;
        else
            sum += tmp;
    }

    return sum;
}

func_ret_t
create_matrix(int **mp, int size){
  int *m;
  int i,j;
  int *coe;
  int coe_i =0;

  coe = vmalloc(sizeof(int) * (2*size-1));
  for (i=0; i < size; i++)
    {
      coe_i = exp_x_10(i);
      j=size-1+i;
      coe[j]=coe_i;
      j=size-1-i;
      coe[j]=coe_i;
    }

  m = (int *) vmalloc(sizeof(int)*size*size);
  if ( m == NULL) {
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }

  *mp = m;
  vfree(coe);

  return RET_SUCCESS;
}

void
matrix_multiply(int *inputa, int *inputb, int *output, int size){
  int i, j, k;

  for (i=0; i < size; i++)
    for (k=0; k < size; k++)
      for (j=0; j < size; j++)
        output[i*size+j] = inputa[i*size+k] * inputb[k*size+j];

}

func_ret_t
lud_verify(int *m, int *lu, int matrix_dim){
  int i,j,k;
  int *tmp = (int *)vmalloc(matrix_dim*matrix_dim*sizeof(int));

  for (i = 0; i < matrix_dim; i++)
      for (j = 0; j < matrix_dim; j++) {
          int sum = 0;
          int l, u;
          for (k = 0; k <= MIN(i, j); k++) {
              if (i == k)
                  l = 1;
              else
                  l = lu[i * matrix_dim + k];
              u = lu[k * matrix_dim + j];
              sum += l * u;
          }
          tmp[i * matrix_dim + j] = sum;
      }
  pr_info(">>>>>LU<<<<<<<\n");
  for (i = 0; i < matrix_dim; i++) {
      for (j = 0; j < matrix_dim; j++) {
          pr_info("%d ", lu[i * matrix_dim + j]);
      }
      pr_info("\n");
  }
  pr_info(">>>>>result<<<<<<<\n");
  for (i = 0; i < matrix_dim; i++) {
      for (j = 0; j < matrix_dim; j++) {
          pr_info("%d ", tmp[i * matrix_dim + j]);
      }
      pr_info("\n");
  }
  pr_info(">>>>>input<<<<<<<\n");
  for (i = 0; i < matrix_dim; i++) {
      for (j = 0; j < matrix_dim; j++) {
          pr_info("%d ", m[i * matrix_dim + j]);
      }
      pr_info("\n");
  }

  for (i = 0; i < matrix_dim; i++) {
      for (j = 0; j < matrix_dim; j++) {
          if (abs(m[i * matrix_dim + j] - tmp[i * matrix_dim + j]) > 10)
              pr_info("dismatch at (%d, %d): (o)%d (n)%d\n", i, j,
                  m[i * matrix_dim + j], tmp[i * matrix_dim + j]);
      }
  }
  vfree(tmp);

  return RET_SUCCESS;
}

void
matrix_duplicate(int *src, int **dst, int matrix_dim) {
    int s = matrix_dim*matrix_dim*sizeof(int);
    int *p = (int *) vmalloc (s);
    memcpy(p, src, s);
    *dst = p;
}

void
print_matrix(int *m, int matrix_dim) {
    int i, j;
    for (i=0; i<matrix_dim;i++) {
      for (j=0; j<matrix_dim;j++)
        pr_info("%d ", m[i*matrix_dim+j]);
      pr_info("\n");
    }
}
