//Defines and macros
#include <linux/types.h>

#define MY_ALIGN_SIZE 8
#define MY_ALIGN(size) (((size) + (MY_ALIGN_SIZE-1)) & ~(MY_ALIGN_SIZE-1))

// --- Struct to store memory's block metadata
typedef struct chunkStatus {
  u64 size;
  u64 available;
  struct chunkStatus* next;
  struct chunkStatus* prev;
  char end[1]; 		//end represents the end of the metadata struct
} chunkStatus;

#define STRUCT_SIZE sizeof(chunkStatus)

chunkStatus* findChunk(chunkStatus *headptr, u64 size);
void splitChunk(chunkStatus* ptr, u64 size);
void mergeChunk(chunkStatus *freed);
void *mymalloc(u64 _size);
char myfree(void *ptr);
void mymalloc_init(void* ptr, u64 size);

extern void* shm_start;
extern void* shm_end;
extern u64 shm_size;