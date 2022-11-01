#include <linux/types.h>
#include <linux/spinlock.h>

#include "mymemory.h"

// --- Global variables
chunkStatus *head = NULL;
DEFINE_SPINLOCK(lock);

void* shm_start;
void* shm_end;
u64 shm_size;


/* findChunk: search for the first chunk that fits (size equal or more) the request
              of the user.
     chunkStatus *headptr: pointer to the first block of memory in the heap
     u64 size: size requested by the user
     retval: a poiter to the block which fits the request 
	     or NULL, in case there is no such block in the list
*/
chunkStatus* findChunk(chunkStatus *headptr, u64 size) {
    chunkStatus* ptr = headptr;
    while(ptr != NULL) {
        //pr_warn("need %lu+%lu from %lu  (av? %d)\n", size, STRUCT_SIZE, ptr->size, ptr->available);
        if(ptr->available == 1 && ptr->size >= (size + STRUCT_SIZE))
            return ptr;
        ptr = ptr->next;
    }
    return NULL;  
}

/* splitChunk: split one big block into two. The first will have the size requested by the user.
  	       the second will have the remainder.
     chunkStatus* ptr: pointer to the block of memory which is going to be splitted.
     u64 size: size requested by the user
     retval: void, the function modifies the list
*/
void splitChunk(chunkStatus* ptr, u64 size)
{
    chunkStatus *newChunk;	
    newChunk = (chunkStatus*)(ptr->end + size);
    newChunk->size = ptr->size - size - STRUCT_SIZE;
    newChunk->available = 1;
    newChunk->next = ptr->next;
    newChunk->prev = ptr;
    if((newChunk->next) != NULL)
        (newChunk->next)->prev = newChunk;

    ptr->size = size;
    ptr->available = 0;
    ptr->next = newChunk;
}

/* mergeChunkPrev: merge one freed chunk with its predecessor (in case it is free as well)
     chunkStatus* freed: pointer to the block of memory to be freed.
     retval: void, the function modifies the list
*/
void mergeChunkPrev(chunkStatus *freed)
{ 
    chunkStatus *prev;
    prev = freed->prev;
    
    if(prev != NULL && prev->available == 1) {
        prev->size = prev->size + freed->size + STRUCT_SIZE;
        prev->next = freed->next;
        if( (freed->next) != NULL )
        (freed->next)->prev = prev;
    }
}

/* mergeChunkNext: merge one freed chunk with the following chunk (in case it is free as well)
     chunkStatus* freed: pointer to the block of memory to be freed.
     retval: void, the function modifies the list
*/
void mergeChunkNext(chunkStatus *freed)
{  
    chunkStatus *next;
    next = freed->next;
    
    if(next != NULL && next->available == 1) {
        freed->size = freed->size + STRUCT_SIZE + next->size;
        freed->next = next->next;
        if( (next->next) != NULL )
            (next->next)->prev = freed;
    }
}

void mymalloc_init(void* ptr, u64 size) {
    shm_start = ptr;
    shm_end = ptr+size;
    shm_size = size;
    
    head = ptr;
    head->size = size - STRUCT_SIZE;
    head->available = 1;
    head->next = NULL;
    head->prev = NULL;
    //pr_warn("inited mymalloc\n");
}

/* mymalloc: allocates memory on the heap of the requested size. The block
             of memory returned should always be padded so that it begins
             and ends on a word boundary.
     u64 size: the number of bytes to allocate.
     retval: a pointer to the block of memory allocated or NULL if the 
             memory could not be allocated. 
             (NOTE: the system also sets errno, but we are not the system, 
                    so you are not required to do so.)
*/
void *mymalloc(u64 _size) {
    u64 size = MY_ALIGN(_size);
    unsigned long flags;
    chunkStatus *freeChunk = NULL;

    //pthread_mutex_lock(&lock);
    spin_lock_irqsave(&lock, flags);
    freeChunk = findChunk(head, size);
    if(freeChunk == NULL) {				//Didn't find any chunk available
        //pthread_mutex_unlock(&lock);
        spin_unlock_irqrestore(&lock, flags);
        return NULL;
    } //A chunk was found
    if(freeChunk->size > size) //If chunk is too big, split it
        splitChunk(freeChunk, size);

    //pthread_mutex_unlock(&lock);    
    spin_unlock_irqrestore(&lock, flags);
    return freeChunk->end;
}

/* myfree: unallocates memory that has been allocated with mymalloc.
     void *ptr: pointer to the first byte of a block of memory allocated by 
                mymalloc.
     retval: 0 if the memory was successfully freed and 1 otherwise.
             (NOTE: the system version of free returns no error.)
*/
char myfree(void *ptr) {
    unsigned long flags;
	chunkStatus *toFree;

	spin_lock_irqsave(&lock, flags);
	//pthread_mutex_lock(&lock);

	toFree = ptr - STRUCT_SIZE;	
	if(toFree >= head ) {   //check if after end of buffer
        toFree->available = 1;	
        mergeChunkNext(toFree);
        mergeChunkPrev(toFree);
        //pthread_mutex_unlock(&lock);
        spin_unlock_irqrestore(&lock, flags);
        return 0;
	}
	else {
        //#endif
        //pthread_mutex_unlock(&lock);
        spin_unlock_irqrestore(&lock, flags);
        return 1;
	}
}

