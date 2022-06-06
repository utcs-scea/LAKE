#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <xxhash.h>
#include <getopt.h>

#define RANDOM_SEED 12345
#define PAGE_SIZE getpagesize()

struct args {
  int n_pages;
  double frac_identical;
};

void parse_args(int argc, char **argv, struct args *args) {
  int args_valid = 0;
  int opt;

  // All pages are identical by default
  args->frac_identical = 1.0;

  // Parse args
  while ((opt = getopt(argc, argv, "n:f:")) != -1) {
    switch (opt) {
    case 'n':
      args_valid = 1;
      args->n_pages = atoi(optarg);
      break;
    case 'f':
      args->frac_identical = atof(optarg);
      break;
    }
  }
  if (!args_valid) {
    printf("Usage: ./generator -n <num_pages> -f <fraction_of_identical_pages>\n");
    exit(1);
  }
}

int main(int argc, char **argv) {
  struct args args = { 0 };
  parse_args(argc, argv, &args);

  // Set random seed
  srand(RANDOM_SEED);

  // Create a model identical page
  char *page = (char *) malloc(PAGE_SIZE);
  for (int i = 0; i < PAGE_SIZE; ++i) {
    page[i] = i;
  }

  // mmap pages
  void *pages = mmap(NULL, PAGE_SIZE * args.n_pages, 
    PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  for (int i = 0; i < args.n_pages; ++i) {
    char *page_start = ((char *) pages) + PAGE_SIZE * i;
    memcpy(page_start, page, PAGE_SIZE);

    // Create some non-identical pages
    if (i > args.n_pages * args.frac_identical) {
      page_start[rand() % PAGE_SIZE] = rand();
    }
  }

  // Mark pages as mergeable for KSM
  int ret = madvise(pages, PAGE_SIZE * args.n_pages, MADV_MERGEABLE);
  if (ret) {
    printf("Failed to advise\n");
  }

  // Compute the expected checksum value
  uint32_t seed = 17;
  uint32_t checksum = XXH32(page, PAGE_SIZE, seed);

  uint32_t *page_contents = (uint32_t *) page;
  printf("Allocated %d pages\n", args.n_pages);
  printf("%0.2f%% of them should have the same content\n", args.frac_identical * 100);
  printf("Hanging indefinitely\n");
  while(1);
}
