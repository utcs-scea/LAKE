#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    FILE *fp, *wfp;
    char *input_f, output_f[128];
    int i;
    size_t size;
    int no_of_nodes, edge_list_size, source;
    int start, edgeno, id, cost;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
		exit(0);
	}

	input_f = argv[1];
    strcpy(output_f, input_f);
    size = strlen(input_f);
    strcpy(&output_f[size-3], "bin");
    printf("Dump input file to %s\n", output_f);

	fp = fopen(input_f, "r");
    wfp = fopen(output_f, "wb");
	if (!fp || !wfp) {
		printf("Error Reading or writing graph file\n");
		return -1;
	}

	fscanf(fp, "%d", &no_of_nodes);
    fwrite(&no_of_nodes, sizeof(int), 1, wfp);

	/* initalize the memory */
	for(i = 0; i < no_of_nodes; i++)  {
		fscanf(fp, "%d %d", &start, &edgeno);
        fwrite(&start, sizeof(int), 1, wfp);
        fwrite(&edgeno, sizeof(int), 1, wfp);
	}

	/* read the source node from the file */
	fscanf(fp, "%d", &source);
    fwrite(&source, sizeof(int), 1, wfp);

	fscanf(fp, "%d", &edge_list_size);
    fwrite(&edge_list_size, sizeof(int), 1, wfp);

	for(i = 0; i < edge_list_size ; i++) {
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
        fwrite(&id, sizeof(int), 1, wfp);
        fwrite(&cost, sizeof(int), 1, wfp);
	}

	if (fp)
		fclose(fp);
    if (wfp)
        fclose(wfp);

    return 0;
}
