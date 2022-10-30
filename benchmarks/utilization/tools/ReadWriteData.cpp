#include <fstream>
#include <iostream>
#include <unistd.h>

void dropCache()
{
    FILE *fp = fopen("/proc/sys/vm/drop_caches", "w");
    fprintf(fp, "3");
    fclose(fp);
    //system("./drop_cache");
}

#include <stdio.h>
#include <stdlib.h>

int main()
{
   FILE *fptr;
   fptr = fopen("temp.dat","w");

   if(fptr == NULL)
   {
      printf("Error!");   
      exit(1);             
   }
   
   long GB = 100*100*100*100*10;
   long MB = 1000*1000;
   char *data = (char *) malloc (100 * MB);

   for(int i = 0; i < 2 * GB; i+=100 * MB)
    fprintf(fptr,"%s",data);

   fclose(fptr);


   if ((fptr = fopen("temp.dat","r")) == NULL){
       printf("Error! opening file");
       exit(1);
   }

   while (fscanf(fptr,"%s",data) != EOF);
   fclose(fptr); 

   return 0;
}
