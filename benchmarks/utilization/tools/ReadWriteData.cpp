#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace std;

void dropCache()
{
    FILE *fp = fopen("/proc/sys/vm/drop_caches", "w");
    fprintf(fp, "3");
    fclose(fp);
    //system("./drop_cache");
}

int main () {
   //char data[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK...";
   char *data = (char*) malloc (100);
   long GB = 100*100*100*100*10;
   long MB = 1000*1000;
   int data_size = 40;
   // open a file in write mode.
   ofstream outfile;
   outfile.open("temp.dat");

   for(int i = 0; i < 2 * MB; i+=data_size)
    outfile << data << endl;

   outfile.close();
   dropCache();
   //sleep
   sleep(2);

   // open a file in read mode.
   ifstream infile; 
   infile.open("temp.dat"); 

   while (!infile.eof())
    {
        infile >> data;
    }

   // close the opened file.
   infile.close();

   return 0;
}
