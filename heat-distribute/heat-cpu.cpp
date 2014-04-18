#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define WIDTH 10
#define HEIGH 10
#define NUM_POINTS 20
#define MAX_TEMP 10000
#define NUM_PROCS 2
#define NUM_LOOPS 2
int*** init_matrix ()
{
	srand(1234);
	printf("generating input matrix size %d X %d with %d random heat sources \n",WIDTH,HEIGH,NUM_POINTS);
	int** second_matrix= (int**) malloc(HEIGH*sizeof(int*));
	int** matrix = (int**) malloc(HEIGH*sizeof(int*));
	for( int j=0; j< HEIGH; j++)
	{
		matrix[j] = (int*) malloc(WIDTH*sizeof(int));
		second_matrix[j] = (int*) malloc(WIDTH*sizeof(int));
	}
	for(int i = 0; i < NUM_POINTS; i++)
	{
		int width_index = rand()%WIDTH;
		int heigh_index = rand()%HEIGH;
		matrix[heigh_index][width_index] = rand()%MAX_TEMP ;	
		second_matrix[heigh_index][width_index] = matrix[heigh_index][width_index] ;	
//		printf("width : %d    heigh : %d \n",width_index, heigh_index);
	}
 int *** result = (int ***)(malloc(2*sizeof(int**)));
 result[0] = matrix;
 result[1] = second_matrix;
 

	return result;
}
void exchage_data(int** source_matrix, int** dest_matrix)
{
		for(int i=0; i<HEIGH; i++)
			for(int j=0;j<WIDTH; j++)
				{
					source_matrix[i][j] = dest_matrix[i][j];
				}
}
void calculating(int** source_matrix, int** dest_matrix)
{
	for(int k = 0 ; k <  NUM_LOOPS ; k ++)
	{

		for(int i=0; i<HEIGH; i++)
			for(int j=0;j<WIDTH; j++)
				{
					int sum = 0;
					for(int x = -1; x <2; x++)
						for(int y=-1; y < 2 ;y++)
							{
								if( (i+x) <  HEIGH && (j+y) <  WIDTH && (i+x) > 0 &&(j+y) >0)
								sum += source_matrix[i+x][j+y];
							}
					sum/=9;
					dest_matrix[i][j] = sum;				
				}
	//				 printf(" test %d \n",source_matrix[0][0]);	
		exchage_data(source_matrix, dest_matrix);
	
	}			
			
}

int main(int argc, char *argv[])
{
	int ***matrix = init_matrix();
	calculating (matrix[0], matrix[1]);
	return 0;
}

