#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <cuda.h>

#define BUFF_LEN 256 
#define N 64
#define WIDTH 10000
#define HEIGH 10000
#define NUM_POINTS 2000
#define MAX_TEMP 10000
#define NUM_PROCS 2
#define NUM_LOOPS 2000
#define DIFFERENCE 50
// Enumeration of CUDA devices accessible for the process.
void enumCudaDevices(char *buff)
{
    char tmpBuff[BUFF_LEN];
    int i, devCount;

    cudaGetDeviceCount(&devCount);
    sprintf(tmpBuff," %3d", devCount);
    strncat(buff, tmpBuff, BUFF_LEN);

    for (i = 0; i < devCount; i++)
    {
        cudaDeviceProp devProp;

        cudaGetDeviceProperties(&devProp, i);
        sprintf(tmpBuff, "  %d:%s", i, devProp.name);
        strncat(buff, tmpBuff, BUFF_LEN);
    }
    
}

void test_dst(int* dev_dst, int num_rows)
{
	int* test_dst = (int* )malloc((num_rows)*WIDTH*sizeof(int));
	cudaMemcpy(test_dst, dev_dst, (num_rows)*WIDTH*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i<num_rows;i++)
	{
		for(int j = 0; j <WIDTH;j++)
			printf("%d ",test_dst[i*WIDTH+j]);
		printf("\n");
	}
}

void test_matrix(int* matrix, int num_rows)
{
	for(int i = 0; i<num_rows;i++)
	{
		for(int j = 0; j <WIDTH;j++)
			printf("%d ",matrix[i*WIDTH+j]);
		printf("\n");
	}
}

int* init_matrix ()
{
	srand(1234);
	printf("generating input matrix size %d X %d with %d random heat sources \n",WIDTH,HEIGH,NUM_POINTS);
	int* matrix = (int*) malloc((WIDTH*HEIGH)*sizeof(int));
	memset(matrix, 0, sizeof(matrix[0]) * WIDTH* HEIGH);	
	for(int i = 0; i < NUM_POINTS; i++)
	{
		int width_index = rand()%WIDTH;
		int heigh_index = rand()%HEIGH;
		matrix[WIDTH*heigh_index + width_index] = rand()%MAX_TEMP ;	
//		printf("width : %d    heigh : %d \n",width_index, heigh_index);
	}

	return matrix;
}

int* scatter_matrix(int* source_matrix, int rows_per_proc)
{
	int size = rows_per_proc*WIDTH*sizeof(int);
	int *per_proc_matrix = (int*) malloc(size);
	//MPI_Scatter(matrix, rows_per_proc*WIDTH , MPI_INT, recv_matrix, rows_per_proc*WIDTH , MPI_INT, 0 , MPI_COMM_WORLD);
	per_proc_matrix= (int*) malloc(rows_per_proc * WIDTH * sizeof(int));

	MPI_Scatter(source_matrix, rows_per_proc*WIDTH , MPI_INT, per_proc_matrix , rows_per_proc*WIDTH , MPI_INT, 0 , MPI_COMM_WORLD);

//	printf("\n \ntest scatter_src \n");	
//	test_matrix(per_proc_matrix, rows_per_proc);
	
	return per_proc_matrix;	
}

__global__ void heat_distribute(int* source_matrix, int* dest_matrix,int numthreads)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index <numthreads)
	{
		int heigh_index = index/WIDTH + 1 ;
		int width_index = index%WIDTH;
		int sum=0;
		for(int x = -1; x <2; x++)
		 for(int y=-1; y < 2 ;y++)
		{
			int new_width_index = width_index + x;
			int new_heigh_index = heigh_index + y;
			sum += 1.0f*source_matrix[new_heigh_index*WIDTH  + new_width_index];
		}	
		dest_matrix[index+WIDTH] = sum/9;
	}
} 

void update_matrix (int* dev_src, int* dev_dst, int* rows_per_proc, int myrank)
{
	cudaMemcpy(dev_src, dev_dst, (rows_per_proc[myrank]+2)*WIDTH*sizeof(int), cudaMemcpyDeviceToDevice);
}


int** init_gpu_memory(int* rows_per_proc,int myrank)
{
	int* dev_src, *dev_dst;
	int row_size = WIDTH*sizeof(int);

//	int size = rows_per_proc*row_size;

	cudaMalloc((void**)&dev_src,(rows_per_proc[myrank]+2)*row_size);
	cudaMalloc((void**)&dev_dst,(rows_per_proc[myrank]+2)*row_size);
	
//	cudaMemcpy(dev_src+WIDTH, src_matrix, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_dst , dev_src, size+2*row_size, cudaMemcpyDeviceToDevice);
	
	
	
	int ** result = (int**) malloc(2*sizeof(int*));
	result[0] = dev_src;
	result[1]= dev_dst;
	//~ printf("\n \ntest matrix_src %d \n",myrank);	
	//~ test_matrix(src_matrix, rows_per_proc);
	//~ printf("\n \ntest dev_src %d\n",myrank);	
	//~ test_dst(dev_src, rows_per_proc+2);
	//~ printf("\n \ntest dev_dst %d\n",myrank);
	//~ test_dst(dev_dst,rows_per_proc+2);
	
	return result;	
}

void exchange_data(int* dev_src, int* rows_per_proc, int myrank, int numprocs)
{

	//int row_size = WIDTH*sizeof(int);
//	int* send_row = (int*)malloc(row_size);
//	int *recv_row = (int*) malloc (row_size);
	MPI_Status status[2];// = new MPI_Status();
	MPI_Request request[2];
	//int offset = (rows_per_proc+1) * row_size;
	
	if(myrank == 0)
	{
		int offset = (rows_per_proc[myrank]) * WIDTH;

//		cudaMemcpy(send_row, dev_src+offset, row_size, cudaMemcpyDeviceToHost);
		MPI_Isend(dev_src+offset, WIDTH , MPI_INT, myrank+1, 0, MPI_COMM_WORLD,&request[0]);
		MPI_Irecv(dev_src+offset+WIDTH , WIDTH , MPI_INT, myrank+1 ,0, MPI_COMM_WORLD,&request[1]);		
//		cudaMemcpy(dev_src+offset+WIDTH, recv_row, row_size, cudaMemcpyHostToDevice);

	}	
	if (myrank == numprocs-1)
	{
		int offset = WIDTH;
//		cudaMemcpy(send_row, dev_src+offset, row_size, cudaMemcpyDeviceToHost);
		MPI_Isend(dev_src+offset, WIDTH , MPI_INT, myrank-1, 0, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(dev_src, WIDTH , MPI_INT, myrank-1 ,0, MPI_COMM_WORLD,&request[1]);		
//		cudaMemcpy(dev_src, recv_row, row_size, cudaMemcpyHostToDevice);		
	}
	else
	{
	//TODO: more than two GPUs 
	}
	MPI_Waitall(2 ,request ,status ) ;
	//~ printf("\n \ntest exchange data %d\n",myrank);	
	//~ test_dst(dev_src, rows_per_proc+2);

}

void run_heat_kernel(int myrank, int* dev_src, int* dev_dst,int numprocs, int* rows_per_proc)
{

	//~ printf("row per proc: %d \n", rows_per_proc);
	int numthreads = rows_per_proc[myrank]*WIDTH;
	if(numprocs >1)
		exchange_data(dev_src, rows_per_proc, myrank, numprocs);	
	
	heat_distribute<<<1,numthreads>>>(dev_src,dev_dst,numthreads);

	//~ printf("\n \ntest dev_src %d\n",myrank);	
	//~ test_dst(dev_src, rows_per_proc+2);
	//~ printf("\n \ntest dev_dst %d\n",myrank);
	//~ test_dst(dev_dst,rows_per_proc+2);
	MPI_Barrier(MPI_COMM_WORLD);
	
	update_matrix (dev_src, dev_dst, rows_per_proc, myrank);
	

   	//printf ("return from kernel %d , with value = %d \n", myrank, c);
	
}


int main(int argc, char *argv[])
{
    int i, myrank, numprocs;
    char pName[MPI_MAX_PROCESSOR_NAME],
    buff[BUFF_LEN];
	int difference = DIFFERENCE;
	int loops = NUM_LOOPS;

	int* source_matrix;
	int* per_proc_matrix;
	int* dev_src;
	int* dev_dst;
	int* rows_per_proc;
	
	if(argc == 3 )
	{
		difference = atoi(argv[2]);
		loops = atoi(argv[1]);
	}
		double start_time;
		double end_time; 
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(pName, &i);	
	
	int average_number = HEIGH/numprocs;
	rows_per_proc = (int*) malloc(numprocs*sizeof(int));
	rows_per_proc[0] = average_number;
	if(numprocs >1 )
	{
		rows_per_proc[0]= average_number + difference;
		for(int i=1; i<numprocs;i++)
		rows_per_proc[i] = average_number - difference/(numprocs-1);
	}
			
    sprintf(buff, "%-15s %3d", pName, myrank);

// Find local CUDA devices

    enumCudaDevices(buff);
//    run_add_kernel(myrank);
    
// Collect and print the list of CUDA devices from all MPI processes
    if (myrank == 0)
    {
        char devList[10][BUFF_LEN];
 
        MPI_Gather(buff, BUFF_LEN, MPI_CHAR,devList, BUFF_LEN, MPI_CHAR,0, MPI_COMM_WORLD);
        for (i = 0; i < numprocs; i++)
            printf("%s\n", devList + i);
    }
    else
        MPI_Gather(buff, BUFF_LEN, MPI_CHAR, NULL, 0, MPI_CHAR, 0, MPI_COMM_WORLD);

//heat distribution begin here
	int ** dev_pointer = init_gpu_memory(rows_per_proc, myrank);	
	dev_src = dev_pointer[0];
	dev_dst = dev_pointer[1];

	//printf(" proccess %d : %d  rows\n",myrank,  rows_per_proc[myrank]);
	
   if(myrank == 0)
	{
		source_matrix = init_matrix();
/*		 for(int i=0; i<HEIGH ; i++)
		 {
			 for(int j = 0; j<WIDTH;j++)
				 printf("%d ", source_matrix[i*WIDTH+j]);
			 printf("\n");
		 }

*/
	for(int i=0;i<numprocs;i++)
		printf("process %d : %d rows \n",i, rows_per_proc[i]);
	printf("launching computing kernel .... \n\n");
	start_time = MPI_Wtime();
	}
	
	if(numprocs>1)
	{
		//dev_src = scatter_matrix(source_matrix,rows_per_proc);
	
	//	per_proc_matrix= (int*) malloc(rows_per_proc*WIDTH*sizeof(int));
	//	MPI_Scatter(source_matrix, rows_per_proc[myrank]*WIDTH , MPI_INT, dev_src+WIDTH , rows_per_proc[myrank]*WIDTH , MPI_INT, 0 , MPI_COMM_WORLD);
		MPI_Request request[1];
		MPI_Status status[1];
		if(myrank ==0)
		{
			MPI_Isend(source_matrix+rows_per_proc[0]*WIDTH, rows_per_proc[1]*WIDTH, MPI_INT,1, 0, MPI_COMM_WORLD, &request[0]);
			cudaMemcpy(dev_src+WIDTH, source_matrix, rows_per_proc[0] *WIDTH * sizeof(int), cudaMemcpyHostToDevice);
			MPI_Waitall(1,request,status);
		}
		else
		{
			
			MPI_Recv(dev_src+WIDTH,rows_per_proc[1]*WIDTH,MPI_INT, 0,0, MPI_COMM_WORLD,&status[0]); 
		}
			
	}
	else
	{
		per_proc_matrix = (int *) malloc(WIDTH*HEIGH*sizeof(int));
		for(int i =0;i <WIDTH*HEIGH; i++)
			per_proc_matrix[i] = source_matrix[i];	
		cudaMemcpy(dev_src+WIDTH, per_proc_matrix, rows_per_proc[myrank]*WIDTH*sizeof(int),cudaMemcpyHostToDevice);
	}
//		printf("\n \ntest scatter_src %d \n", myrank);	
//		test_dst(dev_src, rows_per_proc[myrank]+2);
	

	for(int i = 0; i<loops; i++)
		run_heat_kernel(myrank, dev_src, dev_dst, numprocs, rows_per_proc);

//	cudaMemcpy(per_proc_matrix, dev_dst+WIDTH, rows_per_proc*WIDTH*sizeof(int), cudaMemcpyDeviceToHost);
	
	if(numprocs >1)
		MPI_Gather(dev_dst+WIDTH, rows_per_proc[myrank]*WIDTH, MPI_INT, source_matrix ,rows_per_proc[myrank]*WIDTH, MPI_INT, 0 , MPI_COMM_WORLD);
	else
	{
//		for(int i =0 ; i < WIDTH*HEIGH; i++)
//			source_matrix[i] = per_proc_matrix[i];

		cudaMemcpy(source_matrix, dev_dst+WIDTH, rows_per_proc[myrank]*WIDTH*sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	if(myrank == 0)
	 {
		 end_time = MPI_Wtime();
		 printf("Elapsed time: %f secs", (end_time - start_time));
/*		 printf("\n result : \n");
		 for(int i=0; i<HEIGH ; i++)
		 {
			for(int j = 0; j<WIDTH;j++)
				printf("%d ", source_matrix[i*WIDTH+j]);
			printf("\n");
		 }
*/	}

	 MPI_Finalize();
    return 0;
}


