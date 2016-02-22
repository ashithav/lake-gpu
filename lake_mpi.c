/* 
Single Author info:
avelayu Ashitha Velayudhan
Group info:
1. avelayu Ashitha Velayudhan
2. prajago4 Priyadarshini Rajagopal
3. smnatara Sekharan Muthusamy Natarajan
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

#define DEFAULT_TAG (0)
#define ROOT (0)

void init(double *u, double *pebbles, int n);
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);
extern int run_gpu_mpi(double*u, double* u0, double* u1, double* pebbles, int n, double h, double end_time, int nthreads,
                   double* u0_row_recvd,double* u0_col_recvd, double corner,
                   int rank );

int range(int rank, int n, int* x_min, int* x_max, int* y_min, int* y_max)
{
    if (rank == 0)
    {
        *x_min = 0;
        *x_max = n/2-1;
        *y_min = 0;
        *y_max = n/2-1;
    }
    else if (rank == 1)
    {
        *x_min = 0;
        *x_max = n/2-1;
        *y_min = n/2;
        *y_max = n-1;
    }
    else if (rank == 2)
    {
        *x_min = n/2;
        *x_max = n-1;
        *y_min = 0;
        *y_max = n/2-1;
    }
    else if (rank == 3)
    {
        *x_min = n/2;
        *x_max = n-1;
        *y_min = n/2;
        *y_max = n-1;
    }
    return 0;
}
/* n - dimension of grid 
 * From the values (n * n) stored in 1-D array input, extract the values rank is working on (1 quadrant)
 * and put it in output array (n/2 * n/2)
 */
int divide_into_quadrants(int n, int rank,double* input, double* output)
{
    int start_row = 0;
    int end_row   = 0;
    int start_index = 0;
    int row_size  = n/2;
    int x_min,x_max;
    int y_min,y_max;
    int r = 0;
    double* output_ptr = output;

    range(rank, n, &x_min, &x_max, &y_min, &y_max);
    start_row = x_min;
    end_row = x_max;

    //printf("Rank:%d start_row:%d end_row:%d\n",rank,start_row, end_row);

        
    for (r = start_row; r <= end_row; r++)
    {
        if (rank == 0 || rank == 2)
        {
            /*copy the first half of rows */
            start_index = r * n;
        }
        else
        {
            /*copy the last half of rows */
            start_index = r * n + row_size;
        }

        //printf("Rank%d: Copying array starting from index:%d to:%d\n",rank,start_index,(start_index+row_size-1));
        memcpy(output_ptr, input+start_index, row_size*sizeof(double));
        output_ptr += row_size;
    }
}

/*
 * Quadrant Division among ranks 
 * |--------------------------------------
 * |                |                    |
 * |                |                    |
 * |  rank0         |      rank2         |
 * |                |                    |
 * |                |                    |
 * ---------------------------------------
 * |                |                    |
 * |                |                    |
 * |  rank2         |      rank3         |
 * |                |                    |
 * |                |                    |
 * |-------------------------------------
 */
/* We need boundary values from the neighbour 
 * rank 0 and rank 2 need column on the right 
 * rank 1 and rank 3 need column on the left */

int get_column(int n, int for_rank, double* input, double* output)
{
    int i=0,j=0,k=0;
    int idx = 0;
    int x_min,x_max;
    int y_min,y_max;
    int rank = 0;

    if (for_rank == 0)
        rank = 1;
    if (for_rank == 1)
        rank = 0;
    if (for_rank == 2)
        rank = 3;
    if (for_rank == 3)
        rank = 2;

    x_min = 0;
    x_max = (n/2)-1;
    y_min = 0;
    y_max = (n/2)-1;
    if (rank == 0 || rank == 2)
    {
        j = y_max;
        k = 0;
        for(i=x_min; i<= x_max;i++)
        {
            idx = i*(n/2)+j;
            output[k++] = input[idx];
        }
    }
    else if (rank == 1 || rank == 3)
    {
        j = y_min;
        k = 0;
        for(i=x_min; i<= x_max;i++)
        {
            idx = i*(n/2)+j;
            output[k++] = input[idx];
        }

    }
}

int get_row(int n, int for_rank, double* input, double* output)
{
    int i=0,j=0,k=0;
    int idx = 0;
    int x_min,x_max;
    int y_min,y_max;
    int rank = 0;

    if (for_rank == 0)
        rank = 2;
    if (for_rank == 1)
        rank = 3;
    if (for_rank == 2)
        rank = 0;
    if (for_rank == 3)
        rank = 1;

    x_min = 0;
    x_max = (n/2)-1;
    y_min = 0;
    y_max = (n/2)-1;
    if (rank == 0 || rank == 1)
    {
        i = x_max;
        k = 0;
        for(j=y_min; j<= y_max;j++)
        {
            idx = i*(n/2)+j;
            output[k++] = input[idx];
        }
    }
    else if (rank == 2 || rank == 3)
    {
        i = x_min;
        k = 0;
        for(j=y_min; j<= y_max;j++)
        {
            idx = i*(n/2)+j;
            output[k++] = input[idx];
        }

    }

}
int get_corner(int n, int rank, double* input, double* output)
{
    int x_min,x_max;
    int y_min,y_max;


    x_min = 0;
    x_max = (n/2)-1;
    y_min = 0;
    y_max = (n/2)-1;


    /*x - row y column */
    if (rank == 0)
        *output = input[x_max*(n/2) + y_max];
    else if (rank == 1)
        *output = input[x_max*(n/2) + y_min];
    else if(rank == 2)
        *output = input[x_min*(n/2) + y_max];
    else if(rank == 3)
        *output = input[x_min*(n/2) + y_min];
        
}

int combine_quadrants(double* input_rank0, double* input_rank1, double* input_rank2, double* input_rank3, double* output, int n)
{
    int row;
    int start_row;
    int end_row;
    double* output_ptr = output;
    int row_size  = n/2;
    int x_min,x_max;
    int y_min,y_max;

    /* Get the number of rows of rank = 0/1*/
    range(0, n, &x_min, &x_max, &y_min, &y_max);
    start_row = x_min;
    end_row = x_max;

    for (row = start_row; row <= end_row; row++)
    {
        memcpy(output_ptr,input_rank0,sizeof(double)*(row_size));
        output_ptr += row_size;
        memcpy(output_ptr,input_rank1,sizeof(double)*(row_size));
        output_ptr += row_size;
        input_rank0 += row_size;
        input_rank1 += row_size;
    }
    /* Get the number of row of rank = 2/3*/
    range(2, n, &x_min, &x_max, &y_min, &y_max);
    start_row = x_min;
    end_row = x_max;

    for (row = start_row; row <= end_row; row++)
    {
        memcpy(output_ptr, input_rank2,sizeof(double)*(row_size));
        output_ptr += row_size;
        memcpy(output_ptr, input_rank3,sizeof(double)*(row_size));
        output_ptr += row_size;
        input_rank2 += row_size;
        input_rank3 += row_size;
    }
}
/*
 * Quadrant Division among ranks 
 * |--------------------------------------
 * | rows           | c |c|             |
 * |----------------| o |o|             |  
 * | rows           | l |l|             |
 * |----------------| m |m|             |  
 * |  rank0         | n |n| rank1       |
 * ---------------------------------------
 * |                |                   |
 * |                |                   |
 * |  rank2         |      rank3        |
 * |                |                   |
 * |                |                   |
 * |------------------------------------|
 */

int run_mpi(double*u, double* u0, double* u1, double* pebbles, int n, double h, double end_time, int nthreads)
{
    int rank=0;
    int status;
    int x_min,x_max;
    int y_min,y_max;
    int narea_quadrant = (n/2)*(n/2);
    int narea = n*n;
    int column_size = n/2;
    int row_size = n/2;
    int i,j;
    int iter=0;
    int size = n;
    int niter=0;

    double *u_small;
    double *u0_small;
    double *u1_small;
    double *pebbles_small;
    /* The bounday column and row we get from neighbours */
    double* u0_col_recvd;
    double* u0_col_sent;
    double corner_sent;
    double corner_recvd;
    double* u0_row_recvd;
    double* u0_row_sent;
    double t, dt;
    double* array;
    double* u_small1, *u_small2, *u_small3;


    t = 0.;
    dt = h / 2;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    /* create smaller arrays for inputs: u0, u1,pebbles 
     * for each rank. 
     * Divide the data into quadrants */
    u_small = (double*)malloc(sizeof(double) * narea_quadrant);
    u_small1 = (double*)malloc(sizeof(double) * narea_quadrant);
    u_small2 = (double*)malloc(sizeof(double) * narea_quadrant);
    u_small3 = (double*)malloc(sizeof(double) * narea_quadrant);
    u0_small = (double*)malloc(sizeof(double) * narea_quadrant);
    u1_small = (double*)malloc(sizeof(double) * narea_quadrant);
    pebbles_small = (double*)malloc(sizeof(double) * narea_quadrant);

    divide_into_quadrants(n,rank, u0, u0_small);
    divide_into_quadrants(n,rank, u1, u1_small);
    divide_into_quadrants(n,rank, pebbles, pebbles_small);

    /* TODO: CudaCopy*/
    u0_col_recvd = (double*)malloc(sizeof(double) * column_size);
    u0_col_sent = (double*)malloc(sizeof(double) * column_size);
    u0_row_recvd = (double*)malloc(sizeof(double) * column_size);
    u0_row_sent = (double*)malloc(sizeof(double) * column_size);

    range(rank, n, &x_min, &x_max, &y_min, &y_max);
    
#if 0
    if (rank == 0)
    {
        printf("GRID VALUES\n");
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u0[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
        printf("QUDRANT VALUES\n");
        for(i=0; i<n/2;i++)
        {
            for(j=0; j<(n/2);j++)
                printf("%g ",u0_small[i*(n/2)+j]); 
            printf("\n");
        }
        printf("..............................\n");


    }
#endif
    cuda_init(u0_small,  u1_small, pebbles_small, n/2, rank);


    size = n;
    array = u0_small;
    while(1)
    {
        if (rank == 0)
        {
            /* Get right u0_col_recvd */
            MPI_Recv(u0_col_recvd,column_size,MPI_DOUBLE,1,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            get_column(n, 1, array, u0_col_sent);
            MPI_Send(u0_col_sent,column_size,MPI_DOUBLE,1,DEFAULT_TAG,MPI_COMM_WORLD);

            /* get the corner of rank 3*/
            MPI_Recv(&corner_recvd,1,MPI_DOUBLE,3,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG
#endif
            /* send the corner to rank 3 */
            get_corner(size, rank, array, &corner_sent);
            MPI_Send(&corner_sent,1,MPI_DOUBLE,3,DEFAULT_TAG, MPI_COMM_WORLD);
        }
        else if (rank == 1)
        {
            /* Get the boundary column values needed by rank=0 and Send it */ 
            get_column(n, 0, array, u0_col_sent);
//            printf("niter:%d Rank:%d col_sent: %g %g %g %g\n",niter,rank,u0_col_sent[0],u0_col_sent[1],u0_col_sent[2],u0_col_sent[4]);
            MPI_Send(u0_col_sent,column_size,MPI_DOUBLE,0,DEFAULT_TAG,MPI_COMM_WORLD);
            MPI_Recv(u0_col_recvd,column_size,MPI_DOUBLE,0,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* get the corner of rank 2*/
            MPI_Recv(&corner_recvd,1,MPI_DOUBLE,2,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* send the corner to rank 2 */
            get_corner(size, rank, array, &corner_sent);
            MPI_Send(&corner_sent,1,MPI_DOUBLE,2,DEFAULT_TAG, MPI_COMM_WORLD);
        }
        else if (rank == 2)
        {
            /* Get the boundary column values needed by rank=3 and Send it */ 

            get_column(n, 3, array, u0_col_sent);
            MPI_Send(u0_col_sent,column_size,MPI_DOUBLE,3,DEFAULT_TAG, MPI_COMM_WORLD);
            MPI_Recv(u0_col_recvd,column_size,MPI_DOUBLE,3,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* send the corner to rank 1 */
            get_corner(size, rank, array, &corner_sent);
            MPI_Send(&corner_sent,1,MPI_DOUBLE,1,DEFAULT_TAG, MPI_COMM_WORLD);
            /* get the corner of rank 1 */
            MPI_Recv(&corner_recvd,1,MPI_DOUBLE,1,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == 3)
        {
            MPI_Recv(u0_col_recvd,column_size,MPI_DOUBLE,2,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            get_column(n, 2, array, u0_col_sent);
            MPI_Send(u0_col_sent,column_size,MPI_DOUBLE,2,DEFAULT_TAG, MPI_COMM_WORLD);

            /* send the corner to rank 0 */
            get_corner(size, rank, array, &corner_sent);
            MPI_Send(&corner_sent,1,MPI_DOUBLE,0,DEFAULT_TAG, MPI_COMM_WORLD);
            /* get the corner of rank 0 */
            MPI_Recv(&corner_recvd,1,MPI_DOUBLE,0,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        if (rank == 0)
        {
            /* Get right u0_row_recvd */
            MPI_Recv(u0_row_recvd,row_size,MPI_DOUBLE,2,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            get_row(size, 2, array, u0_row_sent);
            MPI_Send(u0_row_sent,row_size,MPI_DOUBLE,2,DEFAULT_TAG,MPI_COMM_WORLD);
            /* Get bottom row */
        }
        else if (rank == 1)
        {
            /* Get the boundary row values needed by rank=0 and Send it */ 
            get_row(size, 3, array, u0_row_sent);
            MPI_Send(u0_row_sent,row_size,MPI_DOUBLE,3,DEFAULT_TAG,MPI_COMM_WORLD);
            MPI_Recv(u0_row_recvd,row_size,MPI_DOUBLE,3,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
        else if (rank == 2)
        {
            /* Get the boundary row values needed by rank=3 and Send it */ 

            get_row(size, 0, array, u0_row_sent);
            MPI_Send(u0_row_sent,row_size,MPI_DOUBLE,0,DEFAULT_TAG, MPI_COMM_WORLD);
            MPI_Recv(u0_row_recvd,row_size,MPI_DOUBLE,0,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
        else if (rank == 3)
        {
            MPI_Recv(u0_row_recvd,row_size,MPI_DOUBLE,1,DEFAULT_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            get_row(size, 1, array, u0_row_sent);
            MPI_Send(u0_row_sent,row_size,MPI_DOUBLE,1,DEFAULT_TAG, MPI_COMM_WORLD);

        }

#ifdef DEBUG
        printf("niter:%d Rank:%d col_recvd: %g %g %g %g\n",niter,rank,u0_col_recvd[0],u0_col_recvd[1],u0_col_recvd[2],u0_col_recvd[3]);
        printf("niter:%d Rank:%d u0_row_recvd: %g %g %g %g\n",niter,rank,u0_row_recvd[0],u0_row_recvd[1],u0_row_recvd[2],u0_row_recvd[3]);
        printf("niter:%d Rank:%d u0_row_sent: %g %g %g %g\n",niter,rank,u0_row_sent[0],u0_row_sent[1],u0_row_sent[2],u0_row_sent[3]);
        printf("Rank: %d corner received:%g\n",rank,corner_recvd);
        printf("Rank: %d corner sent:%g\n",rank,corner_sent);
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        /* Compute each quadrant using CUDA */
        run_gpu_mpi(u_small, u0_small, u1_small, pebbles_small, n/2, h, end_time, nthreads,
                    u0_row_recvd,u0_col_recvd,
                    corner_recvd,
                    rank);

#if 0
        if (rank == 0)
        {
            
            run_gpu_mpi(u_small, u0_small, u1_small, pebbles_small, n/2, h, end_time, nthreads,
                    u0_row_recvd,u0_col_recvd,
                    corner_recvd,
                    rank);
#if 1
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
#endif

        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 1)
        {
            run_gpu_mpi(u_small, u0_small, u1_small, pebbles_small, n/2, h, end_time, nthreads,
                    u0_row_recvd,u0_col_recvd,
                    corner_recvd,
                    rank);
#if 0
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
#endif

        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 2)
        {
            run_gpu_mpi(u_small, u0_small, u1_small, pebbles_small, n/2, h, end_time, nthreads,
                    u0_row_recvd,u0_col_recvd,
                    corner_recvd,
                    rank);
#if 1
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
#endif

        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 3)
        {
            run_gpu_mpi(u_small, u0_small, u1_small, pebbles_small, n/2, h, end_time, nthreads,
                    u0_row_recvd,u0_col_recvd,
                    corner_recvd,
                    rank);
#if 0
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
#endif

        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        memcpy(u1_small,u0_small,sizeof(double)*narea_quadrant);
        memcpy(u0_small,u_small,sizeof(double)*narea_quadrant);
        //u1_small = u0_small;
        //u0_small = u_small;
        array = u_small;

        niter++;
        if(!tpdt(&t, dt, end_time)) break;
    }
    cuda_final(u_small,n); 
#ifdef DEBUG
    if (rank == 0)
    {
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1)
    {
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 2)
    {
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 3)
    {
            printf("Rank:%d\n",rank);
            int k;
            printf("GPU_Iteration:%d\n",iter++);
            for (k =0; k < narea_quadrant;k++)
            {
                if(k%(n/2) == 0)
                    printf("\n");
                printf("%g ",u_small[k]);
            }
            printf("\n...............................");
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Get data from rank 1*/
        MPI_Recv(u_small1,narea_quadrant,MPI_DOUBLE,1,DEFAULT_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        /* Get data from rank 2*/
        MPI_Recv(u_small2,narea_quadrant,MPI_DOUBLE,2,DEFAULT_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        /* Get data from rank 3*/
        MPI_Recv(u_small3,narea_quadrant,MPI_DOUBLE,3,DEFAULT_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Send(u_small,narea_quadrant,MPI_DOUBLE,ROOT,DEFAULT_TAG,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        combine_quadrants(u_small,u_small1,u_small2,u_small3,u,n);
#ifdef DEBUG
        int k;
        for (k =0; k < narea;k++)
        {
            if(k%n == 0)
                printf("\n");
            printf("%g ",u[k]);
        }
        printf("\n...............................");
#endif
    }
    return 0;
}

int main(int argc, char** argv)
{
    int status;
    int rank;

    if(argc != 5)
    {
        printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
        return 0;
    }

    status = MPI_Init( &argc, &argv );
    if (status != MPI_SUCCESS) 
    { 
        printf("Error in MPI INIT");
        return -1;
    }


    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    int     npoints   = atoi(argv[1]);
    int     npebs     = atoi(argv[2]);
    double  end_time  = (double)atof(argv[3]);
    int     nthreads  = atoi(argv[4]);
    int 	  narea	    = npoints * npoints;

    double *u_i0, *u_i1;
    double *u_cpu, *u_mpi, *pebs;
    double h;

    double elapsed_cpu, elapsed_mpi;
    struct timeval cpu_start, cpu_end, mpi_start, mpi_end;

    u_i0 = (double*)malloc(sizeof(double) * narea);
    u_i1 = (double*)malloc(sizeof(double) * narea);
    pebs = (double*)malloc(sizeof(double) * narea);

    u_cpu = (double*)malloc(sizeof(double) * narea);
    u_mpi = (double*)malloc(sizeof(double) * narea);

    printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

    h = (XMAX - XMIN)/npoints;

    init_pebbles(pebs, npebs, npoints);
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);
#if 0
    /* TODO: remove*/
    dummy_init(u_i0, pebs, npoints);
    dummy_init(u_i1, pebs, npoints);
#endif


    print_heatmap("lake_i.dat", u_i0, npoints, h);

    if (rank == ROOT)
    {
    gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);
    print_heatmap("lake_fc.dat", u_cpu, npoints, h);

    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);
    }

    gettimeofday(&mpi_start, NULL);
    run_mpi(u_mpi, u_i0, u_i1, pebs, npoints, h, end_time,nthreads);
    gettimeofday(&mpi_end, NULL);
    elapsed_mpi = ((mpi_end.tv_sec + mpi_end.tv_usec * 1e-6)-(
                mpi_start.tv_sec + mpi_start.tv_usec * 1e-6));
    printf("MPI took %f seconds\n", elapsed_mpi);


    if(rank == 0)
    print_heatmap("lake_fm.dat", u_mpi, npoints, h);

    free(u_i0);
    free(u_i1);
    free(pebs);
    free(u_cpu);
    free(u_mpi);

    MPI_Finalize();
    return 0;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo;
  double t, dt;
  int iter=0;
  int k = 0;

  un = (double*)malloc(sizeof(double) * n * n);
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
#if 0
    printf("\nCPU Iteration:%d\n",iter++);

    printf("CPU UC values\n");
    for (k=0; k < n*n;k++)
    {
        printf("%g ",uc[k]);
        if((k+1)%n == 0)
            printf("\n");
    }
#endif

    evolve9pt(un, uc, uo, pebbles, n, h, dt, t);

#ifdef DEBUG
    printf("CPU output values\n");
    for (k=0; k < n*n;k++)
    {
        printf("%g ",un[k]);
        if((k+1)%n == 0)
            printf("\n");
    }
#endif

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }
  
  memcpy(u, un, sizeof(double) * n * n);
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}
void dummy_init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = idx;
    }
  }
}

void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
          /* TODO: remove this value */
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 
                    uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx+n-1] + uc[idx+n+1] + uc[idx-n-1] + uc[idx-n+1])- 5 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

void print_heatmap(char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");  

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }
  
  fclose(fp);
} 
