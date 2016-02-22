/* 
Single Author info:
prajago4 Priyadarshini Rajagopal
Group info:
1. avelayu Ashitha Velayudhan
2. prajago4 Priyadarshini Rajagopal
3. smnatara Sekharan Muthusamy Natarajan
*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

extern int tpdt(double *t, double dt, double end_time);
/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}
enum TYPE {
    TOP_THREAD =1,
    BOT_THREAD,
    LEFT_THREAD,
    RIGHT_THREAD,
    TOP_BLOCK,
    BOT_BLOCK,
    LEFT_BLOCK,
    RIGHT_BLOCK
};
__global__ void evolve_kernal(double *un, double *uc, double *uo, double *pebbles, double h, double dt,double  t,
                              double* uc_row, double* uc_col, double* corner_ptr, int * rank_ptr)

{
    int top_thread = 0,bottom_thread=0,left_thread=0,right_thread=0;
    int top_block = 0,bottom_block=0,left_block=0,right_block=0;
    double corner = *corner_ptr;
    int rank = *rank_ptr;
    double west,south,east,north;
    double north_west, north_east, south_west, south_east;

    int threads_per_block = blockDim.x * blockDim.y;
    int idx = threadIdx.x + blockIdx.x*blockDim.x + threadIdx.y*blockDim.x*gridDim.x + blockIdx.y*(threads_per_block)*gridDim.x;
    int n = gridDim.x * blockDim.x ;
    float f = -expf(-1.0 * t) * pebbles[idx];

    if (threadIdx.x == 0)
        left_thread = 1;
    if (threadIdx.x == blockDim.x-1)
        right_thread = 1;
    if (threadIdx.y == 0)
        top_thread = 1;
    if (threadIdx.y == blockDim.y-1)
        bottom_thread = 1;

    if (blockIdx.x == 0)
        left_block = 1;
    if (blockIdx.x == gridDim.x-1)
        right_block = 1;
    if (blockIdx.y == 0)
        top_block = 1;
    if (blockIdx.y == gridDim.y-1)
        bottom_block = 1;

    if ((top_thread && top_block && left_thread && left_block))
    {
#if 0
        int i,j;
        printf("GPU RANK:%d GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",uc[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
#endif
    }

    if (rank == 0)
    {
        if ((top_block && top_thread) || (left_block && left_thread))
        {
            /* Border*/
            un[idx] = 0.;
        }
        else
        {
            if (bottom_block && bottom_thread && right_block && right_thread)
            {
                /* Corner that is not a border */
                north = uc[idx - n];
                south = uc_row[blockIdx.x * blockDim.x + threadIdx.x];
                east = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                west = uc[idx-1];
                north_east = uc_col[blockIdx.y * blockDim.y + threadIdx.y - 1];
                north_west = uc[idx-n-1];
                south_east = corner;
                south_west = uc_row[blockIdx.x * blockDim.x + threadIdx.x -1];
            }
            else if(right_thread && right_block)
            {
                /* Right most values*/
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                west = uc[idx-1];
                north_east = uc_col[blockIdx.y * blockDim.y + threadIdx.y-1];
                north_west = uc[idx-n-1];
                south_east = uc_col[blockIdx.y * blockDim.y + threadIdx.y + 1];
                south_west = uc[idx+n-1];

            }
            else if (bottom_thread && bottom_block)
            {   
                /* Bottom values */
                north = uc[idx - n];
                south = uc_row[blockIdx.x * blockDim.x + threadIdx.x];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc[idx-n+1];
                north_west = uc[idx-n-1];
                south_east = uc_row[blockIdx.x * blockDim.x + threadIdx.x + 1];
                south_west = uc_row[blockIdx.x * blockDim.x + threadIdx.x - 1];
            }
            else
            {
                /* Non-border middle threads*/
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc[idx-n+1];
                north_west = uc[idx-n-1];
                south_east = uc[idx+n+1];
                south_west = uc[idx+n-1];
            }
            un[idx] = 2*uc[idx] - uo[idx] + 0.1 *(dt * dt) *((west + east + north + south + 
                    0.25 * (south_west + south_east + north_east + north_west)- 5 * uc[idx])/(h * h) + f);
            //printf("0:0:  west:%g east:%g north:%g south:%g final:%g\n",west,east,north,south,un[idx]);

        }
    }
    else if (rank == 1)
    {
        if ((top_block && top_thread) || (right_block && right_thread))
        {
            /* Border elements */
            un[idx] = 0.;
        }
        else {
            if (bottom_block && bottom_thread && left_block && left_thread)
            {
                /* Corner that is not a border */

                north = uc[idx-n];
                south = uc_row[blockIdx.x * blockDim.x + threadIdx.x ];
                east = uc[idx + 1];
                west = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                north_east = uc[idx-n+1];
                north_west = uc_col[blockIdx.y * blockDim.y + threadIdx.y -1];
                south_east = uc_row[blockIdx.x * blockDim.x + threadIdx.x + 1];
                south_west = corner;

            }
            else if(left_thread && left_block)
            {
                /*left most threads  */
                north = uc[idx-n];
                south = uc[idx+n];
                east = uc[idx + 1];
                west = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                north_east = uc[idx-n+1];
                north_west = uc_col[blockIdx.y * blockDim.y + threadIdx.y -1];
                south_east = uc[idx+n+1];
                south_west = uc_col[blockIdx.y * blockDim.y + threadIdx.y + 1];
            }
            else if (bottom_thread && bottom_block)
            {
                /* Bottom most threads */
                north = uc[idx-n];
                south = uc_row[blockIdx.x * blockDim.x + threadIdx.x];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc[idx-n+1];
                north_west = uc[idx-n-1];
                south_east = uc_row[blockIdx.x * blockDim.x + threadIdx.x + 1];
                south_west = uc_row[blockIdx.x * blockDim.x + threadIdx.x - 1];
            }
            else
            {
                /* Non- border values.. midddle threads*/
                /* Non-border middle threads*/
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc[idx-n+1];
                north_west = uc[idx-n-1];
                south_east = uc[idx+n+1];
                south_west = uc[idx+n-1];
            }
            un[idx] = 2*uc[idx] - uo[idx] + 0.1 *(dt * dt) *((west + east + north + south + 
                    0.25 * (south_west + south_east + north_east + north_west)- 5 * uc[idx])/(h * h) + f);
            //printf("1: west:%g east:%g north:%g south:%g final:%g\n",west,east,north,south,un[idx]);
        }
    }
    else if (rank == 2)
    {
        if ((left_thread && left_block) ||(bottom_thread && bottom_block))
        {
            /* Border */
            un[idx] = 0.;
        }
        else
        {
            if (top_thread && top_block && right_thread && right_block)
            {
                /* Corner that is not a border */
                north = uc_row[blockIdx.x * blockDim.x + threadIdx.x];
                south = uc[idx + n];
                east = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                west = uc[idx - 1];
                north_east = corner;
                north_west = uc_row[blockIdx.x * blockDim.x + threadIdx.x -1];
                south_east = uc_col[blockIdx.y * blockDim.y + threadIdx.y + 1];
                south_west = uc[idx+n-1];
            }
            else if(right_thread && right_block)
            {
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                west = uc[idx - 1];
                north_east = uc_col[blockIdx.y * blockDim.y + threadIdx.y - 1];
                north_west = uc[idx-n-1];
                south_east = uc_col[blockIdx.y * blockDim.y + threadIdx.y + 1];
                south_west = uc[idx+n-1];
            }
            else if(top_thread && top_block)
            {
                north = uc_row[blockIdx.x * blockDim.x + threadIdx.x ];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc_row[blockIdx.x * blockDim.x + threadIdx.x +1];
                north_west = uc_row[blockIdx.x * blockDim.x + threadIdx.x -1];
                south_east = uc[idx + n +1];
                south_west = uc[idx + n -1];
            }
            else
            {
                /* Non-border middle threads*/
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc[idx-n+1];
                north_west = uc[idx-n-1];
                south_east = uc[idx+n+1];
                south_west = uc[idx+n-1];
            }
            un[idx] = 2*uc[idx] - uo[idx] + 0.1 *(dt * dt) *((west + east + north + south + 
                    0.25 * (south_west + south_east + north_east + north_west)- 5 * uc[idx])/(h * h) + f);
            //printf("2: west:%g east:%g north:%g south:%g final:%g\n",west,east,north,south,un[idx]);
        }
    }
    else if (rank == 3)
    {
        if ((right_thread && right_block) || (bottom_thread && bottom_block))
        {
            /* Border */
            un[idx] = 0.;
        }
        else
        {
            if (left_thread && left_block && top_thread && top_block)
            {
                /* Corner that is not a border */
                north = uc_row[blockIdx.x * blockDim.x + threadIdx.x ];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                north_east = uc_row[blockIdx.x * blockDim.x + threadIdx.x +1];
                north_west = corner;
                south_east = uc[idx+n+1];
                south_west = uc_col[blockIdx.y * blockDim.y + threadIdx.y +1];
            }
            else if (left_block && left_thread)
            {
                /* left most elements */
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc_col[blockIdx.y * blockDim.y + threadIdx.y];
                north_east = uc[idx-n+1];
                north_west = uc_col[blockIdx.y * blockDim.y + threadIdx.y -1];
                south_east = uc[idx+n+1];
                south_west = uc_col[blockIdx.y * blockDim.y + threadIdx.y +1];
            }
            else if (top_thread && top_block)
            {
                north = uc_row[blockIdx.x * blockDim.x + threadIdx.x ];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc_row[blockIdx.x * blockDim.x + threadIdx.x +1];
                north_west = uc_row[blockIdx.x * blockDim.x + threadIdx.x -1];
                south_east = uc[idx+n+1];
                south_west = uc[idx+n-1];
            }
            else
            {
                /* Non-border middle threads*/
                north = uc[idx - n];
                south = uc[idx + n];
                east = uc[idx + 1];
                west = uc[idx - 1];
                north_east = uc[idx-n+1];
                north_west = uc[idx-n-1];
                south_east = uc[idx+n+1];
                south_west = uc[idx+n-1];
            }
            un[idx] = 2*uc[idx] - uo[idx] + 0.1 *(dt * dt) *((west + east + north + south + 
                    0.25 * (south_west + south_east + north_east + north_west)- 5 * uc[idx])/(h * h) + f);
            //printf("3: west:%g east:%g north:%g south:%g final:%g\n",west,east,north,south,un[idx]);
        }


    }

}


double *u_dev,*u0_dev,*u1_dev,*pebbles_dev;
double* u0_row_recvd_dev, *u0_col_recvd_dev;
double *corner_dev;
int*rank_dev;
extern "C" int cuda_init(double* u0, double* u1, double* pebbles, int n, int rank )

{

    /*Allocate memory for device variables*/
    CUDA_CALL(cudaMalloc((void **)&u_dev,sizeof(double)*n*n));
    CUDA_CALL(cudaMalloc((void **)&u0_dev,sizeof(double)*n*n));
    CUDA_CALL(cudaMalloc((void **)&u1_dev,sizeof(double)*n*n));
    CUDA_CALL(cudaMalloc((void **)&pebbles_dev,sizeof(double)*n*n));

    CUDA_CALL(cudaMalloc((void **)&u0_row_recvd_dev,sizeof(double)*n));
    CUDA_CALL(cudaMalloc((void **)&u0_col_recvd_dev,sizeof(double)*n));
    CUDA_CALL(cudaMalloc((void **)&corner_dev,sizeof(double)*1));
    CUDA_CALL(cudaMalloc((void **)&rank_dev,sizeof(int)*1));

    //Transferring data from host to device memory
    CUDA_CALL(cudaMemcpy(u0_dev,u0,sizeof(double)*n*n,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(u1_dev,u1,sizeof(double)*n*n,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pebbles_dev,pebbles,sizeof(double)*n*n,cudaMemcpyHostToDevice));
}
extern "C" int cuda_final(double* u, int n)
{
    CUDA_CALL(cudaMemcpy(u,u_dev,sizeof(double)*n*n,cudaMemcpyDeviceToHost));
}

extern "C" int run_gpu_mpi(double*u, double* u0, double* u1, double* pebbles, int n, double h, double end_time, int nthreads,
                   double* u0_row_recvd,double* u0_col_recvd, double corner, int rank )
{
    cudaEvent_t kstart, kstop;
    float ktime;

    /* HW2: Define your local variables here */
    double t, dt;
    t = 0.;
    dt = h / 2;


    /* Set up device timers */  
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaEventCreate(&kstart));
    CUDA_CALL(cudaEventCreate(&kstop));

    /* HW2: Add CUDA kernel call preperation code here */

    /*Variables on the device:*/



    //Setting grid and block dimensions
    dim3 grid(n/nthreads,n/nthreads);
    dim3 block(nthreads,nthreads);

    /* Start GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstart, 0));


    CUDA_CALL(cudaMemcpy(u0_row_recvd_dev,u0_row_recvd,sizeof(double)*n,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(u0_col_recvd_dev,u0_col_recvd,sizeof(double)*n,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(corner_dev, &corner,sizeof(double)*1,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(rank_dev, &rank,sizeof(int)*1,cudaMemcpyHostToDevice));
#if 0
      int k;
      if (rank == 1)
      {
          printf("%d GPU:UC_ROW:\n",rank);
          for(k=0; k< n;k++)
              printf("%g ",u0_row_recvd[k]);
          printf("\n");
          printf("%d GPU: UC_COL:\n",rank);
          for(k=0; k< n;k++)
              printf("%g ",u0_col_recvd[k]);
      }
#endif

    /* HW2: Add main lake simulation loop here */
#if 0
    while(1)
    {
        evolve_kernal<<<grid,block>>>(u_dev, u1_dev, u0_dev, pebbles_dev, h, dt, t,
                    u0_row_recvd_dev, u0_col_recvd_dev,rank);

        CUDA_CALL(cudaMemcpy(u0_dev,u1_dev,sizeof(double)*n*n,cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaMemcpy(u1_dev,u_dev,sizeof(double)*n*n,cudaMemcpyDeviceToDevice));

        if(!tpdt(&t, dt, end_time)) break;
    }
#endif
#ifdef DEBUG
    if (rank == 0)
    {
        int i,j;
        printf("RANK:%d BEFORE GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u1[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
    }
    if (rank == 2)
    {
        int i,j;
        printf("RANK:%d BEFORE GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u1[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
    }
#endif



    evolve_kernal<<<grid,block>>>(u_dev, u1_dev, u0_dev, pebbles_dev, h, dt, t,
            u0_row_recvd_dev, u0_col_recvd_dev, corner_dev,rank_dev);

    CUDA_CALL(cudaMemcpy(u0_dev,u1_dev,sizeof(double)*n*n,cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(u1_dev,u_dev,sizeof(double)*n*n,cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(u,u_dev,sizeof(double)*n*n,cudaMemcpyDeviceToHost));


#ifdef DEBUG
    if (rank == 0)
    {
        int i,j;
        printf("RANK:%d AFTER GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
    }
    if (rank == 1)
    {
        int i,j;
        printf("RANK:%d AFTER GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
    }
    if (rank == 2)
    {
        int i,j;
        printf("RANK:%d AFTER GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
    }
    if (rank == 2)
    {
        int i,j;
        printf("RANK:%d AFTER GRID VALUES\n",rank);
        for(i=0; i<n;i++)
        {
            for(j=0; j<n;j++)
                printf("%g ",u[i*n+j]); 
            printf("\n");
        }
        printf("..............................\n");
    }
#endif



    /* Stop GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstop, 0));
    CUDA_CALL(cudaEventSynchronize(kstop));
    CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
#ifdef DEBUG
    printf("GPU computation: %f msec\n", ktime);
#endif

    /* HW2: Add post CUDA kernel call processing and cleanup here */

    /* timer cleanup */
    CUDA_CALL(cudaEventDestroy(kstart));
    CUDA_CALL(cudaEventDestroy(kstop));
}

