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

#define __DEBUG

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

__global__ void evolve_kernal(double *un, double *uc, double *uo, double *pebbles, double h, double dt,double  t)

{
      int threads_per_block = blockDim.x * blockDim.y;
      int idx = threadIdx.x + blockIdx.x*blockDim.x + threadIdx.y*blockDim.x*gridDim.x + blockIdx.y*(threads_per_block)*gridDim.x;
      int n = gridDim.x * blockDim.x ;
      float f = -expf(-1.0 * t) * pebbles[idx];

      if( (blockIdx.x == 0 && threadIdx.x == 0) || 
          (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) || 
          (blockIdx.y == 0 && threadIdx.y == 0) || 
          (blockIdx.y == gridDim.y - 1 && threadIdx.y == blockDim.y - 1))
      {
        un[idx] = 0.;
      }
      else
      {
          /* un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *(( WEST + EAST +
                                          NORTH + SOUTH + 0.25*(NORTHWEST + NORTHEAST + SOUTHWEST + SOUTHEAST)- 5 * uc[idx])/(h * h) + f(pebbles[idx],t)); */
        un[idx] = 2*uc[idx] - uo[idx] + 0.1 *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx+n-1] + uc[idx+n+1] + uc[idx-n-1] + uc[idx-n+1])- 5 * uc[idx])/(h * h) + f);
      }
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
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
        double *u_dev,*u0_dev,*u1_dev,*pebbles_dev;

        /*Allocate memory for device variables*/
        CUDA_CALL(cudaMalloc((void **)&u_dev,sizeof(double)*n*n));
        CUDA_CALL(cudaMalloc((void **)&u0_dev,sizeof(double)*n*n));
        CUDA_CALL(cudaMalloc((void **)&u1_dev,sizeof(double)*n*n));
        CUDA_CALL(cudaMalloc((void **)&pebbles_dev,sizeof(double)*n*n));
        
        //Setting grid and block dimensions
        dim3 grid(n/nthreads,n/nthreads);
        dim3 block(nthreads,nthreads);

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));
        
        //Transferring data from host to device memory
        CUDA_CALL(cudaMemcpy(u0_dev,u0,sizeof(double)*n*n,cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(u1_dev,u1,sizeof(double)*n*n,cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(pebbles_dev,pebbles,sizeof(double)*n*n,cudaMemcpyHostToDevice));

       
	/* HW2: Add main lake simulation loop here */
	
        while(1)
        {
          evolve_kernal<<<grid,block>>>(u_dev, u1_dev, u0_dev, pebbles_dev, h, dt, t);
          
          CUDA_CALL(cudaMemcpy(u0_dev,u1_dev,sizeof(double)*n*n,cudaMemcpyDeviceToDevice));
          CUDA_CALL(cudaMemcpy(u1_dev,u_dev,sizeof(double)*n*n,cudaMemcpyDeviceToDevice));

          if(!tpdt(&t, dt, end_time)) break;
        }
    CUDA_CALL(cudaMemcpy(u,u_dev,sizeof(double)*n*n,cudaMemcpyDeviceToHost));
        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}

