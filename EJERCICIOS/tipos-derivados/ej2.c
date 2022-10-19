// Escribe un programa MPI que envı́e una columna (seleccionada de forma aleatoria) 
// de una matriz de dimensiones 16x16 (inicializada de forma aleatoria) al resto de procesos
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16
#define M 16

int main(int argc, char** argv) {

	int A[N][M];
	int i, j, world_rank, world_size;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size < 2) {
		fprintf(stderr, "World tiene que ser >= 2%s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
  	}

  	MPI_Status info;
	MPI_Datatype columna;
	MPI_Type_vector (N, 1, M, MPI_INT, &columna);
	MPI_Type_commit (&columna);

	if (world_rank == 0) {
		for (i=0; i<N; i++)
			for (j=0; j<M; j++)
        			A[i][j]=rand()%100;
		printf("%d %d\n", A[0][3], A[15][3]);
		MPI_Send (&A[0][3], 1, columna, 1, 0, MPI_COMM_WORLD);
	} else if (world_rank == 1) {
		MPI_Recv (&A[0][3] ,1, columna, 0, 0, MPI_COMM_WORLD, &info);
		printf("%d %d\n", A[0][3], A[15][3]);
	}

	MPI_Finalize();

}
