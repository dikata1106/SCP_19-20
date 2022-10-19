// Escribe un programa MPI que envı́e los elementos superiores 
// (sin incluir los de la diagonal principal) de una matriz de
// dimensiones arbitrarias (generada e inicializada de forma 
// aleatoria) al resto de procesos

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16
#define M 16

int main(int argc, char** argv) {

	int A[N][M], T[N][M], tam[N], dis[N];
	int i, j, world_rank, world_size;

	for (i=0; i<N; i++) {
		tam[i] = M - i -1;
		dis[i] = ((M+1) * i) +1;
	}

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size < 2) {
		fprintf(stderr, "World tiene que ser >= 2%s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
  	}

  	MPI_Status info;
	MPI_Datatype MTRI;
	MPI_Type_indexed (N, tam, dis, MPI_INT, &MTRI);
	MPI_Type_commit (&MTRI);

	if (world_rank == 0) {
		for (i=0; i<N; i++)
			for (j=0; j<M; j++)
        			A[i][j]=rand()%100;
		printf("%d %d %d\n", A[0][1], A[0][15], A[14][15]);
		MPI_Send (A, 1, MTRI, 1, 0, MPI_COMM_WORLD);
	} else if (world_rank == 1) {
		MPI_Recv (T, 1, MTRI, 0, 0, MPI_COMM_WORLD, &info);
		printf("%d %d %d\n", T[0][1], T[0][15], T[14][15]);
	}

	MPI_Finalize();

}
