// El proceso 0 de un programa MPI tiene que enviar al resto de procesos un array
// de enteros E de longitud igual al numero de procesos que lo componen, un entero
// que representa esa longitud y un array de floats F del mismo tamaño. Cada proceso
// tiene que realizar la operación res = sumatorio [E(i) * F (i)] y el proceso 0
// tiene que imprimir la suma de todos los resultados parciales (res). Escribe el
// código en MPI necesario para realizar el proceso anterior.

//Objetivo: aprender a crear una nueva estructura y manejarla.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Define funcion auxiliar para crear la estructura PARAM
void Crear_Tipo (int *, int *, float *, MPI_Datatype *, int, int);

// Proceso MAIN
int main(int argc, char** argv) {

	// Definicion de variables y punteros
	float res = 0.0, resTot = 0.0, *F;
	int i, world_rank, world_size, *E, I;

	// Inicializa MPI y guarda los valores de tamaño de comunicador y rangos
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Comprueba numero de procesos del comunicador. Aborta si es uno
	if (world_size < 2) {
		fprintf(stderr, "World tiene que ser >= 2%s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
  	}

	// Define estructuras a usar en MPI. Reserva de memoria para los arrays E y F. Crea el tipo PARAM
  	MPI_Status info;
	MPI_Datatype PARAM;
	I = world_size;
	E = malloc(I*sizeof(int));
	F = malloc(I*sizeof(float));
	Crear_Tipo(E, &I, F, &PARAM, I, I);

	// Inicializacion de los arrays E y F
	if (world_rank == 0){
  		for(i=0; i<I; i++) {
			E[i]=rand() % 10;
			F[i]=drand48() * 10.0;
		}
		printf("\nI = %d\n", I);
		printf("E = [\t");
		for(i=0; i<I; i++)
			printf("%d\t\t", E[i]);
		printf("]\n");
		printf("F = [\t");
		for(i=0; i<I; i++)
			printf("%f\t", F[i]);
		printf("]\n");
	}

	// Broadcast de la estructura PARAM
	MPI_Bcast(E, 1, PARAM, 0, MPI_COMM_WORLD);

	// Calculo del elemento correspondiente a cada proceso
	// Nota: si se tratara de que cada proceso calcule mas de un elemento habria
	// habria que hacer primero un broadcast con el numero de elementos a tratar
	// y despues hacer un for para hacer las operaciones para guardarlos en la 
	// variable parcial para cada proceso res
	res = (float)E[world_rank] * F[world_rank];

	// Reduccion de los sumatorios parciales a la suma total para P0
	MPI_Reduce(&res, &resTot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	// Imprimir el resultado total por la SE
	if (world_rank == 0)
		printf("\n\tSumatorio[ E(i) ∗ F (i) ] = %f\n\n", resTot);

	// Liberacion de memoria y cierre de MPI
	free(E);
	free(F);
	MPI_Finalize();
}

// Implementacion de la funcion auxiliar para la creacion de la estructura PARAM
void Crear_Tipo (int *E, int *I, float *F, MPI_Datatype *PARAM, int tamE, int tamF){

	// Array con el tmaño de los elementos a añadir en PARAM
	int tam[3];
	tam [0] = tamE;
	tam [1] = 1;
	tam [2] = tamF;

	// Array con el tipo de los elementos a añadir en PARAM
	MPI_Datatype tipo [3];
	tipo [0] = MPI_INT;
	tipo [1] = MPI_INT;
	tipo [2] = MPI_FLOAT;

	// Array con las direcciones desde el inicio de la estructura para cada elemento
	MPI_Aint dist[3] , dir1 , dir2, dir3;
	MPI_Get_address (E, &dir1);
	MPI_Get_address (I, &dir2);
	MPI_Get_address (F, &dir3);
	dist [0] = 0;
	dist [1] = dir2 - dir1;
	dist [2] = dir3 - dir1;

	// Creacion de la estructura y commit
	MPI_Type_create_struct (3, tam, dist, tipo, PARAM);
	MPI_Type_commit (PARAM);
}
