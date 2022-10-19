// Implementado por Daniel Ruskov
// SCP2019/20 23/05/2020
// A[M][N]*B[N][P] = R[M][P];
// subA[subM][N]; subR[subM][P]

// Desarrollar un programa MPI para multiplicar matrices de forma paralela. La manera en que se distribuya
// el computo entre los procesadores es libre, pero hay que intentar que sea eficiente. El programa
// recibira como parametros el tamano de las matrices (cuadradas) y utilizaraa reserva dinamica de memoria.
// El proceso root inicializara las matrices con valores aleatorios de tipo float y repartira los datos
// siguiendo el criterio decidido. El codigo se ha de documentar y hay que explicar el metodo seguido por
// el algoritmo implementado.

// MEJORAS
// 1. Disminuir los calculos a la hora de enviar los datos, evitar fors. Eso mediante uso de estructuras y enviar todo de una
// 2. Reservar memoria para la matriz B (que sea codigo comun para todos los procesadores, ya que se envia entera)
// fuera del if/else, pero seria añadir mas for. Posiblemente es mejor duplicar el codigo para los diferentes
// procesadores pero evitar mas loops
// 2.1 Variar l reserv de memoria de forma que sea contigua en cada matriz y acceder mediante A[i*j+i]
// 3. Hacer el codigo de inicializacion de la matriz B comun y evitar el Bcast. Posible en esta version porque no se lee de fichero.
// 4. Pasar el codigo a diferentes funciones llamadas desde el main. No hecho para evitar saltos y porque no es codigo muy largo
// 5. Unir codigo en menos if/else diferenciando rank 0

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Definicion de funcion auxiliar para imprimir una matriz
void imprimir(float **, char, int, int);

// Proceso MAIN
int main(int argc, char** argv) {

	// Comprobacion del numero de argumentos
	if (argc != 4) {
      	fprintf(stderr, "Error de uso.\tUso: %s m n p; Siendo A[m][n]*B[n][p] = R[m][p]\n", argv[0]);
        exit(1);
	}

	// Definicion e inicializacion de variables y punteros
    float **A = NULL, **B = NULL, **R = NULL, **subA = NULL, **subR = NULL;
    int i = 0, j = 0, k = 0, M = atoi(argv[1]), N = atoi(argv[2]), P = atoi(argv[3]), world_rank, world_size, subM, floor, ceil, mod;

    // Inicializa MPI y guarda los valores de tamaño de comunicador y rangos
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Comprueba numero de procesos del comunicador. Aborta si es uno
    if (world_size < 2) {
        fprintf(stderr, "World tiene que ser >= 2\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

	// Comprueba la factibilidad del problema por el programa
	if (M < world_size) {
		fprintf(stderr, "No factible por el numero de filas de A y el numero de procesadores\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
	}

	// Define estructuras a usar en MPI
    MPI_Status info;

	// Inicializacion subM para cada procesador junto a inicializaciones varias
	mod = M % world_size;
	ceil = (M / world_size) +1;
	floor = M / world_size;
	subM = world_rank < mod ? ceil : floor;

    // Reserva memoria e inicializaciones root
    if (world_rank == 0) {
		// Si son matrices cuadradas (evita varios for)
		if (M == N && N == P) {
			// Reserva de memoria para las matrices A, B y R
			A = (float **) malloc(N * sizeof(float *));
			B = (float **) malloc(N * sizeof(float *));
			R = (float **) malloc(N * sizeof(float *));
			for (i = 0; i < N; i++) {
				A[i] = (float *) malloc(N * sizeof(float));
				B[i] = (float *) malloc(N * sizeof(float));
				R[i] = (float *) malloc(N * sizeof(float));
			}

			// Inicializacion de las matrices A y B
			for (i = 0; i < N; i++)
				for (j = 0; j < N; j++) {
					A[i][j] = drand48() * 100.0;
					B[i][j] = drand48() * 100.0;
				}
		} else {
			// Reserva de memoria para las matrices A, B y R
			A = (float **) malloc(M * sizeof(float *));
			B = (float **) malloc(N * sizeof(float *));
			R = (float **) malloc(M * sizeof(float *));
			for (i = 0; i < M; i++) {
				A[i] = (float *) malloc(N * sizeof(float));
				R[i] = (float *) malloc(P * sizeof(float));
			}
			for (i = 0; i < N; i++)
				B[i] = (float *) malloc(P * sizeof(float));

			// Inicializacion de las matrices A y B
			for (i = 0; i < M; i++)
				for (j = 0; j < N; j++)
					A[i][j] = drand48() * 100.0;
			for (i = 0; i < N; i++)
				for (j = 0; j < P; j++)
					B[i][j] = drand48() * 100.0;
		}

        // Imprime las matrices A y B
		imprimir(A, 'A', M, N);
		imprimir(B, 'B', N, P);
	} else { // Reserva memoria resto
		// Reserva de memoria para las submatrices subA y subR
		subA = (float **) malloc(subM * sizeof(float *));
		B = (float **) malloc(N * sizeof(float *));
		subR = (float **) malloc(subM * sizeof(float *));
		for (i = 0; i < N; i++)
			B[i] = (float *) malloc(P * sizeof(float));
		for(i = 0; i < subM; i++) {
			subA[i] = (float *) malloc(N * sizeof(float));
			subR[i] = (float *) malloc(P * sizeof(float));
        }
	}

	// Envio/Recepcion de los trozos de A y entera B desde root a cada procesador correspondiente
	if (world_rank == 0) {
		// Envio filas
		for (i = 1; i < world_size; i++) {
			if (i < mod) {
				for (j = 0; j < ceil; j++)
					MPI_Send(&A[i * ceil + j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			} else {
				for (j = 0; j < floor; j++)
					MPI_Send(&A[mod * ceil + (i - mod) * floor + j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}
			for (j = 0; j < N; j++)
				MPI_Send(&B[j][0], P, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		}
	} else {
		// Recibe filas cada proceso las que le le corresponden
		for (i = 0; i < subM; i++)
			MPI_Recv(&subA[i][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &info);
		for (j = 0; j < N; j++)
			MPI_Recv(&B[j][0], P, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &info);
	}

	// printf("%d: %d %d %d, %d\n", world_rank, M, N, P, subM);
	
/*	if (world_rank == 1) {
		imprimir(B, 'B', N, P);
		imprimir(subA, 's', subM, N);
	}
	MPI_Barrier(MPI_COMM_WORLD);
*/
	// Calculos
	if (world_rank == 0) {
		// Calculo de R parte correspondiente root
		for(i = 0; i < subM; i++) {
			for(j = 0; j < P; j++) {
				R[i][j] = 0.0;
				for(k = 0; k < N; k++)
					R[i][j] += A[i][k] * B[k][j];
			}
		}
	} else {
		// Calculo de la submatriz subR correspondiente a cada proceso no root
		for(i = 0; i < subM; i++) {
			for(j = 0; j < P; j++) {
				subR[i][j] = 0.0;
				for(k = 0; k < N; k++)
					subR[i][j] += subA[i][k] * B[k][j];
			}
		}
	}

	// Envio subR y recepcion por root
	if (world_rank == 0) {
		// Recibe filas
		for (i = 1; i < world_size; i++) {
			if (i < mod)
				for (j = 0; j < ceil; j++)
					MPI_Recv(&R[i * ceil + j][0], P, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &info);
			else
				for (j = 0; j < floor; j++)
					MPI_Recv(&R[mod * ceil + (i - mod) * floor + j][0], P, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &info);	
		}
	} else {
		// Envia filas cada proceso las que le le corresponden
		for (i = 0; i < subM; i++)
			MPI_Send(&subR[i][0], P, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}

	// Imprimir R y liberacion memoria
	if (world_rank == 0) {		
		// Imprime la matriz R
		imprimir(R, 'R', M, P);

		// Liberacion de la memoria
		free(A);
		free(B);
		free(R);
    } else { // Tareas para el resto de procesos		
		//libera espacio 
		free(subA);
        free(B);
        free(subR);
	}
	
	// Cierre de MPI
	MPI_Finalize();

	// Finalizacion correcta del programa
	exit(0);
}

// Implementacion de funcion auxiliar para imprimir una matriz
void imprimir(float **M, char nombre, int A, int B) {
	int i, j;
	printf("%c", nombre);
	for (i = 0; i < A; i++) {
		printf("\n");
		for (j = 0; j < B; j++)
			printf("[%.3f]", M[i][j]);
	}
	printf("\n");
}

//https://whereby.com/jose-ehu