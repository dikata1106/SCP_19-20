// Implementado por Daniel Ruskov
// SCP2019/20 
// R[n][n] = A[n][n]*B[n][n];

// Desarrollar un programa MPI para multiplicar matrices de forma paralela. La manera en que se distribuya
// el computo entre los procesadores es libre, pero hay que intentar que sea eficiente. El programa
// recibira como parametros el tamano de las matrices (cuadradas) y utilizaraa reserva dinamica de memoria.
// El proceso root inicializara las matrices con valores aleatorios de tipo float y repartira los datos
// siguiendo el criterio decidido. El codigo se ha de documentar y hay que explicar el metodo seguido por
// el algoritmo implementado.



#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Implementacion de funcion auxiliar para imprimir una matriz
void imprimir(float **M, char nombre, int N) {
	int i, j;
	printf("\n%c", nombre);
        for(i = 0; i < N; i++){
        	printf("\n");
                for(j = 0; j < N; j++)
                        printf("[%.3f]", M[i][j]);
        }
}

// Proceso MAIN
int main(int argc, char** argv) {

	// Comprobacion del numero de argumentos
/*	if (argc != 2) {
*        	fprintf(stderr, "Error de uso.\nUso: ./%s <parametro N (N%4=0)>\n", argv[0]);
*                exit(1);
*        }
*/
        // Por facilidad, se aceptaran solo matrices cuadradas de NxN elementos, 
        // siendo N multiplo de 4 para repartir la matriz entre 16 procesos sin sobrantes
        // Para aceptar cualquier N habria que hacer cambios en el codigo para reparto previo.
        if (atoi(argv[1]) % 4 != 0) {
        	fprintf(stderr, "Solo se acepta n multiplo de 4\n");
                exit(2);
        }

	// Definicion e inicializacion de variables y punteros
        float **A = NULL, **B = NULL, **R = NULL, **subA = NULL, **subB = NULL, **subR = NULL;
        int i = 0, j = 0, k = 0, N = atoi(argv[1]), subN = N/4, world_rank, world_size;

        // Inicializa MPI y guarda los valores de tamaño de comunicador y rangos
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Comprueba numero de procesos del comunicador. Aborta si es uno
        if (world_size < 2) {
                fprintf(stderr, "World tiene que ser >= 2%s\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 3);
        }

	// Define estructuras a usar en MPI
        MPI_Status info;
	MPI_Datatype columna;
	MPI_Type_vector(N, subN, N-subN, MPI_INT, &columna);
	MPI_Type_commit(&columna);
	MPI_Datatype subMatriz;
	MPI_Type_vector(subN, subN, N-subN, MPI_INT, &subMatriz);
	MPI_Type_commit(&subMatriz);

        // Tareas para el proceso root
        if (world_rank == 0){
                // Reserva de memoria para las matrices A, B y R con control de errores
                A = (float **) malloc(N*sizeof(float *));
                B = (float **) malloc(N*sizeof(float *));
                R = (float **) malloc(N*sizeof(float *));
                for(i = 0; i<N; i++) {
                        A[i] = (float *) malloc(N*sizeof(float));
                        B[i] = (float *) malloc(N*sizeof(float));
                        R[i] = (float *) malloc(N*sizeof(float));
                }

                // Inicializacion de las matrices A y B
                for(i = 0; i < N; i++)
                        for(j = 0; j < N; j++) {
                                A[i][j] = drand48()*100.0;
                                B[i][j] = drand48()*100.0;
                        }

                // Imprime las matrices A y B
                printf("\nA");
                for(i = 0; i < N; i++){
                	printf("\n");
                        for(j = 0; j < N; j++)
                                printf("[%.3f]", A[i][j]);
                }
                printf("\nB");
                for(i = 0; i < N; i++){
                	printf("\n");
                        for(j = 0; j < N; j++)
                                printf("[%.3f]", B[i][j]);
                }
                
                //envio de filas y columnas correspondientes
                MPI_Send(&A[0][0], N*subN, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][subN], 1, columna, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&A[0][0], N*subN, MPI_INT, 2, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][2*subN], 1, columna, 2, 0, MPI_COMM_WORLD);
		MPI_Send(&A[0][0], N*subN, MPI_INT, 3, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][3*subN], 1, columna, 3, 0, MPI_COMM_WORLD);
		MPI_Send(&A[subN][0], N*subN, MPI_INT, 4, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][0], 1, columna, 4, 0, MPI_COMM_WORLD);
		MPI_Send(&A[subN][0], N*subN, MPI_INT, 5, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][subN], 1, columna, 5, 0, MPI_COMM_WORLD);
		MPI_Send(&A[subN][0], N*subN, MPI_INT, 6, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][2*subN], 1, columna, 6, 0, MPI_COMM_WORLD);
		MPI_Send(&A[subN][0], N*subN, MPI_INT, 7, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][3*subN], 1, columna, 7, 0, MPI_COMM_WORLD);
		MPI_Send(&A[2*subN][0], N*subN, MPI_INT, 8, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][0], 1, columna, 8, 0, MPI_COMM_WORLD);
		MPI_Send(&A[2*subN][0], N*subN, MPI_INT, 9, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][subN], 1, columna, 9, 0, MPI_COMM_WORLD);
		MPI_Send(&A[2*subN][0], N*subN, MPI_INT, 10, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][2*subN], 1, columna, 10, 0, MPI_COMM_WORLD);
		MPI_Send(&A[2*subN][0], N*subN, MPI_INT, 11, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][3*subN], 1, columna, 11, 0, MPI_COMM_WORLD);
		MPI_Send(&A[3*subN][0], N*subN, MPI_INT, 12, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][0], 1, columna, 12, 0, MPI_COMM_WORLD);
		MPI_Send(&A[3*subN][0], N*subN, MPI_INT, 13, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][subN], 1, columna, 13, 0, MPI_COMM_WORLD);
		MPI_Send(&A[3*subN][0], N*subN, MPI_INT, 14, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][2*subN], 1, columna, 14, 0, MPI_COMM_WORLD);
		MPI_Send(&A[3*subN][0], N*subN, MPI_INT, 15, 0, MPI_COMM_WORLD);
		MPI_Send(&B[0][3*subN], 1, columna, 15, 0, MPI_COMM_WORLD);

		// Calculo de R parte correspondiente a rank0
		for(i = 0; i < subN; i++)
			for(j = 0; j < subN; j++) {
				R[i][j] = 0.0;
				for(k = 0; k < N; k++)
					R[i][j] += A[i][k]*B[k][j];
			}
			
		//recibe las subR en orden guardandolas en la matriz R
		MPI_Recv(&R[0][subN], 1, subMatriz, 1, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[0][2*subN], 1, subMatriz, 2, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[0][3*subN], 1, subMatriz, 3, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[subN][0], 1, subMatriz, 4, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[subN][subN], 1, subMatriz, 5, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[subN][2*subN], 1, subMatriz, 6, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[subN][3*subN], 1, subMatriz, 7, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[2*subN][0], 1, subMatriz, 8, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[2*subN][subN], 1, subMatriz, 9, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[2*subN][2*subN], 1, subMatriz, 10, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[2*subN][3*subN], 1, subMatriz, 11, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[3*subN][0], 1, subMatriz, 12, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[3*subN][subN], 1, subMatriz, 13, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[3*subN][2*subN], 1, subMatriz, 14, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&R[3*subN][3*subN], 1, subMatriz, 15, 0, MPI_COMM_WORLD, &info);
			
		// Imprime la matriz R
                printf("\nR");
                for(i = 0; i < N; i++){
                	printf("\n");
                        for(j = 0; j < N; j++)
                                printf("[%.3f]", R[i][j]);
                }

                // Liberacion de la memoria
                free(A);
                free(B);
                free(R);
        } else { // Tareas para el resto de procesos
		// Reserva de memoria para las submatrices subA, subB y subR con control de errores
                subA = (float **) malloc(subN*sizeof(float *));
                subB = (float **) malloc(N*sizeof(float *));
                subR = (float **) malloc(subN*sizeof(float *));
		for(i = 0; i<N; i++)
                        subB[i] = (float *) malloc(subN*sizeof(float));
                for(i = 0; i<subN; i++) {
                        subA[i] = (float *) malloc(N*sizeof(float));
                        subR[i] = (float *) malloc(N*sizeof(float));
                }

		// Recibe filas y columnas cada proceso las que le le corresponden
		MPI_Recv(&subA[0][0], N*subN, MPI_INT, 0, 0, MPI_COMM_WORLD, &info);
		MPI_Recv(&subB[0][0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &info);

		// Calculo de la submatriz subR correspondiente a cada proceso
		for(i = 0; i < subN; i++)
			for(j = 0; j < subN; j++) {
				subR[i][j] = 0.0;
				for(k = 0; k < N; k++)
					R[i][j] += subA[i][k]*subB[k][j];
			}
			
		// devuelve subR a root
		MPI_Send(&subR[0][0], subN*subN, MPI_INT, 0, 0, MPI_COMM_WORLD);
		
		//libera espacio 
		free(subA);
                free(subB);
                free(subR);
	}
	
        // Cierre de MPI
        MPI_Finalize();

        // Finalizacion correcta del programa
        exit(0);
}
