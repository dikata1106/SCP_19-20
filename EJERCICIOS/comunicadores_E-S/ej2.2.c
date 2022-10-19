// Implementado por Daniel Ruskov
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

// Definicion de funciones auxiliares
void imprimir(float **, char, int);

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
        int world_rank, world_size, pid_f0, pid_f1, pid_f2, pid_f3, pid_c0, pid_c1, pid_c2, pid_c3, n = 4;
        int i = 0, j = 0, k = 0, N = atoi(argv[1]), subN = N/n;
	int procf0[n], procf1[n], procf2[n], procf3[n], procc0[n], procc1[n], procc2[n], procc3[n];

        // Inicializa MPI
        MPI_Init(NULL, NULL);
        
        // Guarda los valores de tamaño de COMM_WORLD y sus rangos
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        // Comprueba numero de procesos del comunicador. Aborta si es uno
        if (world_size < 2) {
                fprintf(stderr, "World tiene que ser >= 2%s\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 3);
        }
        
        // Proceso de creacion de comunicadores fila y columna guardando sus pid
	MPI_Group gr_comm_world, grf0, grf1, grf2, grf3, grc0, grc1, grc2, grc3;
	MPI_Comm CF0, CF1, CF2, CF3, CC0, CC1, CC2, CC3;
	
	for (i=0; i < n; i++){
		procf0[i] = i;
		procf1[i] = i+4;
		procf2[i] = i+8;
		procf3[i] = i+12;
		procc0[i] = i*4;
		procc1[i] = i*4+1;
		procc2[i] = i*4+2;
		procc3[i] = i*4+3;
	}
	MPI_Comm_group(MPI_COMM_WORLD, &gr_comm_world);
	
	MPI_Group_incl(gr_comm_world, n, procf0, &grf0);
	MPI_Group_incl(gr_comm_world, n, procf1, &grf1);
	MPI_Group_incl(gr_comm_world, n, procf2, &grf2);
	MPI_Group_incl(gr_comm_world, n, procf3, &grf3);
	MPI_Group_incl(gr_comm_world, n, procc0, &grc0);
	MPI_Group_incl(gr_comm_world, n, procc1, &grc1);
	MPI_Group_incl(gr_comm_world, n, procc2, &grc2);
	MPI_Group_incl(gr_comm_world, n, procc3, &grc3);
	
	MPI_Comm_create(MPI_COMM_WORLD, grf0, &CF0);
	MPI_Comm_create(MPI_COMM_WORLD, grf1, &CF1);
	MPI_Comm_create(MPI_COMM_WORLD, grf2, &CF2);
	MPI_Comm_create(MPI_COMM_WORLD, grf3, &CF3);
	MPI_Comm_create(MPI_COMM_WORLD, grc0, &CC0);
	MPI_Comm_create(MPI_COMM_WORLD, grc1, &CC1);
	MPI_Comm_create(MPI_COMM_WORLD, grc2, &CC2);
	MPI_Comm_create(MPI_COMM_WORLD, grc3, &CC3);
	
	MPI_Comm_rank(CF0, &pid_f0);
	MPI_Comm_rank(CF1, &pid_f1);
	MPI_Comm_rank(CF2, &pid_f2);
	MPI_Comm_rank(CF3, &pid_f3);
	MPI_Comm_rank(CC0, &pid_c0);
	MPI_Comm_rank(CC1, &pid_c1);
	MPI_Comm_rank(CC2, &pid_c2);
	MPI_Comm_rank(CC3, &pid_c3);

	// Define estructuras a usar en MPI
        MPI_Status info;
	MPI_Datatype subMatriz;
	MPI_Type_vector(subN, subN, N-subN, MPI_INT, &subMatriz);
	MPI_Type_commit(&subMatriz);
	
	for(i = 0; i < N; i++)
		printf("\nCOMM_WORLD:%d", world_rank);

        // Tareas para el proceso root
        if (world_rank == 0){
                // Reserva de memoria para las matrices A, B y R con control de errores
                A = (float **) malloc(N*sizeof(float *));
                B = (float **) malloc(N*sizeof(float *));
                R = (float **) malloc(N*sizeof(float *));
                for(i = 0; i < N; i++) {
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
                imprimir(A, 'A', N);
                imprimir(B, 'B', N);

		//Broadcasts entre comunicadores. Proceso de intercambio de submatrices para el calculo
                
		// Calculo de R parte correspondiente a rank0
		// En este caso cambiara
		for(i = 0; i < subN; i++)
			for(j = 0; j < subN; j++) {
				R[i][j] = 0.0;
				for(k = 0; k < N; k++)
					R[i][j] += A[i][k]*B[k][j];
			}

			
		// Imprime la matriz R
		imprimir(R, 'R', N);

                // Liberacion de la memoria
                free(A);
                free(B);
                free(R);
        } else { // Tareas para el resto de procesos
		// Reserva de memoria para las submatrices subA, subB y subR
                subA = (float **) malloc(subN*sizeof(float *));
                subB = (float **) malloc(subN*sizeof(float *));
                subR = (float **) malloc(subN*sizeof(float *));
                for(i = 0; i < subN; i++) {
                        subA[i] = (float *) malloc(subN*sizeof(float));
                        subB[i] = (float *) malloc(subN*sizeof(float));
                        subR[i] = (float *) malloc(subN*sizeof(float));
                }

		// Proceso de recepcion y calculo de subR, luego devolver a root
		
		//Libera espacio de memoria
		free(subA);
                free(subB);
                free(subR);
	}

	// Liberacion de los comunicadores
	MPI_Comm_free(&CF0);
	MPI_Comm_free(&CF1);
	MPI_Comm_free(&CF2);
	MPI_Comm_free(&CF3);
	MPI_Comm_free(&CC0);
	MPI_Comm_free(&CC1);
	MPI_Comm_free(&CC2);
	MPI_Comm_free(&CC3);
	
        // Cierre de MPI
        MPI_Finalize();

        // Finalizacion correcta del programa
        exit(0);
}

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
