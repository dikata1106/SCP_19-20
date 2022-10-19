/* Daniel Ruskov y Mikel Morillo
*  Practica de Sistemas de Computo Paralelo 2020
*  Ingenieria Informatica UPV/EHU curso 3º IC
*  Ultima modificacion 08/06/2020
*  Informe tecnico aparte con la informacion necesaria sobre este programa
*  Enlaces de interes:
*  https://en.wikipedia.org/wiki/Beta_distribution
*  https://www.gnu.org/software/gsl/doc/html/randist.html
*  https://en.wikipedia.org/wiki/File:Beta_distribution_pdf.svg
*/

/* COMPILACION
*	mpicc -o ejecutable EsteFichero.c -lgsl
*  EJECUCION (con argumentos de ejemplo)
*	./run.sh "executable 100 100 850 50 2 0.8 10 10 2"
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_gamma.h>

#define MAX_EDAD 100
#define MAX_VEL 3
#define TAM_BUF_128 128
#define TAM_BUF_1024 1024
#define TAM_BUF_PACK 25 		// 5 int * 4 bytes + 1 char * 1 byte + 1 float * 4 byte (edad, pos0, pos1, vel0, vel1, estado, probab)
#define SEMILLA 5
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// Estructura-Clase que representa una persona en la simulacion
struct T_Persona {
	int Edad;					// Edad entre 0 y 100
	int Posicion[2];			// Vector p={px, py} que representa posición de individuo en el escenario
	int Velocidad[2];			// Vector v={vx, vy} que representa dirección y la velocidad de movimiento
	char Estado;				// (0) sano, (1) infectado sin sı́ntomas, (2)infectado con sı́ntomas y (3) recuperado
	float ProbabMorision;		// Probabilidad de morir una vez infectado
};

// Estructura-Clase que representa la informacion de la poblacion en la simulacion
struct T_Poblacion {
	int Tamano;					// Número máximo de individuos que tiene la población
	float MediaEdad;			// Media de edad de los individuos
	int RadioContagio;			// Contagiados contagian a otros en un radio menor o igual a este parámetro
	float ProbabContagio;		// Dentro del radio de contagio pueden ser o no contagiados en función de este parámetro
	int PeriodoIncubacion;		// Tiempo desde contagio hasta muestra de sintomas 
	int PeriodoRecuperacion;	// Tiempo desde muestra de sintomas hasta recuperacion
	float ProbabVelDir;			// Probabilidad de cambio de velocidad y dirección (puede cambiar aleatoriamente)
};

// Estructura-Clase que representa la informacion recogida en cada unidad de tiempo resultado de la simulacion
struct T_Metricas {
	int NumPersonas;			// Numero total de personas en la simulacion
	int PersSanas;				// Numero de personas sanas/no contagiadas
	int PersContagSinSintomas;	// Numero de personas contagiadas sin sintomas
	int PersContagConSintomas;	// Numero de personas contagiadas con sintomas
	int PersRecuperadas;		// Numero de personas recuperadas
	int PersMorisionadas;		// Numero de personas fallecidas
	float R0;					// Numero reproductivo básico: capacidad de contagio o número de personas que es capaz de contagiar un paciente infectado
	int TiempoBatch;			// Periodo de tiempo de exportacion de metricas a fichero de salida
};

// Definicion global para las estructuras a utilizar, los descriptores de fichero y variables
struct T_Persona ***Escenario;	// Puntero a plano 2D de simulacion. Matriz de punteros a personas
struct T_Persona *PersonaTmp;	// Puntero a persona, temporal para recepcion de personas en envios
struct T_Poblacion *Poblacion;	// Puntero a datos de poblacion
struct T_Metricas *Metricas;	// Puntero a datos de metricas
struct T_Metricas *MetricasRoot;// Puntero a datos de metricas
MPI_File fmetr, fpos;			// Descriptore de ficheros para recogida de metricas
int Duracion, DimV, DimH;		// Dimensiones del plano 2D y duracion de la simulacion
int subV, myFloor, myCeil, mod;	// Variables para el reparto entre procesos
int i, j, k;					// Indices de ayuda
int world_rank, world_size, fl;	// Variables MPI
double t1, t2, t3;				// Variables para medir tiempo
MPI_Status status;				// Define estructuras a usar en MPI
MPI_Request request;			// Define estructuras a usar en MPI
MPI_Info info;					// Define estructuras a usar en MPI
time_t t;						// Para usar con tiempo
struct tm *tm;					// Para guardar datos de time
gsl_rng *r;						// Para uso de distribuciones
const gsl_rng_type *T;			// Para uso de distribuciones
char BUF[TAM_BUF_1024];			// Buffer para E/S
char buf_pack[TAM_BUF_PACK];	// Buffer para empaquetar persona en el envio de datos entre procesos
int pos;						// Variable usada ara el empaquetado y desempaquetado

// Definicion de funciones como indice y para resolver las depencencias
int Calc_Velocidad();
float Calc_Probab_01();
float Calc_ProbabMorision(int);
void Inicializacion(int, int, int, int, int, float, int, int, int);
void Simulacion();
void Inic_Ficheros();

// Proceso MAIN
int main(int argc, char *argv[]) {

	// Comprobacion del numero de argumentos
	if (argc != 10) {
		fprintf(stderr, "Error de uso!\nUSO: %s DimV DimH NumIndividuos DuracionSimulac RadioContagio ProbabContagio PeriodoIncubacion PeriodoRecuperacion TiempoBatch\n", argv[0]);
		exit(1);
	}
	
	// Inicializa MPI y guarda los valores de tamaño de comunicador y rangos
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_rank == 0) {
        // Comprueba numero de procesos del comunicador. Aborta si es uno
		if (world_size < 2) {
		    fprintf(stderr, "World tiene que ser >= 2\n");
		    MPI_Abort(MPI_COMM_WORLD, 2);
		}

		t1 = MPI_Wtime();	// Instante de tiempo 1 - inicio programa
    }
    
	// Inicializacion del sistema
	// argv -> [(0)NombrePrograma, (1)DimV, (2)DimH, (3)NumIndividuos, (4)Duracion, (5)RadioContagio, (6)ProbabContagio, (7)PeriodoIncubacion, (8)PeriodoRecuperacion, (9)TiempoBatch]
	Inicializacion(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atof(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]));

	// Creacion y apertura de ficheros. Escritura de las cabeceras por rank 0
	Inic_Ficheros();
	
	if (world_rank == 0) {
		t2 = MPI_Wtime();	// Instante de tiempo 2 - fin inicializaciones
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	// Simulacion
	Simulacion();

	// Cierre de ficheros
	MPI_File_close(&fmetr); 
	MPI_File_close(&fpos);

	if (world_rank == 0) {
		t3 = MPI_Wtime();	// Instante de tiempo 3 - fin simulacion

		free(MetricasRoot);	// Libera memoria exclusiva de root
	}

	// Liberacion de memoria todos los procesos
	free(Escenario);
	free(Poblacion);
	free(Metricas);
	free(PersonaTmp);
	gsl_rng_free(r);
	
	// Feedback y tiempos
	if (world_rank == 0) {
		fprintf(stdout, "FINALIZACION CORRECTA\n\tTiempo inicializacion: %1.2f\n\tTiempo simulacion: %1.2f\n\tTiempo total: %1.2f\n", t2-t1, t3-t2, t3-t1);
	}

	// Cierre de MPI
	MPI_Finalize();

	// FIN
	exit(0);
}

// Funcion para calcular el cambio de velocidad -MAX_VEL..MAX_VEL
int Calc_Velocidad() {

	return ((unsigned) rand() % (2 * MAX_VEL + 1)) - MAX_VEL;
}

// Funcion para calcular probabolodad 0..1 aleatoria formato 1.4f
float Calc_Probab_01() {	// DISTRIBUCION UNIFORME

	return ((float) (rand() % 10001)/10000);
}

// Funcion para calcular la probabilidad de morision segun la edad al estar contagiado con sintomas
float Calc_ProbabMorision(int Edad) {	// DISTRIBUCION AD-HOC

	if (Edad < 50)
		return 0.004;
	else if (Edad >= 50 && Edad < 60)
		return 0.013;
	else if (Edad >= 60 && Edad < 70)
		return 0.036;
	else if (Edad >= 70 && Edad < 80)
		return 0.080;
	else 	// Edad>=80
		return 0.148;
}

// Funcion de iniciaizacion de la simulacion
void Inicializacion(int DimensionV, int DimensionH, int NumIndividuos, int Durac, int RadioContagio, float ProbabContagio, int PeriodoIncubacion, int PeriodoRecuperacion, int TiempoBatch) {
	
	// Definicion de variables auxiliares internas a la funcion
	int media = 0, i, j, subNumIndividuos;

	// Inicializacion de algunos parametros necesarios
	DimV = DimensionV;														// Dimension vertical del plano 2D
	DimH = DimensionH;														// Dimension horizontal del plano 2D
	Duracion = Durac;														// Duracion de la simulacion en unidades de tiempo
	t = time(NULL);															// Relacionados con el tiempo para los nombres de fichero ediciones de tiempo
	tm = localtime(&t);														// Relacionados con el tiempo para los nombres de fichero ediciones de tiempo
	srand(time(NULL));														// Relacionado con la semilla para valores aleatorios sin patron
	T = gsl_rng_default;													// Relacionado con distribuciones de GSL
    r = gsl_rng_alloc(T);													// Relacionado con distribuciones de GSL
    gsl_rng_set(r, SEMILLA);												// Relacionado con distribuciones de GSL, inicializacion de valores con semilla
	gsl_rng_env_setup();													// Iniciar libreria
	mod = DimV % world_size;												// Resto de dividir numero de filas entre numero de procesos
	myCeil = (DimV / world_size) + 1;										// Resultado por exceso de dividir numero de filas entre numero de procesos
	myFloor = DimV / world_size;											// Resultado parte entera de dividir numero de filas entre numero de procesos
	subV = world_rank < mod ? myCeil : myFloor;								// Inicializa el numero de filas del plano que correspondera tener cada proceso
	subNumIndividuos = NumIndividuos / world_size;							// Numero de individuos en cada subplano (inicialmente se reparte numero igual de personas para cada proceso)
	
	// Crea el subescenario de la simulacion correspondiente a cada proceso (matriz 2D de punteros a T_Persona inicialmente todas NULL)
	Escenario = (struct T_Persona ***) malloc(subV * sizeof(struct T_Persona **));
	for (i = 0; i < subV; i++) {
		Escenario[i] = (struct T_Persona **) malloc(DimensionH * sizeof(struct T_Persona *));
		for (j = 0; j < DimensionH; j++) {
			Escenario[i][j] = NULL;
		}
	}

	// Inicializa el escenario de simulacion instanciando tantas personas como NumIndividuos
	if ( world_rank == 0) {
		
		// Paciente 0, primer contagiado solo en rank 0
		struct T_Persona *Persona = malloc(sizeof(struct T_Persona));			// Crea la estructura a inicializar
		Persona->Edad = gsl_ran_beta(r, 2, 2) * 100;							// Calcula edad
		Persona->Estado = '1';													// PACIENTE 0 INFECTADO SIN SINTOMAS
		Persona->ProbabMorision = Calc_ProbabMorision(Persona->Edad);			// Probabilidad de morir si contagiado con sintomas
		Persona->Posicion[0] = rand() % (subV);									// Posición del individuo en el eje x
		Persona->Posicion[1] = rand() % (DimensionH);							// Posición del individuo en el eje y
		Persona->Velocidad[0] = Calc_Velocidad();								// Velocidad del individuo en el eje x
		Persona->Velocidad[1] = Calc_Velocidad();								// Velocidad del individuo en el eje y
		media += Persona->Edad;													// Calculo de la edad media
		Escenario[Persona->Posicion[0]][Persona->Posicion[1]] = Persona;		// Posicionar en el escenario
		
		// Resto de individuos de rank 0
		for (i = 1; i < subNumIndividuos; i++) {
			struct T_Persona *Persona = malloc(sizeof(struct T_Persona));		// Crea la estructura a inicializar
			Persona->Edad = gsl_ran_beta(r, 2, 2) * 100;						// Calcula edad
			Persona->Estado = '0';												// Inicialmente sano
			Persona->ProbabMorision = Calc_ProbabMorision(Persona->Edad);		// Probabilidad de morir si contagiado con sintomas
			// Posición del individuo en el eje x, y libre
			while (Escenario[Persona->Posicion[0] = rand() % subV][Persona->Posicion[1] = rand() % DimensionH] != NULL);
			Persona->Velocidad[0] = Calc_Velocidad();							// Velocidad del individuo en el eje x
			Persona->Velocidad[1] = Calc_Velocidad();							// Velocidad del individuo en el eje y
			media += Persona->Edad;												// Calculo de la edad media
			Escenario[Persona->Posicion[0]][Persona->Posicion[1]] = Persona;	// Posicionar en el escenario
		}

		// Inicializa los parametros de las metricas para t=0
		MetricasRoot = (struct T_Metricas *) malloc(sizeof(struct T_Poblacion));	// Crea la estructura a inicializar
		MetricasRoot->NumPersonas = NumIndividuos;									// Numero de personas en la simulacion
		MetricasRoot->PersSanas = NumIndividuos - 1;								// Numero de personas contagiadas (todas menos paciente 0)
		MetricasRoot->PersContagSinSintomas = 1;									// Numero de personas contagiadas sin sintomas (paciente 0)
		MetricasRoot->PersContagConSintomas = 0;									// Numero de persinas contagiadas con sintomas (al principio no hay)
		MetricasRoot->PersRecuperadas = 0;											// Numero de personas recuperadas (al principio no hay)
		MetricasRoot->PersMorisionadas = 0;											// Numero de personas fallecidas (al principio no hay)
		MetricasRoot->R0 = 0.0;														// Capacidad de contagio (inicialmente 0)
		MetricasRoot->TiempoBatch = TiempoBatch;									// Periodo de tiempo entre recogida de metricas en ficheros de salida
	} else {
		
		// Individuos para el resto de procesos
		for (i = 0; i < subNumIndividuos; i++) {
			struct T_Persona *Persona = malloc(sizeof(struct T_Persona));		// Crea la estructura a inicializar
			Persona->Edad = gsl_ran_beta(r, 2, 2) * 100;						// Calcula edad
			Persona->Estado = '0';												// Inicialmente sano
			Persona->ProbabMorision = Calc_ProbabMorision(Persona->Edad);		// Probabilidad de morir si contagiado con sintomas
			// Posición del individuo en el eje x, y libre
			while (Escenario[Persona->Posicion[0] = rand() % subV][Persona->Posicion[1] = rand() % DimensionH] != NULL);
			Persona->Velocidad[0] = Calc_Velocidad();							// Velocidad del individuo en el eje x
			Persona->Velocidad[1] = Calc_Velocidad();							// Velocidad del individuo en el eje y
			media += Persona->Edad;												// Calculo de la edad media
			Escenario[Persona->Posicion[0]][Persona->Posicion[1]] = Persona;	// Posicionar en el escenario
		}
	}
	
	MPI_Allreduce( &media, &media, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);		// Suma de las medias de cada proceso para calculo de la media de edad

	// Inicializa los parametros estadisticos de poblacion
	Poblacion = (struct T_Poblacion *) malloc(sizeof(struct T_Poblacion));	// Crea la estructura a inicializar
	Poblacion->Tamano = DimensionV * DimensionH;							// Maximo de indivuduos, numero de elementos de la matriz escenario
	Poblacion->MediaEdad = media / NumIndividuos;							// Edad media de los individuos de la simulacion
	Poblacion->RadioContagio = RadioContagio;								// Radio en el que se puede contagiar
	Poblacion->ProbabContagio = ProbabContagio;								// Probabilidad de contagio en el radio
	Poblacion->PeriodoIncubacion = PeriodoIncubacion;						// Tiempo hasta la muestra de sintomas
	Poblacion->PeriodoRecuperacion = PeriodoRecuperacion;					// Tiempo hasta la recuperacion
	Poblacion->ProbabVelDir = Calc_Probab_01();								// Probabilidad inicial de cambio de velocidad y/o direccion (0..1)

	// Inicializa los parametros de las metricas para t=0
	Metricas = (struct T_Metricas *) malloc(sizeof(struct T_Poblacion));	// Crea la estructura a inicializar
	Metricas->NumPersonas = NumIndividuos;									// Numero de personas en la simulacion
	Metricas->PersSanas = NumIndividuos - 1;								// Numero de personas contagiadas (todas menos paciente 0)
	Metricas->PersContagSinSintomas = 1;									// Numero de personas contagiadas sin sintomas (paciente 0)
	Metricas->PersContagConSintomas = 0;									// Numero de persinas contagiadas con sintomas (al principio no hay)
	Metricas->PersRecuperadas = 0;											// Numero de personas recuperadas (al principio no hay)
	Metricas->PersMorisionadas = 0;											// Numero de personas fallecidas (al principio no hay)
	Metricas->R0 = 0.0;														// Capacidad de contagio (inicialmente 0)
	Metricas->TiempoBatch = TiempoBatch;									// Periodo de tiempo entre recogida de metricas en ficheros de salida

	PersonaTmp = (struct T_Persona *) malloc(sizeof(struct T_Persona));		// Utiliza en la recepcion de personas cuando cambian entre pprocesos
}

// Funcion dedicada a ejecutar la simulacion con los datos inicializados, y de escribir resultados en los ficheros de salida
void Simulacion() {
	
	// Definicion de variables internas a la funcion
	int i, j, t, x, y, rv, rh;																					// Indices
	int NuevosContagiados, TotContagiados;																		// Variables que ayudan a calcular R0

	// Simulacion
	for (t = 0; t < Duracion; t++) {																			// En cada unidad de tiempo calcular todas la interacciones entre los individuos

		// Exportacion de metricas en los ficheros de salida
		if (t % MetricasRoot->TiempoBatch == 0 || t == Duracion - 1) {											// Si es in instante de tiempo multipo a batch
			
			// Reduccion de las metricas hacia root (en metricas root)
			MPI_Reduce( &Metricas->PersContagSinSintomas, &MetricasRoot->PersContagSinSintomas, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce( &Metricas->PersContagConSintomas, &MetricasRoot->PersContagConSintomas, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce( &Metricas->PersRecuperadas, &MetricasRoot->PersRecuperadas, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce( &Metricas->PersMorisionadas, &MetricasRoot->PersMorisionadas, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce( &NuevosContagiados, &NuevosContagiados, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

			if (world_rank == 0) {

				// Imprime tiempo de batch correspondiente para que los procesos escriban en desorden las posiciones de las personas que contienen
				sprintf(BUF, "\nt=%i\n", t);
				MPI_File_write(fpos, &BUF, strlen(BUF), MPI_CHAR, &status);

				// Calcula resto de metricas para el tiempo de batch correspondiente (Personas sanas y R0)
				MetricasRoot->PersSanas = MetricasRoot->NumPersonas - MetricasRoot->PersContagSinSintomas - MetricasRoot->PersContagConSintomas - MetricasRoot->PersRecuperadas - MetricasRoot->PersRecuperadas - MetricasRoot->PersMorisionadas;
				TotContagiados = MetricasRoot->PersContagSinSintomas + MetricasRoot->PersContagConSintomas;
				MetricasRoot->R0 = TotContagiados > 0 ? (float) NuevosContagiados / (float) TotContagiados : 0.0;			// Evita dividir entre 0		
			}
			
			NuevosContagiados = 0;																				// Reinicia contador de nuevos contagiados tras el calculo de R0

			MPI_Barrier(MPI_COMM_WORLD);

			if (world_rank == 0) {
				// Imprime los valores de la estructura metricas junto al instante t correspondiente
				sprintf(BUF, "%i\t\t%i\t\t%i\t\t%i\t\t%i\t\t%i\t\t%.4f\n", t, MetricasRoot->PersSanas, MetricasRoot->PersContagSinSintomas, MetricasRoot->PersContagConSintomas, MetricasRoot->PersRecuperadas, MetricasRoot->PersMorisionadas, MetricasRoot->R0);
				MPI_File_write(fmetr, &BUF, strlen(BUF), MPI_CHAR, &status);
			}

			// Cada proceso imprime las posiciones de las personas que trata. Hace calculo de la posicion vertical real del escenario
			if (world_rank < mod) {
				for (i = 0; i < subV; i++) {																		// Eje x del escenario
					for (j = 0; j < DimH; j++) {																	// Eje y del escenario
						if (Escenario[i][j] != NULL) {																// Si existe persona situada en esta posicion
							sprintf(BUF,  "\tPos (%i, %i) -> estado %c\n", world_rank * myCeil + i, j, Escenario[i][j]->Estado);
							MPI_File_seek(fmetr, 0, MPI_SEEK_END);
							MPI_File_write(fpos, &BUF, strlen(BUF), MPI_CHAR, &status);
						}
					}
				}
			} else {
				for (i = 0; i < subV; i++) {																		// Eje x del escenario
						for (j = 0; j < DimH; j++) {																// Eje y del escenario
							if (Escenario[i][j] != NULL) {															// Si existe persona situada en esta posicion
								sprintf(BUF,  "\tPos (%i, %i) -> estado %c\n", mod * myCeil + (world_rank - mod) * myFloor + i, j, Escenario[i][j]->Estado);
								MPI_File_seek(fpos, 0, MPI_SEEK_END);
								MPI_File_write(fpos, &BUF, strlen(BUF), MPI_CHAR, &status);
							}
						}
					}
			}

		}

		// Calculo de interacciones, contagios, nuevos estados, nueva velocidad y futura posicion
		//NuevosContagiados = 0;																					// Reinicia el contador de nuevos contagios para el nuevo instante de tiempo
		for (i = 0; i < subV; i++) {																			// Eje x del escenario
			for (j = 0; j < DimH; j++) {																		// Eje y del escenario
				if (Escenario[i][j] != NULL) {																	// Si existe persona situada en esta posicion
					
					// Calcular su futura posicion
					if (Calc_Probab_01() <= Poblacion->ProbabVelDir) {											// Probabilidad cambio velocidad eje x
						Escenario[i][j]->Velocidad[0] = Calc_Velocidad();										// Calculo velocidad eje x
					}
					if (world_rank == 0) {
						if(Escenario[i][j]->Velocidad[0] < 0) {													// Si velocidad negativa
							Escenario[i][j]->Posicion[0] = MAX(0, i + Escenario[i][j]->Velocidad[0]);			// Posicion no inferor a 0 en eje x
						} else {																				// Si velocidad positiva
							Escenario[i][j]->Posicion[0] = i + Escenario[i][j]->Velocidad[0];					// Posicion con la que puede pasar a rank 1			
						}
					} else if (world_rank == world_size -1) {
						if(Escenario[i][j]->Velocidad[0] < 0) {													// Si velocidad negativa
							Escenario[i][j]->Posicion[0] = i + Escenario[i][j]->Velocidad[0];					// Posicion con la que uede pasar a world_size - 2
						} else {																				// Si velocidad positiva
							Escenario[i][j]->Posicion[0] = MIN(DimV - 1, i + Escenario[i][j]->Velocidad[0]);	// Posicion no superior a la maxima del eje x				
						}
					} else {
						Escenario[i][j]->Posicion[0] = i + Escenario[i][j]->Velocidad[0];						// Posicion con la que puede psar a world_rank +/- 1
					}
					if (Calc_Probab_01() <= Poblacion->ProbabVelDir) {											// Probabilidad cambio velocidad eje y
						Escenario[i][j]->Velocidad[1] = Calc_Velocidad();										// Calculo velocidad eje y
					}
					if (Escenario[i][j]->Velocidad[1] < 0 ) {													// Si velocidad negativa
						Escenario[i][j]->Posicion[1] = MAX(0, j + Escenario[i][j]->Velocidad[1]);				// Posicion no inferor a 0 en eje y
					} else {																					// Si velocidad positiva
						Escenario[i][j]->Posicion[1] = MIN(DimH - 1, j + Escenario[i][j]->Velocidad[1]);		// Posicion no superior a la maxima del eje y
					}

					// Actualizacion de estado y metricas
					switch (Escenario[i][j]->Estado) {															// Segun su estado
						case '0':																				// Sano
							x = MAX(0, i - Poblacion->RadioContagio);											// x = i - radio de contagio sin salir del plano en vertical
							rv = MIN(subV, i + Poblacion->RadioContagio + 1);									// rv = i + radio de contagio sin salir del plano en vertical
							rh = MIN(DimH, j + Poblacion->RadioContagio + 1);									// rh = j + radio de contagio sin salir del plano en horizontal
							while (x < rv && Escenario[i][j]->Estado == '0'){									// Para rango vertical del rango de contagio y mientras aun esta sano
								y = MAX(0, j - Poblacion->RadioContagio);										// y = j - radio de contagio sin salir del plano en horizontal. reinicia y para cada x
								while (y < rh && Escenario[i][j]->Estado == '0') {								// Para ranho horizontal dentro del rango te contagio y mientras aun esta sano
									if (Escenario[x][y] != NULL) {												// Si existe persona en posicion del radio
										if (Escenario[x][y]->Estado == '1' || Escenario[x][y]->Estado == '2') {	// Si es una persona contagiada
											if (Calc_Probab_01() <= Poblacion->ProbabContagio) {				// Ver si la contagia a la sana de i, j
												Escenario[i][j]->Estado = '1';									// Cambia a estado contagiado sin sintomas
												Metricas->PersContagSinSintomas++;								// Aumento contador personas sin sintomas
												Metricas->PersSanas--;											// Disminuye contador de personas sanas
												NuevosContagiados++;											// Aumenta contador de nuevos contagiados para instante t
											}
										}
									}
									y++;																		// Aumenta posicion en horizontal
								}
								x++;																			// Aumenta posicion en vertical
							}
							break;																				// Sale de switch
						case '1':																				// Contagiado sin sintomas
							if ((int) (gsl_ran_beta(r, 2, 2) * 14) <= Poblacion->PeriodoIncubacion) {			// Si supera periodo de incubacion
								Escenario[i][j]->Estado = '2';													// Pasa a estado contagiado con sintomas
								Metricas->PersContagSinSintomas--;												// Disminuye contador de contagiado sin sintomas
								Metricas->PersContagConSintomas++;												// Incrementa contador de contagiados con sintomas
							}
							break;																				// Sale de switch
						case '2':																				// Contagiado con sintomas
							if (Calc_Probab_01() <= Escenario[i][j]->ProbabMorision) {							// Ver si la palma
								Escenario[i][j] = NULL;															// Elimina la persona (c murio)
								Metricas->PersContagConSintomas--;												// Disminuye el contador de contagiados con sintomas
								Metricas->PersMorisionadas++;													// Aumenta el contador de fallecidos
							} else if ((int)  (gsl_ran_beta(r, 2, 2) * 14) <= Poblacion->PeriodoRecuperacion) {	// Ver si se recupera
								Escenario[i][j]->Estado = '3';													// Cambia a recuperado
								Metricas->PersContagConSintomas--;												// Disminuye contador de contagiados con sintomas
								Metricas->PersRecuperadas++;													// Aumenta contador de recuperados
							}
							break;																				// Sale del switch
						case '3':																				// Recuperado
							break;																				// Sale del switch (solo se desplaza, calculado arriba)
						default:																				// Caso de error
							Escenario[i][j] = NULL;																// Elimina la persona (c murio)
							Metricas->PersMorisionadas++;														// Cuenta fallecimiento de una persona
							break;																				// Sale del switch
					}
				}
			}
		}

		// Desplazar personas segun su velocidad para t = t + 1 (posicion ya calculada)
		// Primero todos los procesos envian las personas que cambian de proceso
		for (i = 0; i < subV; i++) {																			// Eje x del escenario
			for (j = 0; j < DimH; j++) {																		// Eje y del escenario
				if (Escenario[i][j] != NULL) {																	// Si existe persona situada en esta posicion
					if (Escenario[i][j]->Posicion[0] < 0) {

						// Empaquetado de la persona a enviar
						pos = 0;
						MPI_Pack(&Escenario[i][j]->Edad, 5, MPI_INT, buf_pack, TAM_BUF_PACK, &pos, MPI_COMM_WORLD);
						MPI_Pack(&Escenario[i][j]->Estado, 1, MPI_CHAR, buf_pack, TAM_BUF_PACK, &pos, MPI_COMM_WORLD);
						MPI_Pack(&Escenario[i][j]->ProbabMorision, 1, MPI_FLOAT, buf_pack, TAM_BUF_PACK, &pos, MPI_COMM_WORLD);

						// Envio no bloqueante de la persona empaquetada
						MPI_Isend(buf_pack, TAM_BUF_PACK, MPI_PACKED, world_rank-1, 0, MPI_COMM_WORLD, &request);

						Escenario[i][j] = NULL;																	// Eliminar de la anterior
					} else if (Escenario[i][j]->Posicion[0] > 0) {

						// Calcula posicion vertical de la persona para el proceso al que se le envia (world_rank + 1)
						Escenario[i][j]->Posicion[0] = Escenario[i][j]->Posicion[0] - subV;

						// Empaquetado de la persona a enviar
						pos = 0;
						MPI_Pack(&Escenario[i][j]->Edad, 5, MPI_INT, buf_pack, TAM_BUF_PACK, &pos, MPI_COMM_WORLD);
						MPI_Pack(&Escenario[i][j]->Estado, 1, MPI_CHAR, buf_pack, TAM_BUF_PACK, &pos, MPI_COMM_WORLD);
						MPI_Pack(&Escenario[i][j]->ProbabMorision, 1, MPI_FLOAT, buf_pack, TAM_BUF_PACK, &pos, MPI_COMM_WORLD);
						
						// Envio no bloqueante de la persona empaquetada
						MPI_Isend(buf_pack, TAM_BUF_PACK, MPI_PACKED, world_rank+1, 0, MPI_COMM_WORLD, &request);
	
						Escenario[i][j] = NULL;																	// Eliminar de la anterior
					}
				}
			}
		}

		// Ahora todos los procesos reciben las personas que les corresponde llegar
		// Se comprueba si hay mensajes por recibir
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &fl, &status);
		while (fl > 0) { // Mientras haya mensajes por recibir

			// Se recibe la persona correspondiente con la informacion de status
			MPI_Irecv(buf_pack, TAM_BUF_PACK, MPI_PACKED, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &request);

			// Se desempaqueta la prsona recibida y se guarda en PersonaTmp
			pos = 0;
			MPI_Unpack(buf_pack, TAM_BUF_PACK, &pos, &PersonaTmp->Edad, 5, MPI_INT, MPI_COMM_WORLD);
			MPI_Unpack(buf_pack, TAM_BUF_PACK, &pos, &PersonaTmp->Estado, 1, MPI_CHAR, MPI_COMM_WORLD);
			MPI_Unpack(buf_pack, TAM_BUF_PACK, &pos, &PersonaTmp->ProbabMorision, 1, MPI_FLOAT, MPI_COMM_WORLD);

			// Si la persona se ha desplazado hacia arriba, se calcula su posicion correspondiente en el nuevo proceso
			if (status.MPI_SOURCE == world_rank+1) {
				PersonaTmp->Posicion[0] = subV + PersonaTmp->Posicion[0];
			}

			// Se posiciona la persona en el subplano del proceso receptor
			Escenario[PersonaTmp->Posicion[0]][PersonaTmp->Posicion[1]] = PersonaTmp;

			// Se vuelve a comprobar si falta informacion por recibir
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &fl, &status);
		}

		// Sincronizar procesos para siguiente instante de tiempo
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

// Funcion crear y abrir ficheros de salida e  imprimir sus cabeceras con informacion inicial 
void Inic_Ficheros() {

	char buffer[TAM_BUF_128];		// Buffer auxiliar para crear los nombres de fichero y escribir en ellos

	// Apertura de ficheros modo escrtitura. Los crea o los sobreescribe si existen
	strftime(buffer, TAM_BUF_128, "Metricas_%Y%m%d-%H%M%S.txt", tm);
	MPI_File_open(MPI_COMM_WORLD, buffer, MPI_MODE_CREATE, info, &fmetr);
	strftime(buffer, TAM_BUF_128, "Posiciones_%Y%m%d-%H%M%S.txt", tm);
	MPI_File_open(MPI_COMM_WORLD, buffer, MPI_MODE_CREATE, info, &fpos);
	
	strftime(buffer, TAM_BUF_128, "%Y/%m/%d %H:%M:%S", tm);

	if (world_rank == 0) {
		// Cabecera para el fichero de metricas
		sprintf(BUF, "=======================================================================================================\nDatos de la simulacion\t%s\n\tPlano de simulacion2D: %ix%i\n\tNumero de individuos: %i\t\tMedia de edad: %.2f\n\tRadio de contagio: %i\t\t\tProbabilidad de contagio: %.4f\n\tPeriodo de incubacion: %i\t\tPeriodo de recuperacion: %i\n\tDuracion de la simulacion: %i\t\tPeriodo de batch para metricas: %i\nMas datos sobre el movimiento en el fichero de posiciones\n=======================================================================================================\nInstante t\tSanos\t\tSin sintomas\tCon sintomas\tRecuperados\tFallecidos\tR0\n=======================================================================================================\n", buffer, DimV, DimH,	Metricas->NumPersonas, Poblacion->MediaEdad, Poblacion->RadioContagio, Poblacion->ProbabContagio, Poblacion->PeriodoIncubacion, Poblacion->PeriodoRecuperacion, Duracion, Metricas->TiempoBatch);
		MPI_File_write(fmetr, &BUF, strlen(BUF), MPI_CHAR, &status);
	
		// Cabecera para el fichero de posiciones
		sprintf(BUF, "=========================================================================\nDatos de la simulacion\t%s\n\tNumero de individuos: %i\n\tRadio de contagio: %i\n\tDuracion de la simulacion: %i\n\tPeriodo de batch para metricas: %i\n\tPlano de simulacion2D: %ix%i\n\tEstado: (0)Sano, (1)Sin sintomas, (2)Con sintomas y (3)Recuperado\nMas datos detallados en el fichero de metricas\n=========================================================================\n", buffer, Metricas->NumPersonas, Poblacion->RadioContagio, Duracion, Metricas->TiempoBatch, DimV, DimH);
		MPI_File_write(fpos, &BUF, strlen(BUF), MPI_CHAR, &status);
	}
}

