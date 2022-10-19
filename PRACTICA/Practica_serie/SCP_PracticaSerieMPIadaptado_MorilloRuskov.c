/* Daniel Ruskov y Mikel Morillo
*  Practica de Sistemas de Computo Paralelo 2020
*  Ingenieria Informatica UPV/EHU curso 3º IC
*  Ultima modificacion 08/05/2020
*  Informe tecnico aparte con la informacion necesaria sobre este programa
*  Enlaces de interes:
*  https://en.wikipedia.org/wiki/Beta_distribution
*  https://www.gnu.org/software/gsl/doc/html/randist.html
*  https://en.wikipedia.org/wiki/File:Beta_distribution_pdf.svg
*/

/* COMPILACION
*	gcc -o nombre_ejecutable nombre_este_archivo.c -lgsl
*/

/* Dear teacher/programmer:
*	When we wrote this code, only God and we knew how it worked.
*	Now, only God knows!
*	So if you are trying to understand how it works and you succeed, 
*	you deserve a score of 10 and each one of us too. Else, we can leave 
*	it at 5 for each one. Please, do not forget that, at some point in life,
*	it compiled, ran and worked correctly!
*	If you are trying to optimize this routine and fail, (very likely)
*	please increase the following counter as a warning to the next developer:
*
*	total_hours_lost_here = 0;
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
#define SEMILLA 5
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// Estructura-Clase que representa una persona en la simulacion
struct T_Persona {
	int Edad;					// Edad entre 0 y 100
	char Estado;				// (0) sano, (1) infectado sin sı́ntomas, (2)infectado con sı́ntomas y (3) recuperado
	float ProbabMorision;		// Probabilidad de morir una vez infectado
	int Posicion[2];			// Vector p={px, py} que representa posición de individuo en el escenario
	int Velocidad[2];			// Vector v={vx, vy} que representa dirección y la velocidad de movimiento
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

// Definicion global para las estructuras a utilizar y los descriptores de fichero y var
struct T_Persona ***Escenario;	// Puntero a plano 2D de simulacion. Matriz de punteros a personas
struct T_Poblacion *Poblacion;	// Puntero a datos de poblacion
struct T_Metricas *Metricas;	// Puntero a datos de metricas
FILE *fmetr, *fpos;				// Descriptore de ficheros para recogida de metricas
int Duracion, DimV, DimH;		// Dimensiones del plano 2D y duracion de la simulacion
time_t t;						// Para usar con tiempo
struct tm *tm;					// Para guardar datos de time
gsl_rng *r;						// Para uso de distribuciones
const gsl_rng_type *T;			// Para uso de distribuciones
int world_size, world_rank;		// Variables para mpi
double t1, t2, t3;				// Variables para medir tiempo

// Definicion de funciones como indice y para resolver las depencencias
int Calc_Velocidad();
float Calc_Probab_01();
float Calc_ProbabMorision(int);
void Inicializacion(int, int, int, int, int, float, int, int, int);
void Simulacion();
void Inic_Ficheros();
// int Calc_Edad();					//***
// int Calc_PeriodoIncubacion();	//***
// int Calc_PeriodoRecuperacion();	//***
// double gsl_ran_beta(const gsl_rng * r, double a, double b);

// Proceso MAIN
int main(int argc, char *argv[]) {

	// Comprobacion del numero de argumentos
	if (argc != 10) {
		fprintf(stderr, "Error de uso!\nUSO: %s DimV DimH NumIndividuos DuracionSimulac RadioContagio ProbabContagio PeriodoIncubacion PeriodoRecuperacion TiempoBatch\n", argv[0]);
		exit(1);
	}
	
	// Inicializa MPI y guarda los valores de tamaño de comunicador y rangos. Para ejecucion serie hacer con world_size = 1
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	t1 = MPI_Wtime();
	// Inicializacion del sistema
	//fprintf(stdout, "Inicializando datos...");		//Feedback
	// argv -> [(0)NombrePrograma, (1)DimV, (2)DimH, (3)NumIndividuos, (4)Duracion, (5)RadioContagio, (6)ProbabContagio, (7)PeriodoIncubacion, (8)PeriodoRecuperacion, (9)TiempoBatch]
	Inicializacion(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atof(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]));
	//fprintf(stdout, "\t\tinicializados.\n");		// Feedback

	// Creacion y apertura de ficheros. Escritura de las cabeceras
	//fprintf(stdout, "Inicializando ficheros...");	//Feedback
	Inic_Ficheros();
	//fprintf(stdout, "\tinicializados.\n");			//Feedback
	
	t2 = MPI_Wtime();
	
	// Simulacion
	//fprintf(stdout, "Ejecutando simulacion...");	// Feedback
	Simulacion();
	//fprintf(stdout, "\trealizado.\n");				// Feedback

	t3 = MPI_Wtime();

	// Cierre de ficheros
	//fprintf(stdout, "Cerrando ficheros...");		// Feedback
	fmetr = NULL;	//fclose(fmetr);
	fpos = NULL;	//fclose(fpos);
	//fprintf(stdout, "\t\tcerrados.\n");				// Feedback

	// Liberacion de memoria
	//fprintf(stdout, "Liberando memoria...");		// Feedback
	free(Escenario);
	free(Poblacion);
	gsl_rng_free(r);
	//fprintf(stdout, "\t\tliberada.\n");				// Feedback

	// FIN
	fprintf(stdout, "FINALIZACION CORRECTA\n\tTiempo inicializacion: %1.2f\n\tTiempo simulacion: %1.2f\n\tTiempo total: %1.2f\n", t2-t1, t3-t2, t3-t1);		// Feedback
	
	// Cierre de MPI
	MPI_Finalize();
	exit(0);
}

// Funcion para calcular el cambio de vvelocidad -MAX_VEL..MAX_VEL
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
	int media = 0, i, j;

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
	
	// Crea el escenario de la simulacion (matriz 2D de punteros a T_Persona inicialmente todas NULL)
	Escenario = (struct T_Persona ***) malloc(DimensionV * sizeof(struct T_Persona **));
	for (i = 0; i < DimensionV; i++) {
		Escenario[i] = (struct T_Persona **) malloc(DimensionH * sizeof(struct T_Persona *));
		for (j = 0; j < DimensionH; j++) {
			Escenario[i][j] = NULL;//(struct T_Persona *) malloc(sizeof(struct T_Persona));
		}
	}

	// Inicializa el escenario de simulacion instanciando tantas personas como NumIndividuos
	// Paciente 0, primer contagiado
	struct T_Persona *Persona = malloc(sizeof(struct T_Persona));			// Crea la estructura a inicializar
	Persona->Edad = gsl_ran_beta(r, 2, 2) * 100;							// Calcula edad
	Persona->Estado = '1';													// PACIENTE 0 INFECTADO SIN SINTOMAS
	Persona->ProbabMorision = Calc_ProbabMorision(Persona->Edad);			// Probabilidad de morir si contagiado con sintomas
	Persona->Posicion[0] = rand() % (DimensionV);							// Posición del individuo en el eje x
	Persona->Posicion[1] = rand() % (DimensionH);							// Posición del individuo en el eje y
	Persona->Velocidad[0] = Calc_Velocidad();								// Velocidad del individuo en el eje x
	Persona->Velocidad[1] = Calc_Velocidad();								// Velocidad del individuo en el eje y
	media += Persona->Edad;													// Calculo de la edad media
	Escenario[Persona->Posicion[0]][Persona->Posicion[1]] = Persona;		// Posicionar en el escenario
	// Resto de individuos
	for (i = 1; i < NumIndividuos; i++) {
		struct T_Persona *Persona = malloc(sizeof(struct T_Persona));		// Crea la estructura a inicializar
		Persona->Edad = gsl_ran_beta(r, 2, 2) * 100;						// Calcula edad
		Persona->Estado = '0';												// Inicialmente sano
		Persona->ProbabMorision = Calc_ProbabMorision(Persona->Edad);		// Probabilidad de morir si contagiado con sintomas
		// Posición del individuo en el eje x, y libre
		while (Escenario[Persona->Posicion[0] = rand() % DimensionV][Persona->Posicion[1] = rand() % DimensionH] != NULL);
		Persona->Velocidad[0] = Calc_Velocidad();							// Velocidad del individuo en el eje x
		Persona->Velocidad[1] = Calc_Velocidad();							// Velocidad del individuo en el eje y
		media += Persona->Edad;												// Calculo de la edad media
		Escenario[Persona->Posicion[0]][Persona->Posicion[1]] = Persona;	// Posicionar en el escenario
	}
	
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
}

// Funcion dedicada a ejecutar la simulacion con los datos inicializados, y de escribir resultados en los ficheros de salida
void Simulacion() {
	
	// Definicion de variables internas a la funcion
	int i, j, t, x, y, rv, rh;																					// Indices
	int NuevosContagiados, TotContagiados;																		// Variables que ayudan a calcular R0

	// Simulacion
	for (t = 0; t < Duracion; t++) {																			// En cada unidad de tiempo calcular todas la interacciones entre los individuos

		// Exportacion de metricas en los ficheros de salida
		if (t % Metricas->TiempoBatch == 0 || t == Duracion - 1) {												// Si es in instante de tiempo multipo a batch
			fprintf(fpos, "\nt=%i\n", t);																		// Imprime el valor del instante t
			for (i = 0; i < DimV; i++) {																		// Eje x del escenario
				for (j = 0; j < DimH; j++) {																	// Eje y del escenario
					if (Escenario[i][j] != NULL) {																// Si existe persona situada en esta posicion
						fprintf(fpos, "\tPos (%i, %i) -> estado %c\n", i, j, Escenario[i][j]->Estado);			// Imprime su posicion y estado
					}
				}
			}
			// Imprime los valores de la estructura metricas junto al instante t correspondiente
			fprintf(fmetr, "%i\t\t%i\t\t%i\t\t%i\t\t%i\t\t%i\t\t%.4f\n", t, Metricas->PersSanas, Metricas->PersContagSinSintomas, Metricas->PersContagConSintomas, Metricas->PersRecuperadas, Metricas->PersMorisionadas, Metricas->R0);
		}

		// Calculo de interacciones, contagios, nuevos estados, nueva velocidad y futura posicion
		NuevosContagiados = 0;																					// Reinicia el contador de nuevos contagios para el nuevo instante de tiempo
		for (i = 0; i < DimV; i++) {																			// Eje x del escenario
			for (j = 0; j < DimH; j++) {																		// Eje y del escenario
				if (Escenario[i][j] != NULL) {																	// Si existe persona situada en esta posicion
					
					// Calcular su futura posicion
					if (Calc_Probab_01() <= Poblacion->ProbabVelDir) {											// Probabilidad cambio velocidad eje x
						Escenario[i][j]->Velocidad[0] = Calc_Velocidad();										// Calculo velocidad eje x
					}
					if(Escenario[i][j]->Velocidad[0] < 0) {														// Si velocidad negativa
						Escenario[i][j]->Posicion[0] = MAX(0, i + Escenario[i][j]->Velocidad[0]);				// Posicion no inferor a 0 en eje x
					} else {																					// Si velocidad positiva
						 Escenario[i][j]->Posicion[0] = MIN(DimV - 1, i + Escenario[i][j]->Velocidad[0]);		// Posicion no superior a la maxima del eje x				
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
							rv = MIN(DimV, i + Poblacion->RadioContagio + 1);									// rv = i + radio de contagio sin salir del plano en vertical
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

		// Calculo de R0
		TotContagiados = Metricas->PersContagSinSintomas + Metricas->PersContagConSintomas;
		Metricas->R0 = TotContagiados > 0 ? (float) NuevosContagiados / (float) TotContagiados : 0.0;			// Evita dividir entre 0
		// Desplazar personas segun su velocidad para t = t + 1 (posicion ya calculada)
		for (i = 0; i < DimV; i++) {																			// Eje x del escenario
			for (j = 0; j < DimH; j++) {																		// Eje y del escenario
				if (Escenario[i][j] != NULL) {																	// Si existe persona situada en esta posicion
					if (Escenario[Escenario[i][j]->Posicion[0]][Escenario[i][j]->Posicion[1]] == NULL) {		// Si es posicion libre
						Escenario[Escenario[i][j]->Posicion[0]][Escenario[i][j]->Posicion[1]] = Escenario[i][j];// Desplazar a la nueva posicion
						Escenario[i][j] = NULL;																	// Eliminar de la anterior
					} /*else {	// Restaurar los valores de posicion para que no se acumulen, pero no hace falta porque no se acumulan por la forma de calcularlos antes del switch
						Escenario[i][j]->Posicion[0] = i;
						Escenario[i][j]->Posicion[1] = j;
					}*/
				}
			}
		}
	}
}

// Funcion crear y abrir ficheros de salida e  imprimir sus cabeceras con informacion inicial 
void Inic_Ficheros() {

	char buffer[TAM_BUF_128];		// Buffer auxiliar para crear los nombres de fichero y escribir en ellos

	// Apertura de ficheros modo escrtitura. Los crea o los sobreescribe si existen
	strftime(buffer, TAM_BUF_128, "Metricas_%Y%m%d-%H%M%S.txt", tm);
	fmetr = fopen(buffer, "w");		// Metricas_AAAAMMDD-HHMMSS.txt
	strftime(buffer, TAM_BUF_128, "Posiciones_%Y%m%d-%H%M%S.txt", tm);
	fpos = fopen(buffer, "w");		// Posiciones_AAAAMMDD-HHMMSS.txt
	strftime(buffer, TAM_BUF_128, "%Y/%m/%d %H:%M:%S", tm);

	// Cabecera para el fichero de metricas
	fprintf(fmetr, "=======================================================================================================\n");
	fprintf(fmetr, "Datos de la simulacion\t%s\n", buffer);
	fprintf(fmetr, "\tPlano de simulacion2D: %ix%i\n", DimV, DimH);
	fprintf(fmetr, "\tNumero de individuos: %i\t\tMedia de edad: %.2f\n", Metricas->NumPersonas, Poblacion->MediaEdad);
	fprintf(fmetr, "\tRadio de contagio: %i\t\t\tProbabilidad de contagio: %.4f\n", Poblacion->RadioContagio, Poblacion->ProbabContagio);
	fprintf(fmetr, "\tPeriodo de incubacion: %i\t\tPeriodo de recuperacion: %i\n", Poblacion->PeriodoIncubacion, Poblacion->PeriodoRecuperacion);
	fprintf(fmetr, "\tDuracion de la simulacion: %i\t\tPeriodo de batch para metricas: %i\n", Duracion, Metricas->TiempoBatch);
	fprintf(fmetr, "Mas datos sobre el movimiento en el fichero de posiciones\n");
	fprintf(fmetr, "=======================================================================================================\n");
	fprintf(fmetr, "Instante t\tSanos\t\tSin sintomas\tCon sintomas\tRecuperados\tFallecidos\tR0\n");
	fprintf(fmetr, "=======================================================================================================\n");
	
	// Cabecera para el fichero de posiciones
	fprintf(fpos, "=========================================================================\n");
	fprintf(fpos, "Datos de la simulacion\t%s\n", buffer);
	fprintf(fpos, "\tNumero de individuos: %i\n", Metricas->NumPersonas);
	fprintf(fpos, "\tRadio de contagio: %i\n", Poblacion->RadioContagio);
	fprintf(fpos, "\tDuracion de la simulacion: %i\n", Duracion);
	fprintf(fpos, "\tPeriodo de batch para metricas: %i\n", Metricas->TiempoBatch);
	fprintf(fpos, "\tPlano de simulacion2D: %ix%i\n", DimV, DimH);
	fprintf(fpos, "\tEstado: (0)Sano, (1)Sin sintomas, (2)Con sintomas y (3)Recuperado\n");
	fprintf(fpos, "Mas datos detallados en el fichero de metricas\n");
	fprintf(fpos, "=========================================================================\n");
}


