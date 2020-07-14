
/******************************************************************************
*
* IMD -- The ITAP Molecular Dynamics Program
*
* Copyright 1996-2007 Institute for Theoretical and Applied Physics,
* University of Stuttgart, D-70550 Stuttgart
*
******************************************************************************/

/******************************************************************************
* $Revision: 1.89 $
* $Date: 2007/11/15 17:43:09 $
******************************************************************************/

#define MAIN

/* include file also declares global variables and has function prototypes */
#include "imd.h"
#ifdef PAPI
#include <papi.h>
#endif

#if defined(CBE)
#include "imd_cbe.h"
#endif

/* Main module of the IMD Package;
   Various versions are built by conditional compilation */

int main(int argc, char **argv)
{
  int start, num_threads, i;
  int simulation = 1, finished = 0;
  real tmp;
#ifdef PAPI
  float rtime, ptime, mflops;
  long_long flpins;
#endif

#if defined(MPI) || defined(NEB)
  MPI_Init(&argc, &argv);
  init_mpi();
#endif

  /* initialize timers */
  imd_init_timer( &time_total,      0, NULL,        NULL    );
  imd_init_timer( &time_setup,      1, "setup",     "white" );
  imd_init_timer( &time_main,       0, NULL,        NULL    );
  imd_init_timer( &time_output,     1, "output",    "cyan"  );
  imd_init_timer( &time_input,      1, "input",     "orange");
  imd_init_timer( &time_integrate,  1, "integrate", "green" );
  imd_init_timer( &time_forces,     1, "forces",    "yellow");

  read_command_line(argc,argv);

  /* loop for online visualization */
  do {

  time(&tstart);


  /* start some timers (after starting MPI!) */
  imd_start_timer(&time_total);
  imd_start_timer(&time_setup);

  /* read parameters for first simulation phase */
  finished = read_parameters(paramfilename, simulation);

  /* initialize all potentials */
  setup_potentials();

#ifdef CBE
  /* CBE initialization must be after potential setup */
  if ( num_spus != cbe_init(num_spus, -1) ) {
    error("Could not initialize enough SPU threads!\n");
  }
  else {
    fprintf(stdout, "CBE info: %d SPUs are used.\n", num_spus);
  }
  fprintf(stdout, "CBE info: Pointers on PPU are %u bits wide\n", 
                  (unsigned) PPU_PTRBITS );
#endif  /* CBE */

#ifdef TIMING
  imd_start_timer(&time_input);
#endif
#ifdef NEB
  read_atoms_neb(infilename);
#else
  /* filenames starting with _ denote internal generation of configuration */
  if ('_' == infilename[0]) {
    generate_atoms(infilename);
  }
  else {
    read_atoms(infilename);
  }
#endif
#ifdef TIMING
  imd_stop_timer(&time_input);
#endif

#ifdef EPITAX
  if (0 == myid) 
    printf("EPITAX: Largest substrate atom number: %d\n", epitax_sub_n);
  epitax_number = epitax_sub_n;
  epitax_level  = substrate_level();
  check_boxheight();
#endif

#ifdef EWALD
  init_ewald();
#endif

  /* initialize socket I/O */
#ifdef SOCKET_IO
  if ((myid == 0) && (socket_int > 0)) init_socket();
#endif

  start = steps_min;  /* keep starting step number */

  /* write .eng file header */
  if ((imdrestart==0) && (eng_int>0)) write_eng_file_header();

  imd_stop_timer(&time_setup);
  imd_start_timer(&time_main);
#ifdef PAPI
  PAPI_flops(&rtime,&ptime,&flpins,&mflops);
#endif

#ifdef REFPOS
  init_refpos();
#endif

#ifdef CNA
  init_cna();
#endif

#ifdef LASER
  init_laser(); 
#endif

#ifdef TTM
  init_ttm();
#endif

  /* first phase of the simulation */
  if (steps_min <= steps_max) {
    if ((0==myid) && (0==myrank))
      printf("Starting simulation %d\n", simulation);
    if (main_loop(simulation)) finished = 1;
  }

  /* execute further phases of the simulation */
  while (finished==0) {
    simulation++;
    finished = read_parameters(paramfilename, simulation);
    if (steps_min <= steps_max) {
      make_box();  /* make sure the box size is still ok */
      if ((0==myid) && (0==myrank))
        printf("Starting simulation %d\n", simulation);
      if (main_loop(simulation)) finished = 1;
    }
  }

  imd_stop_timer(&time_main);
  imd_stop_timer(&time_total);
#ifdef PAPI
  PAPI_flops(&rtime,&ptime,&flpins,&mflops);
#endif

  /* write execution time summary */
  if ((0 == myid) && (0 == myrank)){
    if (NULL!= eng_file) fclose( eng_file);
    if (NULL!=msqd_file) fclose(msqd_file);
    time(&tend);
    steps_max -= start;
    if (steps_max < 0)
      error("Start step greater than maximum number of steps");

    printf("Done simulation.\n\n");
    printf("%s\n\n",progname);
    printf("started at %s", ctime(&tstart));
    printf("finished at %s\n", ctime(&tend));

#ifdef PAPI
    printf("Achieved %f megaflops per CPU\n\n", mflops);
#endif

#ifdef NBLIST
    printf("Neighbor list update every %d steps on average\n\n", 
           steps_max / nbl_count);
#endif

#ifdef EPITAX
    if (0 == myid) printf("EPITAX: %d atoms created.\n", nepitax);
#endif

#ifdef OMP
    num_threads = omp_get_max_threads();
#else
    num_threads = 1;
#endif

#ifdef MPI
    printf("Did %d steps with %ld atoms on %d physical CPUs.\n", 
           steps_max+1, natoms, num_cpus * num_threads / hyper_threads);
#ifdef OMP
    printf("Used %d processes with %d threads each.\n", num_cpus, num_threads);
#endif
#else
#ifdef OMP
    printf("Did %d steps with %ld atoms on %d physical CPUs.\n", 
           steps_max+1, natoms, num_threads / hyper_threads);
#else
    printf("Did %d steps with %ld atoms.\n", steps_max+1, natoms);
#endif
#endif

    printf("Used %f seconds cputime,\n", time_total.total);
    printf("%f seconds excluding setup time,\n", time_main.total);
    tmp =  ((num_cpus * num_threads / hyper_threads) * time_main.total / 
           (steps_max+1)) / natoms;
    printf("%e cpuseconds per step and atom\n", tmp);
    printf("(inverse is %e).\n\n", 1.0/tmp);

#ifdef TIMING
    printf("Output time:   %e seconds or %.1f %% of main loop\n",
           time_output.total, 100*time_output.total /time_main.total);
    printf("Input  time:   %e seconds or %.1f %% of main loop\n",
           time_input.total,100*time_input.total/time_main.total);
#endif

     fflush(stdout);
     fflush(stderr);
  }

  /* if looping, clean up old simulation */
  if (loop) {

    int k;

    /* empty all cells */
    for (k=0; k<nallcells; k++) cell_array[k].n = 0;

    /* free potential tables */
#if defined(PAIR)
    free_pot_table(&pair_pot);
#endif
#ifdef TTBP
    free_pot_table(&smooth_pot);
#endif
#ifdef EAM2
    free_pot_table(&embed_pot);
    free_pot_table(&rho_h_tab);
#ifdef EEAM
    free_pot_table(&emod_pot);
#endif
#endif

    volume_init = 0.0;

    /* force init_cells */
    max_height.x = 0.0;
    max_height.y = 0.0;
#ifndef TWOD
    max_height.z = 0.0;
#endif

    /* close open files */
    if (myid==0) {
      if ( eng_file!=NULL) fclose( eng_file);  eng_file=NULL;
      if (msqd_file!=NULL) fclose(msqd_file); msqd_file=NULL;
    }
  }

  } while (loop);

  /* kill MPI */
#if defined(MPI) || defined(NEB)
  shutdown_mpi();
#endif

/* Added by Frank Pister */
#if defined(CBE)
   cbe_shutdown();
#endif

  /* Modified by F.P.:  We return, we don't exit :-) */
  return 0;
}
