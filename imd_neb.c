
/******************************************************************************
*
* IMD -- The ITAP Molecular Dynamics Program
*
* Copyright 1996-2007 Institute for Theoretical and Applied Physics,
* University of Stuttgart, D-70550 Stuttgart
*
******************************************************************************/

/******************************************************************************
*
* imd_neb -- functions for the NEB method
*
******************************************************************************/

/******************************************************************************
* $Revision: 1.1 $
* $Date: 2007/11/15 17:43:09 $
******************************************************************************/

#include "imd.h"

/* auxiliary arrays */
real *pos=NULL, *pos_l=NULL, *pos_r=NULL;
real pot_l=0.0, pot_r=0.0;				

/******************************************************************************
*
*  initialize MPI (NEB version)
*
******************************************************************************/

void init_mpi(void)
{
  /* Initialize MPI */
  MPI_Comm_size(MPI_COMM_WORLD,&num_cpus);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if (0 == myrank) { 
    printf("NEB: Starting up MPI with %d processes.\n", num_cpus);
  }
}

/******************************************************************************
*
*  shutdown MPI (NEB version)
*
******************************************************************************/

void shutdown_mpi(void)
{
  MPI_Barrier(MPI_COMM_WORLD);   /* Wait for all processes to arrive */
#ifdef MPELOG
  MPE_Log_sync_clocks();
#ifdef NO_LLMPE
  MPE_Finish_log( progname );
#endif
#endif
  MPI_Finalize();                /* Shutdown */
}

/******************************************************************************
*
*  allocate auxiliary arrays
*
******************************************************************************/

void alloc_pos(void) 
{
  pos   = (real *) malloc( DIM * natoms * sizeof(real ) );
  pos_l = (real *) malloc( DIM * natoms * sizeof(real ) );
  pos_r = (real *) malloc( DIM * natoms * sizeof(real ) );
  if ((NULL==pos) || (NULL==pos_l) || (NULL==pos_r))
    error("cannot allocate NEB position array");
}

/******************************************************************************
*
*  read all configurations (including initial and final)
*
******************************************************************************/

void read_atoms_neb(str255 infilename)
{
  str255 fname;
  int i, k, n;

  /* keep a copy of the outfile name without replica suffix */
  neb_outfilename = strdup(outfilename);

  /* read positions of initial configuration */
  if (0==myrank) {
    sprintf(fname, "%s.%02d", infilename, 0);
    myrank = 1;  /* avoid double info messages */
    read_atoms(fname);
    myrank = 0;
    alloc_pos();
    for (k=0; k<NCELLS; k++) {
      cell *p = CELLPTR(k);
      for (i=0; i<p->n; i++) { 
        n = NUMMER(p,i);
        pos_l X(n) = ORT(p,i,X);				/*@Zhiming Chen@: It requires a continuous numbering from 0 to N-1. */
        pos_l Y(n) = ORT(p,i,Y);
        pos_l Z(n) = ORT(p,i,Z);
      }
    }
    /* compute and write energy of initial configuration */
    calc_forces(0);
    pot_l=tot_pot_energy;				/*@Zhiming Chen@: get energy for initial image. */
    sprintf(outfilename, "%s.%02d", neb_outfilename, 0);
    write_eng_file_header();
    write_eng_file(0);
    fclose(eng_file);
    eng_file = NULL;
  }

  /* read positions of final configuration */
  
	  if ( (neb_fe == -1) && (neb_nrep-3==myrank) ) {
	    sprintf(fname, "%s.%02d", infilename, neb_nrep-1);
	    read_atoms(fname);
	    if (NULL==pos) alloc_pos();
	    for (k=0; k<NCELLS; k++) {
	      cell *p = CELLPTR(k);
	      for (i=0; i<p->n; i++) { 
	        n = NUMMER(p,i);
	        pos_r X(n) = ORT(p,i,X);
	        pos_r Y(n) = ORT(p,i,Y);
	        pos_r Z(n) = ORT(p,i,Z);
	      }
	    }
	    /* compute and write energy of initial configuration */
	    calc_forces(0);
	    pot_r=tot_pot_energy;				/*@Zhiming Chen@: get energy for final image. */
	    sprintf(outfilename, "%s.%02d", neb_outfilename, neb_nrep-1);
	    write_eng_file_header();
	    write_eng_file(0);
	    fclose(eng_file);
	    eng_file = NULL;
	  }
  	
  /* read positions of my configuration */
  sprintf(fname, "%s.%02d", infilename, myrank+1);
  read_atoms(fname);
  if (NULL==pos) alloc_pos();
  sprintf(outfilename, "%s.%02d", neb_outfilename, myrank+1);


}

/******************************************************************************
*
*  exchange positions with neighbor replicas
*
******************************************************************************/

void neb_sendrecv_pos(void)
{
  int i, k, n, cpu_l, cpu_r;
  MPI_Status status;

  /* fill pos array */
  for (k=0; k<NCELLS; k++) {
    cell *p = CELLPTR(k);
    for (i=0; i<p->n; i++) { 
      n = NUMMER(p,i);
      pos X(n) = ORT(p,i,X);
      pos Y(n) = ORT(p,i,Y);
      pos Z(n) = ORT(p,i,Z);
    }
  }

  /* ranks of left/right cpus */
  cpu_l = (0 == myrank) ? MPI_PROC_NULL : myrank - 1;
  if (neb_fe != -1) {
  	cpu_r = (neb_nrep - 2 == myrank) ? MPI_PROC_NULL : myrank + 1;
  	}
  else {
  	cpu_r = (neb_nrep - 3 == myrank) ? MPI_PROC_NULL : myrank + 1;
  	}
  

  /* send positions to right, receive from left */
  MPI_Sendrecv(pos,   DIM*natoms, REAL, cpu_r, BUFFER_TAG,
	       pos_l, DIM*natoms, REAL, cpu_l, BUFFER_TAG,
	       MPI_COMM_WORLD, &status );
  MPI_Sendrecv(&tot_pot_energy, 1, REAL, cpu_r, BUFFER_TAG,
  	       &pot_l, 1, REAL, cpu_l, BUFFER_TAG,
  	       MPI_COMM_WORLD, &status );				/*@Zhiming Chen@: pass energy. */

  /* send positions to left, receive from right */
  MPI_Sendrecv(pos,   DIM*natoms, REAL, cpu_l, BUFFER_TAG,
	       pos_r, DIM*natoms, REAL, cpu_r, BUFFER_TAG,
	       MPI_COMM_WORLD, &status );
  MPI_Sendrecv(&tot_pot_energy, 1, REAL, cpu_l, BUFFER_TAG,
	       &pot_r, 1, REAL, cpu_r, BUFFER_TAG,
	       MPI_COMM_WORLD, &status );				/*@Zhiming Chen@: pass energy. */
}

/******************************************************************************
*
*  modify forces according to NEB
*
******************************************************************************/

void calc_forces_neb(void)
{
  real norm3=0.0, norm4=0.0, norm5=0.0, f_norm=0.0;
  real tmp;
  real src[2], dest[2];
  real proj_spring=0.0, proj=0.0, k_proj=0.0;
  int climb=1, climbs=1;
  int k, i, n;
  int ci=0;

  norm1=0.0;
  norm2=0.0;
  checkangle=0.0;
  checkangle2=0.0;
  
  MPI_Barrier(MPI_COMM_WORLD);
  /* @Zhiming Chen@: stop if the band convergenced according to delta energy. */
  if (neb_delta > 0.0 && neb_delta > fabs(old_totpotenergy-tot_pot_energy) / natoms ) {
  	neb_stop = 1;
  	}
  old_totpotenergy = tot_pot_energy;

/*  MPI_Barrier(MPI_COMM_WORLD);	 */
  
  /* exchange positions with neighbor replicas */
  neb_sendrecv_pos();

	
  if ( (neb_fe > -1) && (myrank == neb_nrep-2) ) {
		pot_r = tot_pot_energy-0.1;
		pot_l = tot_pot_energy+0.1;
    	}

  /* compute tangent of current NEB path */
  for (i=0; i<DIM*natoms; i+=DIM) {

    vektor dl, dr;
    real x;

    /* distance to left and right replica */
    dl.x = pos  [i  ] - pos_l[i  ];
    dl.y = pos  [i+1] - pos_l[i+1];
    dl.z = pos  [i+2] - pos_l[i+2];

    if ( (neb_fe > -1) && (myrank == neb_nrep-2) ) {
		dr.x =  dr.y = dr.z = 0.0;		
    	}
    else {
		dr.x = pos_r[i  ] - pos  [i  ];
    		dr.y = pos_r[i+1] - pos  [i+1];
    		dr.z = pos_r[i+2] - pos  [i+2];
		}

    /* apply periodic boundary conditions */
    if (1==pbc_dirs.x) {
      x = - round( SPROD(dl,tbox_x) );
      dl.x += x * box_x.x;
      dl.y += x * box_x.y;
      dl.z += x * box_x.z;
      x = - round( SPROD(dr,tbox_x) );
      dr.x += x * box_x.x;
      dr.y += x * box_x.y;
      dr.z += x * box_x.z;
    }
    if (1==pbc_dirs.y) {
      x = - round( SPROD(dl,tbox_y) );
      dl.x += x * box_y.x;
      dl.y += x * box_y.y;
      dl.z += x * box_y.z;
      x = - round( SPROD(dr,tbox_y) );
      dr.x += x * box_y.x;
      dr.y += x * box_y.y;
      dr.z += x * box_y.z;
    }
    if (1==pbc_dirs.z) {
      x = - round( SPROD(dl,tbox_z) );
      dl.x += x * box_z.x;
      dl.y += x * box_z.y;
      dl.z += x * box_z.z;
      x = - round( SPROD(dr,tbox_z) );
      dr.x += x * box_z.x;
      dr.y += x * box_z.y;
      dr.z += x * box_z.z;
    }

	checkangle += dl.x * dr.x + dl.y * dr.y + dl.z * dr.z;	/* check angle of two spring vectors. */

   	norm1 += SPROD(dl, dl);
   	norm2 += SPROD(dr, dr);
	
   if ( pot_r > tot_pot_energy && tot_pot_energy >= pot_l ) {	
	pos[i  ] = dr.x;
  	pos[i+1] =dr.y;
   	pos[i+2] =dr.z;
   	}
  else if ( pot_r <= tot_pot_energy && tot_pot_energy < pot_l ) {
   	pos[i  ] = dl.x;	
  	pos[i+1] = dl.y;
   	pos[i+2]= dl.z;
  	}
  else if ( pot_r < tot_pot_energy && pot_l < tot_pot_energy ) {
  	pos[i  ] = dl.x*(tot_pot_energy-pot_r) + dr.x*(tot_pot_energy-pot_l);	
  	pos[i+1] = dl.y*(tot_pot_energy-pot_r) + dr.y*(tot_pot_energy-pot_l);
   	pos[i+2]= dl.z*(tot_pot_energy-pot_r) + dr.z*(tot_pot_energy-pot_l);
	norm3 += SPRODN(pos+i, pos+i);
  	}
  else if ( pot_r > tot_pot_energy && pot_l > tot_pot_energy ) {
  	pos[i  ] = dl.x*(pot_l-tot_pot_energy) + dr.x*(pot_r-tot_pot_energy);	
  	pos[i+1] = dl.y*(pot_l-tot_pot_energy) + dr.y*(pot_r-tot_pot_energy);
   	pos[i+2]= dl.z*(pot_l-tot_pot_energy) + dr.z*(pot_r-tot_pot_energy);
	norm4 += SPRODN(pos+i, pos+i);
  	}
  else {
  	pos[i  ] = dr.x + dl.x;
  	pos[i+1] = dr.y + dl.y;
   	pos[i+2]= dr.z + dl.z;
	norm5 += SPRODN(pos+i, pos+i);
  	}

  }

   /* check angle and norms. */
/*   if (0 == steps % neb_ke_step) {
   	printf("rank=%d, ca=%f, norm1=%f, norm2=%f\n", myrank, checkangle, norm1, norm2);		
	fflush(stdout);
   	}	*/

   norm1 = SQRT(norm1);
   norm2 = SQRT(norm2);
   norm3 = SQRT(norm3);
   norm4 = SQRT(norm4);
   norm5 = SQRT(norm5);

  if ( pot_r > tot_pot_energy && tot_pot_energy >= pot_l ) {
  	for (i=0; i<DIM*natoms; i+=DIM) {
		pos[i]=pos[i]/norm2;
		pos[i+1]=pos[i+1]/norm2;
		pos[i+2]=pos[i+2]/norm2;
  		}
  	}
  else if ( pot_r <= tot_pot_energy && tot_pot_energy < pot_l ) {
  	for (i=0; i<DIM*natoms; i+=DIM) {
		pos[i]=pos[i]/norm1;
		pos[i+1]=pos[i+1]/norm1;
		pos[i+2]=pos[i+2]/norm1;
  		}
  	}
  else if ( pot_r < tot_pot_energy && pot_l < tot_pot_energy ) {
  	for (i=0; i<DIM*natoms; i+=DIM) {
		pos[i]=pos[i]/norm3;
		pos[i+1]=pos[i+1]/norm3;
		pos[i+2]=pos[i+2]/norm3;
  		}
	/* @ Zhiming Chen @: smart climbing. */
	if (smart_climb == 1) {
		climb=2;
		climbs=0;
		}
  	}
  else if ( pot_r > tot_pot_energy && pot_l > tot_pot_energy ) {
  	for (i=0; i<DIM*natoms; i+=DIM) {
		pos[i]=pos[i]/norm4;
		pos[i+1]=pos[i+1]/norm4;
		pos[i+2]=pos[i+2]/norm4;
  		}
	/* @ Zhiming Chen @: smart climbing, but sliding here. */
	if (smart_climb == 1) {
		climb=0;
		climbs=0;
		}
  	}
  else {
  	for (i=0; i<DIM*natoms; i+=DIM) {
		pos[i]=pos[i]/norm5;
		pos[i+1]=pos[i+1]/norm5;
		pos[i+2]=pos[i+2]/norm5;
  		}
  	}

  proj_spring = neb_k * (norm2-norm1);

  for (k=0; k<NCELLS; k++) {
    cell *p = CELLPTR(k);
    for (i=0; i<p->n; i++) { 
      n = NUMMER(p,i);
      proj += pos X(n) * KRAFT(p,i,X) + pos Y(n) * KRAFT(p,i,Y) + pos Z(n) * KRAFT(p,i,Z);
    }
  }

  /* @Zhiming Chen@: check the angle between natural force and spring force. */
  checkangle2 = proj;

  for (ci=0; ci<10; ci++){
  		if (myrank == neb_climb[ci]-1) {
  			climb=2;
			climbs=0;
/*			printf("\n my rank is %d, I'm climbing image %d \n", myrank, neb_climb[ci]);		*/
  		}
		else if (myrank == neb_slide[ci]-1) {
			climb=0;
			climbs=0;
/*			printf("\n my rank is %d, I'm sliding image %d \n", myrank, neb_slide[ci]);		*/
		}
  	}
  
  tmp = -proj * climb + proj_spring * climbs;

/* @Zhiming Chen@: The Free-End method suggested by Ju Li. We'll use another way here
                              which is better. */
  if ( (neb_ofe != -1) && (neb_fe > -1) && (myrank == neb_nrep-2) ) {
	 for (k=0; k<NCELLS; k++) {
	    cell *p = CELLPTR(k);
	    for (i=0; i<p->n; i++) { 
	      f_norm += SQR(KRAFT(p, i, X)) + SQR(KRAFT(p, i, Y)) + SQR(KRAFT(p, i, Z));
	    }
	  }
	 f_norm = SQRT(f_norm);
	 for (k=0; k<NCELLS; k++) {
	      cell *p = CELLPTR(k);
	      for (i=0; i<p->n; i++) { 
		        n = NUMMER(p,i);
		        pos_r X(n) = KRAFT(p,i,X) / f_norm;
		        pos_r Y(n) = KRAFT(p,i,Y) / f_norm;
		        pos_r Z(n) = KRAFT(p,i,Z) / f_norm;
			 k_proj += (pos_r X(n) * pos X(n) + pos_r Y(n) * pos Y(n) + pos_r Z(n) * pos Z(n)) * proj_spring;
	      }
	    }	 
  	}	

  if ( (neb_fe > -1) && (myrank == neb_nrep-2) ) {
  	for (k=0; k<NCELLS; k++) {
	    cell *p = CELLPTR(k);
	    for (i=0; i<p->n; i++) { 
	      n = NUMMER(p,i);
/*		@Zhiming Chen@: The Free-End method suggested by Ju Li. 
		We'll use another way here which is better. 				*/
		if (neb_ofe != -1) {			
	      		KRAFT(p, i, X) = -k_proj * pos_r X(n) + pos X(n) * proj_spring;
	      		KRAFT(p, i, Y) = -k_proj * pos_r Y(n) + pos Y(n) * proj_spring;
	     		KRAFT(p, i, Z) = -k_proj * pos_r Z(n) + pos Z(n) * proj_spring;		
			}
		else {
	      		KRAFT(p, i, X) -= pos X(n) * proj;
	      		KRAFT(p, i, Y) -= pos Y(n) * proj;
	      		KRAFT(p, i, Z) -= pos Z(n) * proj;
			}
	    }
	  }
  	}
  else {
	  for (k=0; k<NCELLS; k++) {
	    cell *p = CELLPTR(k);
	    for (i=0; i<p->n; i++) { 
	      n = NUMMER(p,i);
	      KRAFT(p, i, X) += pos X(n) * tmp;
	      KRAFT(p, i, Y) += pos Y(n) * tmp;
	      KRAFT(p, i, Z) += pos Z(n) * tmp;
	    }
	  }
  	}
 
	if ( (neb_ke_step > 0) && (0 == steps % neb_ke_step ) ) {
		 for (k=0; k<NCELLS; k++) {
		    cell *p = CELLPTR(k);
		    for (i=0; i<p->n; i++) {
		      f_norm += SQR(KRAFT(p, i, X)) + SQR(KRAFT(p, i, Y)) + SQR(KRAFT(p, i, Z));
		    }
		  }
	f_norm = SQRT(f_norm);
	src[0] = f_norm;
 	src[1] = norm1;
	MPI_Allreduce( src, dest, 2, REAL, MPI_SUM, MPI_COMM_WORLD);
	neb_k = dest[0] / dest[1];	
	
	/* @Zhiming Chen@: check smart climbing. */
/*	if (climb==2) {printf("I'm %d, i'm climbing\n", myrank);}
	if (climb==0) {printf("I'm %d, i'm sliding\n", myrank);}
	fflush(stdout);
*/		
	if (myrank == 0) {
		printf("step %d, updated spring constant: %.1e\n", steps, neb_k);	
		fflush(stdout);	
		}	
	}
}

/******************************************************************************
*
*  write file with total fnorm, for monitoring convergence
*
******************************************************************************/

void write_neb_eng_file(int steps)
{
  static int flush_count=0;
  str255 fname;

  /* write header */
  if (steps==0) {
    sprintf(fname, "%s.eng", neb_outfilename);
    neb_eng_file = fopen(fname,"a");
    if (NULL == neb_eng_file) 
      error_str("Cannot open properties file %s", fname);
    fprintf(neb_eng_file, "# nfc fnorm\n");
  }

  /* open .eng file if not yet open */
  if (NULL == neb_eng_file) {
    sprintf(fname, "%s.eng", neb_outfilename);
    neb_eng_file = fopen(fname,"a");
    if (NULL == neb_eng_file) 
      error_str("Cannot open properties file %s.eng", outfilename);
  }

  fprintf(neb_eng_file, "%d %e\n", nfc, neb_fnorm);

  /* flush .eng file every flush_int writes */
  if (flush_count++ > flush_int) {
    fflush(neb_eng_file);
    flush_count=0;
  }
}
