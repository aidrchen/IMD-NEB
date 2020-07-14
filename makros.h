
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
* makros.h -- Some useful makros for IMD
*
******************************************************************************/

/******************************************************************************
* $Revision: 1.60 $
* $Date: 2007/11/09 16:26:03 $
******************************************************************************/

/* avoid confusion with GNU error() */
#define error(arg) imderror(arg)

#if defined(__GNUC__)
#define INLINE inline
#else
#define INLINE
#endif

/* avoid p % q, which is terribly slow */
/* on SGI, inline doesn't really work :-( */
#if !defined(MONO)
#if defined(t3e)
#pragma _CRI inline(MOD)
#elif defined(ALPHA)
#pragma inline(MOD)
#elif defined(sgi)
#pragma inline global (MOD)
#endif
INLINE static int MOD(shortint p, int q)
{
  int stmp=p;
  while (stmp>=q) stmp-=q;
  return stmp;
}
#endif

/* Sometimes we use array where we should use vectors but... */
#define X(i) [SDIM*(i)  ]
#define Y(i) [SDIM*(i)+1]
#define Z(i) [SDIM*(i)+2]
#define W(i) [SDIM*(i)+3]

#if defined(VEC) && defined(INDEXED_ACCESS)

#ifdef MONOLJ

#define SORTE(cell,i)           0
#define VSORTE(cell,i)          0
#define NUMMER(cell,i)          0
#define MASSE(cell,i)           1.0
#define POTENG(cell,i)          0.0

#else

#ifdef MONO
#define SORTE(cell,i)           0
#else
#define SORTE(cell,i)           (atoms.sorte  [(cell)->ind[i]])
#endif

#define VSORTE(cell,i)          (atoms.vsorte [(cell)->ind[i]])
#define NUMMER(cell,i)          (atoms.nummer [(cell)->ind[i]])
#define MASSE(cell,i)           (atoms.masse  [(cell)->ind[i]])
#define POTENG(cell,i)          (atoms.pot_eng[(cell)->ind[i]])

#endif

#define ORT(cell,i,sub)         (atoms.ort    sub((cell)->ind[i]))
#define KRAFT(cell,i,sub)       (atoms.kraft  sub((cell)->ind[i]))
#define IMPULS(cell,i,sub)      (atoms.impuls sub((cell)->ind[i]))

#ifdef EAM2
#define EAM_RHO(cell,i)         (atoms.eam_rho[(cell)->ind[i]])
#define EAM_DF(cell,i)          (atoms.eam_dF [(cell)->ind[i]])
#ifdef EEAM
#define EAM_P(cell,i)           (atoms.eeam_p_h[(cell)->ind[i]])
#define EAM_DM(cell,i)          (atoms.eeam_dM [(cell)->ind[i]])
#endif
#endif
#ifdef ADP
#define ADP_MU(cell,i,sub)      (atoms.adp_mu sub((cell)->ind[i]))
#define ADP_LAMBDA(cell,i,sub)  (atoms.adp_lambda[(cell)->ind[i]].sub)
#endif

#ifdef CG
#define CG_G(cell,i,sub)        (atoms.g       sub((cell)->ind[i]))
#define CG_H(cell,i,sub)        (atoms.h       sub((cell)->ind[i]))
#define OLD_ORT(cell,i,sub)     (atoms.old_ort sub((cell)->ind[i]))
#endif

#ifdef DAMP
#define DAMPF(cell,i)           (atoms.damp_f[(cell)->ind[i]])
#endif

#ifdef DISLOC
#define EPOT_REF(cell,i)        (atoms.Epot_ref   [(cell)->ind[i]])
#define ORT_REF(cell,i,sub)     (atoms.ort_ref sub((cell)->ind[i]))
#endif
#ifdef CNA
#define MARK(cell,i)            (atoms.mark[(cell)->ind[i]])
#endif
#ifdef AVPOS
#define AV_POS(cell,i,sub)      (atoms.avpos sub((cell)->ind[i]))
#define SHEET(cell,i,sub)       (atoms.sheet sub((cell)->ind[i]))
#define AV_EPOT(cell,i)         (atoms.av_epot  [(cell)->ind[i]])
#endif
#ifdef NNBR
#define NBANZ(cell,i)           (atoms.nbanz[(cell)->ind[i]])
#endif
#ifdef REFPOS
#define REF_POS(cell,i,sub)     (atoms.refpos sub((cell)->ind[i]))
#endif
#ifdef NVX
#define HEATCOND(cell,i)        (atoms.heatcond[(cell)->ind[i]])
#endif
#ifdef STRESS_TENS
#define PRESSTENS(cell,i,sub)   (atoms.presstens[(cell)->ind[i]].sub)
#endif
#ifdef SHOCK
#define PXAVG(cell,i)           (atoms.pxavg[(cell)->ind[i]])
#endif
#ifdef COVALENT
/* not supported in VEC mode */
#endif
#ifdef NBLIST
#define NBL_POS(cell,i,sub)     (atoms.nbl_pos sub((cell)->ind[i]))
#endif
#ifdef UNIAX
#define ACHSE(cell,i,sub)       (atoms.achse       sub((cell)->ind[i]))
#define DREH_IMPULS(cell,i,sub) (atoms.dreh_impuls sub((cell)->ind[i]))
#define DREH_MOMENT(cell,i,sub) (atoms.dreh_moment sub((cell)->ind[i]))
#endif

#else /* not VEC or direct access */

#ifdef MONOLJ

#define SORTE(cell,i)           0
#define VSORTE(cell,i)          0
#define NUMMER(cell,i)          0
#define MASSE(cell,i)           1.0
#define POTENG(cell,i)          0.0

#else

#if defined(MONO)
#define SORTE(cell,i)           0
#else
#define SORTE(cell,i)           ((cell)->sorte[i])
#endif

#define VSORTE(cell,i)          ((cell)->vsorte[i])
#define NUMMER(cell,i)          ((cell)->nummer[i])
#define MASSE(cell,i)           ((cell)->masse[i])
#ifdef CBE_DIRECT
#define POTENG(cell,i)          ((cell)->kraft W(i))
#else
#define POTENG(cell,i)          ((cell)->pot_eng[i])
#endif

#endif

#define ORT(cell,i,sub)         ((cell)->ort sub(i))
#define KRAFT(cell,i,sub)       ((cell)->kraft sub(i))
#define IMPULS(cell,i,sub)      ((cell)->impuls sub(i))

#ifdef EAM2
#define EAM_RHO(cell,i)         ((cell)->eam_rho[i])
#define EAM_DF(cell,i)          ((cell)->eam_dF [i])
#ifdef EEAM
#define EAM_P(cell,i)           ((cell)->eeam_p_h[i])
#define EAM_DM(cell,i)          ((cell)->eeam_dM [i])
#endif
#endif

#ifdef DAMP
#define DAMPF(cell,i)           ((cell)->damp_f[i])
#endif

#ifdef ADP
#define ADP_MU(cell,i,sub)      ((cell)->adp_mu sub(i))
#define ADP_LAMBDA(cell,i,sub)  ((cell)->adp_lambda[i].sub)
#endif

#ifdef CG
#define CG_G(cell,i,sub)        ((cell)->g sub(i))
#define CG_H(cell,i,sub)        ((cell)->h sub(i))
#define OLD_ORT(cell,i,sub)     ((cell)->old_ort sub(i))
#endif
#ifdef DISLOC
#define EPOT_REF(cell,i)        ((cell)->Epot_ref[i])
#define ORT_REF(cell,i,sub)     ((cell)->ort_ref sub(i))
#endif
#ifdef CNA
#define MARK(cell,i)            ((cell)->mark[i])
#endif
#ifdef AVPOS
#define AV_POS(cell,i,sub)      ((cell)->avpos sub(i))
#define SHEET(cell,i,sub)       ((cell)->sheet sub(i))
#define AV_EPOT(cell,i)         ((cell)->av_epot[i])
#endif
#ifdef NNBR
#define NBANZ(cell,i)           (cell)->nbanz[(i)]
#endif
#ifdef REFPOS
#define REF_POS(cell,i,sub)     ((cell)->refpos sub(i))
#endif
#ifdef NVX
#define HEATCOND(cell,i)        ((cell)->heatcond[i])
#endif
#ifdef STRESS_TENS
#define PRESSTENS(cell,i,sub)   ((cell)->presstens[i].sub)
#endif
#ifdef SHOCK
#define PXAVG(cell,i)           ((cell)->pxavg[i])
#endif
#ifdef COVALENT
#define NEIGH(cell,i)           ((cell)->neigh[i])
#define NSORTE(neigh,i)         ((neigh)->typ[i])
#define NZELLE(neigh,i)         ((cell *) (neigh)->cl[i])
#define NNUMMER(neigh,i)        ((neigh)->num[i])
#endif
#ifdef NBLIST
#define NBL_POS(cell,i,sub)     ((cell)->nbl_pos sub(i))
#endif
#ifdef UNIAX
#define ACHSE(cell,i,sub)       ((cell)->achse sub(i))
#define DREH_IMPULS(cell,i,sub) ((cell)->dreh_impuls sub(i))
#define DREH_MOMENT(cell,i,sub) ((cell)->dreh_moment sub(i))
#endif

#endif /* VEC */

#ifdef BUFCELLS
#define CELLS(k) cells[k]
#else
#define CELLS(k) k
#endif

#if !defined(VEC) || defined(INDEXED_ACCESS)
#define CELLPTR(k) (cell_array + CELLS(k))
#define NCELLS ncells
#else
#define CELLPTR(k) (&atoms)
#define NCELLS 1
#endif

#ifdef VEC
#define MOVE_ATOM      move_atom_mini
#define INSERT_ATOM    insert_atom
#define ALLOC_MINICELL alloc_minicell
#else
#define MOVE_ATOM      move_atom
#define INSERT_ATOM    move_atom
#define ALLOC_MINICELL alloc_cell
#endif

/* Max gibt den groesseren von zwei Werten */
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* Min gibt den kleineren  von zwei Werten */
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* Sqr quadriert sein Argument */
#if defined(__GNUC__) || defined(__SASC)
inline static real SQR(real x)
{
  return x*x;
}
#else
#define SQR(a) ((a)*(a))
#endif

/* Abs berechnet den Betrag einer Zahl */
#define ABS(a) ((a) >0 ? (a) : -(a))

/* How many dimension are there? */
#ifdef TWOD
#define SDIM 2
#define  DIM 2
#elif defined(CBE_DIRECT)
#define SDIM 4
#define  DIM 3
#else
#define SDIM 3
#define  DIM 3
#endif

#ifdef MEAM
#define  I(a,b) [(((b)*(neigh_len)) + (a))] 
#define IX(a,b) [(((b)*(neigh_len)) + (a))].x
#define IY(a,b) [(((b)*(neigh_len)) + (a))].y
#define IZ(a,b) [(((b)*(neigh_len)) + (a))].z
#endif

/* Skalarprodukt */
/* Vectors */
#define SPROD3D(a,b) (((a).x * (b).x) + ((a).y * (b).y) + ((a).z * (b).z))
#define SPROD2D(a,b) (((a).x * (b).x) + ((a).y * (b).y))
/* Arrays */
#define SPRODN3D(a,b) \
  (((a)[0] * (b)[0]) + ((a)[1] * (b)[1]) + ((a)[2] * (b)[2]))
#define SPRODN2D(a,b) (((a)[0] * (b)[0]) + ((a)[1] * (b)[1]))
/* Mixed Arrray, Vector */
#define SPRODX3D(a,b) \
   (((a)[0] * (b).x) + ((a)[1] * (b).y) + ((a)[2] * (b).z))
#define SPRODX2D(a,b) (((a)[0] * (b).x) + ((a)[1] * (b).y))
                           

#ifdef TWOD
#define SPROD(a,b)         SPROD2D(a,b)
#define SPRODN(a,b)        SPRODN2D(a,b)
#define SPRODX(a,b)        SPRODX2D(a,b)
#else
#define SPROD(a,b)         SPROD3D(a,b)
#define SPRODN(a,b)        SPRODN3D(a,b)
#define SPRODX(a,b)        SPRODX3D(a,b)
#endif

#ifdef DOUBLE
#define round(a)		rint(a)
#else
#define round(a)		rintf(a)
#endif

/* Dynamically allocated 3D array -- sort of */
#define PTR_3D(var,i,j,k,dim_i,dim_j,dim_k) \
  (((var) + ((i)*(dim_j)*(dim_k)) + ((j)*(dim_k)) + (k)))

/* Dynamically allocated 3D array -- half vector version */
#define PTR_3D_V(var,i,j,k,dim) \
  (((var) + ((i)*(dim.y)*(dim.z)) + ((j)*(dim.z)) + (k)))

/* Dynamically allocated 3D array -- full vector version */
#define PTR_3D_VV(var,coord,dim) \
  (((var) + ((coord.x)*(dim.y)*(dim.z)) + ((coord.y)*(dim.z)) + (coord.z)))

/* Dynamically allocated 2D array -- sort of */
#define PTR_2D(var,i,j,dim_i,dim_j) \
  (((var) + ((i)*(dim_j)) + (j)))

/* Dynamically allocated 2D array -- half vector version */
#define PTR_2D_V(var,i,j,dim) \
  (((var) + ((i)*(dim.y)) + (j)))

/* Dynamically allocated 2D array -- full vector version */
#define PTR_2D_VV(var,coord,dim) \
  (((var) + ((coord.x)*(dim.y)) + (coord.y)))

#ifdef TWOD
#define PTR     PTR_2D
#define PTR_V   PTR_2D_V
#define PTR_VV  PTR_2D_VV
#else
#define PTR     PTR_3D
#define PTR_V   PTR_3D_V
#define PTR_VV  PTR_3D_VV
#endif

/* different versions of math functions for float and double */
#ifdef DOUBLE
#define FLOOR floor
#define FABS  fabs
#define SQRT  sqrt
#else
#define FLOOR floorf
#define FABS  fabsf
#define SQRT  sqrtf
#endif

#ifndef M_PI
#define M_PI (4.0*atan(1.0))
#endif
