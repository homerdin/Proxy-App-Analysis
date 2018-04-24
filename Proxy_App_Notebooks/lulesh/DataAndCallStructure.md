
# lulesh

## Data Structure showing attributes
lulesh builds and uses a domain object with many data members for most of its calls.  Included here:
```c++
class Domain {
104 //////////////////////////////////////////////////////
105 // Primary data structure
106 //////////////////////////////////////////////////////
107 
108 /*
109  * The implementation of the data abstraction used for lulesh
110  * resides entirely in the Domain class below.  You can change
111  * grouping and interleaving of fields here to maximize data layout
112  * efficiency for your underlying architecture or compiler.
113  *
114  * For example, fields can be implemented as STL objects or
115  * raw array pointers.  As another example, individual fields
116  * m_x, m_y, m_z could be budled into
117  *
118  *    struct { Real_t x, y, z ; } *m_coord ;
119  *
120  * allowing accessor functions such as
121  *
122  *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
123  *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
124  *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
125  */
126 
127 class Domain {
{...}
426    //
427    // IMPLEMENTATION
428    //
429 
430    /* Node-centered */
431    std::vector<Real_t> m_x ;  /* coordinates */
432    std::vector<Real_t> m_y ;
433    std::vector<Real_t> m_z ;
434 
435    std::vector<Real_t> m_xd ; /* velocities */
436    std::vector<Real_t> m_yd ;
437    std::vector<Real_t> m_zd ;
438 
439    std::vector<Real_t> m_xdd ; /* accelerations */
440    std::vector<Real_t> m_ydd ;
441    std::vector<Real_t> m_zdd ;
442 
443    std::vector<Real_t> m_fx ;  /* forces */
444    std::vector<Real_t> m_fy ;
445    std::vector<Real_t> m_fz ;
446 
447    std::vector<Real_t> m_nodalMass ;  /* mass */
448 
449    std::vector<Index_t> m_symmX ;  /* symmetry plane nodesets */
450    std::vector<Index_t> m_symmY ;
451    std::vector<Index_t> m_symmZ ;
452 
453    // Element-centered
454 
455    // Region information
456    Int_t    m_numReg ;
457    Int_t    m_cost; //imbalance cost
458    Index_t *m_regElemSize ;   // Size of region sets
459    Index_t *m_regNumList ;    // Region number per domain element
460    Index_t **m_regElemlist ;  // region indexset 
461 
462    std::vector<Index_t>  m_nodelist ;     /* elemToNode connectivity */
463 
464    std::vector<Index_t>  m_lxim ;  /* element connectivity across each face */
465    std::vector<Index_t>  m_lxip ;
466    std::vector<Index_t>  m_letam ;
467    std::vector<Index_t>  m_letap ;
468    std::vector<Index_t>  m_lzetam ;
469    std::vector<Index_t>  m_lzetap ;
470 
471    std::vector<Int_t>    m_elemBC ;  /* symmetry/free-surface flags for each elem face */
472 
473    std::vector<Real_t> m_dxx ;  /* principal strains -- temporary */
474    std::vector<Real_t> m_dyy ;
475    std::vector<Real_t> m_dzz ;
476 
477    std::vector<Real_t> m_delv_xi ;    /* velocity gradient -- temporary */
478    std::vector<Real_t> m_delv_eta ;
479    std::vector<Real_t> m_delv_zeta ;
480 
481    std::vector<Real_t> m_delx_xi ;    /* coordinate gradient -- temporary */
482    std::vector<Real_t> m_delx_eta ;
483    std::vector<Real_t> m_delx_zeta ;
484 
485    std::vector<Real_t> m_e ;   /* energy */
486 
487    std::vector<Real_t> m_p ;   /* pressure */
488    std::vector<Real_t> m_q ;   /* q */
489    std::vector<Real_t> m_ql ;  /* linear term for q */
490    std::vector<Real_t> m_qq ;  /* quadratic term for q */
491 
492    std::vector<Real_t> m_v ;     /* relative volume */
493    std::vector<Real_t> m_volo ;  /* reference volume */
494    std::vector<Real_t> m_vnew ;  /* new relative volume -- temporary */
495    std::vector<Real_t> m_delv ;  /* m_vnew - m_v */
496    std::vector<Real_t> m_vdov ;  /* volume derivative over volume */
497 
498    std::vector<Real_t> m_arealg ;  /* characteristic length of an element */
499 
500    std::vector<Real_t> m_ss ;      /* "sound speed" */
501 
502    std::vector<Real_t> m_elemMass ;  /* mass */
503 
504    // Cutoffs (treat as constants)
505    const Real_t  m_e_cut ;             // energy tolerance 
506    const Real_t  m_p_cut ;             // pressure tolerance 
507    const Real_t  m_q_cut ;             // q tolerance 
508    const Real_t  m_v_cut ;             // relative volume tolerance 
509    const Real_t  m_u_cut ;             // velocity tolerance 
510 
511    // Other constants (usually setable, but hardcoded in this proxy app)
512 
513    const Real_t  m_hgcoef ;            // hourglass control 
514    const Real_t  m_ss4o3 ;
515    const Real_t  m_qstop ;             // excessive q indicator 
516    const Real_t  m_monoq_max_slope ;
517    const Real_t  m_monoq_limiter_mult ;
518    const Real_t  m_qlc_monoq ;         // linear term coef for q 
519    const Real_t  m_qqc_monoq ;         // quadratic term coef for q 
520    const Real_t  m_qqc ;
521    const Real_t  m_eosvmax ;
522    const Real_t  m_eosvmin ;
523    const Real_t  m_pmin ;              // pressure floor 
524    const Real_t  m_emin ;              // energy floor 
525    const Real_t  m_dvovmax ;           // maximum allowable volume change 
526    const Real_t  m_refdens ;           // reference density 
527 
528    // Variables to keep track of timestep, simulation time, and cycle
529    Real_t  m_dtcourant ;         // courant constraint 
530    Real_t  m_dthydro ;           // volume change constraint 
531    Int_t   m_cycle ;             // iteration count for simulation 
532    Real_t  m_dtfixed ;           // fixed time increment 
533    Real_t  m_time ;              // current time 
534    Real_t  m_deltatime ;         // variable time increment 
535    Real_t  m_deltatimemultlb ;
536    Real_t  m_deltatimemultub ;
537    Real_t  m_dtmax ;             // maximum allowable time increment 
538    Real_t  m_stoptime ;          // end time for simulation 
539 
540 
541    Int_t   m_numRanks ;
542 
543    Index_t m_colLoc ;
544    Index_t m_rowLoc ;
545    Index_t m_planeLoc ;
546    Index_t m_tp ;
547 
548    Index_t m_sizeX ;
549    Index_t m_sizeY ;
550    Index_t m_sizeZ ;
551    Index_t m_numElem ;
552    Index_t m_numNode ;
553 
554    Index_t m_maxPlaneSize ;
555    Index_t m_maxEdgeSize ;
556 
557    // OMP hack 
558    Index_t *m_nodeElemStart ;
559    Index_t *m_nodeElemCornerList ;
560 
561    // Used in setup
562    Index_t m_rowMin, m_rowMax;
563    Index_t m_colMin, m_colMax;
564    Index_t m_planeMin, m_planeMax ;
565 
566 } ;
```

---
## Call Structure for significant cycles (Stopped at inlined fuctions) (Lot in Loops not shown)
```c
Main
-> loop in main
--> LagrangeLeapFrog
---> LagrangeNodal
----> CalcForceForNodes
-----> CalcVolumeForceForElems
------> IntegrateStressForElems
-------> fork
--------> [I] CollectDomainNodesToElemNodes
--------> [I] CalcElemShapeFunctionDerivatives
--------> [I] SumElemStressesToNodeForces
      <--
------> CalcHourglassControlForElems
-------> fork
--------> [I] CollectDomainNodesToElemNodes
--------> [I] CalcElemVolumeDerivative
       <-
-------> CalcFBHourglassForceForElems
--------> fork
---------> CalcElemFBHourglassForce
   <------
---> LagrangeElements
----> CalcLagrangeElements
-----> CalcKinematicsForElems
------> fork
-------> [I] CollectDomainNodesToElemNodes
-------> [I] CalcElemCharacteristicLength
-------> [I] CalcElemShapeFunctionDerivatives
-------> [I] CalcElemVelocityGradient
    <---
----> CalcQForElems
-----> CalcMonotonicQGradientsForElems
------> fork
     <-
-----> CalcMonotonicQForElems
------> CalcMonotonicQRegionForElems
-------> fork
    <---
----> ApplyMaterialPropertiesForElems
-----> EvalEOSForElems
------> fork
     <-
------> CalcEnergyForElems
-------> fork
       <
-------> CalcPressureForElems
-------> fork
       <
-------> fork
       <
-------> CalcPressureForElems
-------> fork
       <
       
 <------      
-> loop in main 
```
