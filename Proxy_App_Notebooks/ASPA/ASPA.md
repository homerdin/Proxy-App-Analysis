
## ASPA
The purpose of ASPA (__A__daptive __S__ampling
__P__roxy __A__pplication) is to enable the evaluation
of a technique known as **adaptive sampling** on advanced computer
architectures.  Adaptive sampling is of interest in simulations
involving multiple physical scales, wherein models of individual
scales are combined using some form of **scale bridging**.
  
Adaptive sampling [Barton2008,Knap2008] attempts to significantly reduce the number of fine-scale evaluations by dynamically constructing a database of fine-scale evaluations and interpolation models.  When the response of the fine-scale model is needed at a new point, the database is
searched for interpolation models centered at 'nearby' points.
Assuming that the interpolation models possess error estimators, they
can be evaluated to determine if the fine-scale response at the
current query point can be obtained to sufficient accuracy simply by
interpolation from previously known states.  If not, the fine-scale
model must be evaluated and the new input/response pair added to the
closest interpolation model.  


---
## Parameters
```
Compiler = icpc (ICC) 18.0.1 20171018
Build_Flags = -g -O3 -march=native -std=c++0x -llapack -lblas
Run_Parameters = point_data.txt value_data.txt 
```

---
## Run on 1 Thread 1 Node.

---
## Roofline  -  Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz
### 1 Threads - 1 - Cores 2300.0 Mhz
|     GB/sec     |  L1 B/W |  L2 B/W |  L3 B/W | DRAM B/W |
|:---------------|:-------:|:-------:|:-------:|:--------:|
|**1 Thread**  | 143.10 |  44.87 | 33.12 |   16.04  |

---
ASPA makes use of a database (**M-tree database**), however the sparse data interpolation (known as **kringing**) takes the vast of the application cycles.  The I/O involved is not significant.  

The code that is performing the bulk of the work (81.1%) is in the BLAS library within the following kernels:

|BLAS Function| % Cycles |
|:------------|:--------:|
|dgemv        |  59.4%   |
|dgemm        |  15.3%   |
|dtrsm        |   5.2%   |
|ddot         |   1.1%   |

---
## Intel Software Development Emulator
| SDE Metrics | ASPA|
|:-----------|:---:|
| **Arithmetic Intensity** | 0.06 |
| **Bytes per Load Inst** | 7.89 |
| **Bytes per Store Inst** | 8.28 |


---
### Experiment Aggregate Metrics

| IPC per Core | Loads per Cycle | L1 Hits per Cycle |  L1 Miss Ratio | L2 Miss Ratio | L3 Miss Ratio | L2 B/W Utilized | L3 B/W Utilized |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2.63|0.91|1.06|2.77%|4.61%|6.55%|9.60%|1.14%|

---
### BLAS performance (Not interesting part of Proxy)

### DGEMV (Level 2)

| IPC per Core | Loads per Cycle | L1 Hits per Cycle |  L1 Miss Ratio | L2 Miss Ratio | L3 Miss Ratio | L2 B/W Utilized | L3 B/W Utilized |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2.57|1.03|1.02|3.36%|2.02%|8.45%|11.39%|0.48%|

### DGEMM (Level 3)

| IPC per Core | Loads per Cycle | L1 Hits per Cycle |  L1 Miss Ratio | L2 Miss Ratio | L3 Miss Ratio | L2 B/W Utilized | L3 B/W Utilized |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|3.90|1.05|1.54|0.43%|6.71%|12.41%|2.46%|0.74%|

---
## Discussion
This Proxy App is more interesting as a model for using adaptive sampling to enable multi-level physics than as a hardware workload.  The interpolation model reaches hardware limitations within libblas. Code is not being vectorized.

---
### Additional Investigation
Look to see if the I/O becomes interesting with MPI Scaling.  
Parent App SAMRAI? (Some code was pulled from this)
