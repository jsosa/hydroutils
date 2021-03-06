C===================================================== XTEST.FOR
      PROGRAM XTEST
C***********************************************************************
C*                                                                     *
C*  FORTRAN CODE WRITTEN FOR INCLUSION IN IBM RESEARCH REPORT RC20525, *
C*  'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3' *
C*                                                                     *
C*  J. R. M. HOSKING                                                   *
C*  IBM RESEARCH DIVISION                                              *
C*  T. J. WATSON RESEARCH CENTER                                       *
C*  YORKTOWN HEIGHTS                                                   *
C*  NEW YORK 10598, U.S.A.                                             *
C*                                                                     *
C*  VERSION 3     AUGUST 1996                                          *
C*                                                                     *
C*  VERSION 3.03  JUNE 2000                                            *
C*  * CHARACTER variable declarations changed to conform with          *
C*    Fortran-77 standard                                              *
C*                                                                     *
C***********************************************************************
C
C  EXAMPLE PROGRAM TO ILLUSTRATE THE USE OF ROUTINE REGTST
C
C  THE ROUTINE READS IN THE SAMPLE L-MOMENTS FOR DIFFERENT SITES IN A
C  REGION AND CALLS ROUTINE REGTST TO COMPUTE DISCORDANCY, HETEROGENEITY
C  AND GOODNESS-OF-FIT STATISTICS.
C
C  PARAMETERS OF PROGRAM:
C  MAXNS  - MAXIMUM NUMBER OF SITES
C  SSEED  - SEED FOR RANDOM-NUMBER GENERATOR
C  NSIM   - NSIM PARAMETER OF ROUTINE REGTST
C  KPRINT - OUTPUT FLAG, KPRINT PARAMETER OF ROUTINE REGTST
C  INFILE - STREAM NUMBER TO WHICH INPUT FILE IS ATTACHED
C  KOUT   - STREAM NUMBER TO WHICH OUTPUT FILE IS ATTACHED
C  NPROB  - NUMBER OF QUANTILES TO BE FOUND
C  A      - ) PARAMETERS OF
C  B      - ) PLOTTING POSITION
C
C  ARRAYS TO BE INITIALIZED IN DATA STATEMENTS:
C  PROB   - PROBABILITIES FOR WHICH QUANTILES ARE TO BE FOUND
C
C  ROUTINES USED: REGTST AND ROUTINES CALLED BY REGTST
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (SSEED=619145091D0,NSIM=500,MAXNS=200)
      PARAMETER (KPRINT=1,INFILE=7,KOUT=6,NPROB=10)
C
      CHARACTER*12 NAMES(MAXNS)
      CHARACTER*60 REGNAM
      DOUBLE PRECISION D(MAXNS),H(3),PARA(5,6),PROB(NPROB),RMOM(5),
     *  VBAR(3),VOBS(3),VSD(3),XMOM(5,MAXNS),Z(5)
      INTEGER LEN(MAXNS)
C
      DATA A/0D0/,B/0D0/
      DATA PROB/0.01D0,0.02D0,0.05D0,0.1D0,0.2D0,
     *  0.5D0,0.90D0,0.95D0,0.99D0,0.999D0/
C
C         READ IN THE AT-SITE L-MOMENTS.
C         DATA FILE MAY CONTAIN ANY NUMBER OF REGIONAL DATA STRUCTURES.
C         A 'REGIONAL DATA STRUCTURE' CONSISTS OF:
C         1. ONE RECORD CONTAINING:
C            (COLUMNS  1- 4) NUMBER OF SITES IN REGION;
C            (COLUMNS  5-56) IDENTIFYING LABEL FOR THE REGION.
C         2. FOR EACH SITE, ONE RECORD CONTAINING:
C            (COLUMNS  1-12) AN IDENTIFYING LABEL FOR THE SITE;
C            (COLUMNS 13-16) THE RECORD LENGTH AT THE SITE;
C            (COLUMNS 17-56) SAMPLE L-MOMENTS L-1, L-CV, T-3, T-4, T-5.
C
    1 READ(INFILE,5000,END=900)NSITES,REGNAM
      WRITE(KOUT,6000)REGNAM,NSITES
      DO 10 ISITE=1,NSITES
      READ(INFILE,5010)NAMES(ISITE),LEN(ISITE),(XMOM(I,ISITE),I=1,5)
   10 CONTINUE
C
C         CALCULATE TEST STATISTICS
C
      SEED=SSEED
      CALL REGTST(NSITES,NAMES,LEN,XMOM,A,B,SEED,NSIM,NPROB,PROB,
     *  KPRINT,KOUT,RMOM,D,VOBS,VBAR,VSD,H,Z,PARA)
      GOTO 1
C
  900 CONTINUE
      WRITE(KOUT,6010)
      STOP
C
 5000 FORMAT(I4,A52)
 5010 FORMAT(A12,I4,5F8.0)
 6000 FORMAT(///1X,A52,I8,' SITES')
 6010 FORMAT(///' ALL DATA PROCESSED')
      END