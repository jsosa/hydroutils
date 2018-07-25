C===================================================== CLUKM.FOR
      SUBROUTINE CLUKM(X,NX,N,NATT,NCLUST,IASSGN,LIST,NUM,SS,MAXIT,
     *  IWORK,RW,NW)
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
C***********************************************************************
C
C  CLUSTER ANALYSIS BY THE K-MEANS ALGORITHM
C
C  PARAMETERS OF ROUTINE:
C  X      * INPUT* ARRAY OF DIMENSION (NX,NATT).  X(I,J) SHOULD
C                  CONTAIN THE J'TH ATTRIBUTE FOR THE I'TH DATA POINT.
C  NX     * INPUT* THE FIRST DIMENSION OF ARRAY X, AS DECLARED IN THE
C                  CALLING PROGRAM.
C  N      * INPUT* NUMBER OF DATA POINTS
C  NATT   * INPUT* NUMBER OF ATTRIBUTES FOR EACH DATA POINT
C  NCLUST * INPUT* NUMBER OF CLUSTERS
C  IASSGN *IN/OUT* ARRAY OF LENGTH N.  ON ENTRY, SHOULD CONTAIN THE
C                  INITIAL ASSIGNMENT OF SITES TO CLUSTERS.  ON EXIT,
C                  CONTAINS THE FINAL ASSIGNMENT.  THE I'TH ELEMENT OF
C                  THE ARRAY CONTAINS THE LABEL OF THE CLUSTER TO WHICH
C                  THE I'TH DATA POINT BELONGS.  LABELS MUST BE BETWEEN
C                  1 AND NCLUST, AND EACH OF THE VALUES 1 THROUGH NCLUST
C                  MUST OCCUR AT LEAST ONCE.
C  LIST   *OUTPUT* ARRAY OF LENGTH N. CONTAINS THE DATA POINTS IN
C                  CLUSTER 1, FOLLOWED BY THE DATA POINTS IN CLUSTER 2,
C                  ETC.  DATA POINTS IN EACH CLUSTER ARE LISTED IN
C                  INCREASING ORDER.  THE LAST DATA POINT IN EACH
C                  CLUSTER IS INDICATED BY A NEGATIVE NUMBER.
C  NUM    *OUTPUT* ARRAY OF LENGTH NCLUST.  NUMBER OF DATA POINTS IN
C                  EACH CLUSTER.
C  SS     *OUTPUT* WITHIN-GROUP SUM OF SQUARES OF THE FINAL CLUSTERS.
C  MAXIT  * INPUT* MAXIMUM NUMBER OF ITERATIONS FOR THE K-MEANS
C                  CLUSTERING ALGORITHM
C  IWORK  * LOCAL* (INTEGER) WORK ARRAY OF LENGTH NCLUST*3
C  RW     * LOCAL* REAL WORK ARRAY OF LENGTH NW.  N.B. THIS ARRAY IS OF
C                  TYPE REAL, NOT DOUBLE PRECISION!
C  NW     * INPUT* LENGTH OF ARRAY RW.  MUST BE AT LEAST
C                  (N+NCLUST)*(NATT+1)+2*NCLUST
C
C  OTHER ROUTINES USED: APPLIED STATISTICS ALGORITHM AS136 (ROUTINES
C                       KMNS,OPTRA,QTRAN), AVAILABLE FROM
C                       HTTP://STAT.LIB.CMU.EDU/APSTAT/136
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION X(NX,NATT)
      INTEGER IASSGN(N),LIST(N),NUM(NCLUST),IWORK(NCLUST,3)
      REAL RW(NW)
      DATA ZERO/0D0/
C
C         SET ADDRESSES FOR SUBDIVISION OF WORK ARRAY
C
      MC=1
      MA=MC+NCLUST*NATT
      MAN1=MA+N*NATT
      MAN2=MAN1+NCLUST
      MWSS=MAN2+NCLUST
      MD=MWSS+NCLUST
      NWREQ=MD+N-1
      IF(NW.LT.NWREQ)GOTO 1000
      LA=MA-1
      LWSS=MWSS-1
C
C         COPY ATTRIBUTES TO WORK ARRAY
C
      IW=LA
      DO 5 IATT=1,NATT
      DO 5 I=1,N
      IW=IW+1
    5 RW(IW)=X(I,IATT)
C
C         COMPUTE CLUSTER CENTERS
C
      DO 10 ICL=1,NCLUST
   10 NUM(ICL)=0
      IWMAX=NCLUST*NATT
      DO 20 IW=1,IWMAX
   20 RW(IW)=ZERO
      DO 40 I=1,N
      ICL=IASSGN(I)
      IF(ICL.LE.0.OR.ICL.GT.NCLUST)GOTO 1010
      NUM(ICL)=NUM(ICL)+1
      IW=ICL
      DO 30 IATT=1,NATT
      RW(IW)=RW(IW)+X(I,IATT)
      IW=IW+NCLUST
   30 CONTINUE
   40 CONTINUE
      DO 60 ICL=1,NCLUST
      NSIZE=NUM(ICL)
      IF(NSIZE.EQ.0)GOTO 1020
      IW=ICL
      DO 50 IATT=1,NATT
      RW(IW)=RW(IW)/NSIZE
      IW=IW+NCLUST
   50 CONTINUE
   60 CONTINUE
C
C         CALL ALGORITHM AS136
C
      CALL KMNS(RW(MA),N,NATT,RW(MC),NCLUST,IASSGN,LIST,NUM,RW(MAN1),
     *  RW(MAN2),IWORK(1,1),RW(MD),IWORK(1,2),IWORK(1,3),MAXIT,RW(MWSS),
     *  IFAULT)
      IF(IFAULT.EQ.2)WRITE(6,7030)
C
C         COMPUTE LIST ARRAY AND FINAL SUM OF SQUARES
C
      I=0
      DO 80 ICL=1,NCLUST
      DO 70 K=1,N
      IF(IASSGN(K).NE.ICL)GOTO 70
      I=I+1
      LIST(I)=K
   70 CONTINUE
      LIST(I)=-LIST(I)
   80 CONTINUE
      SS=ZERO
      DO 90 ICL=1,NCLUST
   90 SS=SS+RW(LWSS+ICL)
C
      RETURN
C
 1000 WRITE(6,7000)NWREQ
      RETURN
 1010 WRITE(6,7010)I
      RETURN
 1020 WRITE(6,7020)ICL
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE CLUKM  : INSUFFICIENT WORKSPACE.',
     *  ' LENGTH OF WORK ARRAY SHOULD BE AT LEAST ',I8)
 7010 FORMAT(' *** ERROR *** ROUTINE CLUKM  :',
     *  ' INVALID INITIAL CLUSTER NUMBER FOR DATA POINT ',I5)
 7020 FORMAT(' *** ERROR *** ROUTINE CLUKM  :',
     *  ' INITIAL CLUSTERS INVALID.  CLUSTER ',I4,' HAS NO MEMBERS.')
 7030 FORMAT(' ** WARNING ** ROUTINE CLUKM  :',
     *  ' ITERATION HAS NOT CONVERGED. RESULTS MAY BE UNRELIABLE.')
C
      END

      SUBROUTINE KMNS(A, M, N, C, K, IC1, IC2, NC, AN1, AN2, NCP, D,
     *    ITRAN, LIVE, ITER, WSS, IFAULT)
C
C     ALGORITHM AS 136  APPL. STATIST. (1979) VOL.28, NO.1
C
C     Divide M points in N-dimensional space into K clusters so that
C     the within cluster sum of squares is minimized.
C
      INTEGER IC1(M), IC2(M), NC(K), NCP(K), ITRAN(K), LIVE(K)
      REAL    A(M,N), D(M), C(K,N), AN1(K), AN2(K), WSS(K), DT(2)
      REAL    ZERO, ONE
C
C     Define BIG to be a very large positive number
C
      DATA BIG /1.E30/, ZERO /0.0/, ONE /1.0/
C
      IFAULT = 3
      IF (K .LE. 1 .OR. K .GE. M) RETURN
C
C     For each point I, find its two closest centres, IC1(I) and
C     IC2(I).     Assign it to IC1(I).
C
      DO 50 I = 1, M
      IC1(I) = 1
      IC2(I) = 2
      DO 10 IL = 1, 2
        DT(IL) = ZERO
        DO 10 J = 1, N
          DA = A(I,J) - C(IL,J)
          DT(IL) = DT(IL) + DA*DA
   10   CONTINUE
      IF (DT(1) .GT. DT(2)) THEN
        IC1(I) = 2
        IC2(I) = 1
        TEMP = DT(1)
        DT(1) = DT(2)
        DT(2) = TEMP
      END IF
      DO 50 L = 3, K
        DB = ZERO
        DO 30 J = 1, N
          DC = A(I,J) - C(L,J)
          DB = DB + DC*DC
          IF (DB .GE. DT(2)) GO TO 50
   30     CONTINUE
        IF (DB .LT. DT(1)) GO TO 40
        DT(2) = DB
        IC2(I) = L
        GO TO 50
   40     DT(2) = DT(1)
        IC2(I) = IC1(I)
        DT(1) = DB
        IC1(I) = L
   50 CONTINUE
C
C     Update cluster centres to be the average of points contained
C     within them.
C
      DO 70 L = 1, K
      NC(L) = 0
      DO 60 J = 1, N
   60   C(L,J) = ZERO
   70 CONTINUE
      DO 90 I = 1, M
      L = IC1(I)
      NC(L) = NC(L) + 1
      DO 80 J = 1, N
   80   C(L,J) = C(L,J) + A(I,J)
   90 CONTINUE
C
C     Check to see if there is any empty cluster at this stage
C
      DO 120 L = 1, K
      IF (NC(L) .EQ. 0) THEN
        IFAULT = 1
        RETURN
      END IF
      AA = NC(L)
      DO 110 J = 1, N
  110   C(L,J) = C(L,J) / AA
C
C     Initialize AN1, AN2, ITRAN & NCP
C     AN1(L) = NC(L) / (NC(L) - 1)
C     AN2(L) = NC(L) / (NC(L) + 1)
C     ITRAN(L) = 1 if cluster L is updated in the quick-transfer stage,
C              = 0 otherwise
C     In the optimal-transfer stage, NCP(L) stores the step at which
C     cluster L is last updated.
C     In the quick-transfer stage, NCP(L) stores the step at which
C     cluster L is last updated plus M.
C
      AN2(L) = AA / (AA + ONE)
      AN1(L) = BIG
      IF (AA .GT. ONE) AN1(L) = AA / (AA - ONE)
      ITRAN(L) = 1
      NCP(L) = -1
  120 CONTINUE
      INDX = 0
      DO 140 IJ = 1, ITER
C
C     In this stage, there is only one pass through the data.   Each
C     point is re-allocated, if necessary, to the cluster that will
C     induce the maximum reduction in within-cluster sum of squares.
C
      CALL OPTRA(A, M, N, C, K, IC1, IC2, NC, AN1, AN2, NCP, D,
     *        ITRAN, LIVE, INDX)
C
C     Stop if no transfer took place in the last M optimal transfer
C     steps.
C
      IF (INDX .EQ. M) GO TO 150
C
C     Each point is tested in turn to see if it should be re-allocated
C     to the cluster to which it is most likely to be transferred,
C     IC2(I), from its present cluster, IC1(I).   Loop through the
C     data until no further change is to take place.
C
      CALL QTRAN(A, M, N, C, K, IC1, IC2, NC, AN1, AN2, NCP, D,
     *       ITRAN, INDX)
C
C     If there are only two clusters, there is no need to re-enter the
C     optimal transfer stage.
C
      IF (K .EQ. 2) GO TO 150
C
C     NCP has to be set to 0 before entering OPTRA.
C
      DO 130 L = 1, K
  130   NCP(L) = 0
  140 CONTINUE
C
C     Since the specified number of iterations has been exceeded, set
C     IFAULT = 2.   This may indicate unforeseen looping.
C
      IFAULT = 2
C
C     Compute within-cluster sum of squares for each cluster.
C
  150 DO 160 L = 1, K
      WSS(L) = ZERO
      DO 160 J = 1, N
        C(L,J) = ZERO
  160 CONTINUE
      DO 170 I = 1, M
      II = IC1(I)
      DO 170 J = 1, N
        C(II,J) = C(II,J) + A(I,J)
  170 CONTINUE
      DO 190 J = 1, N
      DO 180 L = 1, K
  180   C(L,J) = C(L,J) / FLOAT(NC(L))
      DO 190 I = 1, M
        II = IC1(I)
        DA = A(I,J) - C(II,J)
        WSS(II) = WSS(II) + DA*DA
  190 CONTINUE
C
      RETURN
      END
C
C
      SUBROUTINE OPTRA(A, M, N, C, K, IC1, IC2, NC, AN1, AN2, NCP, D,
     *      ITRAN, LIVE, INDX)
C
C     ALGORITHM AS 136.1  APPL. STATIST. (1979) VOL.28, NO.1
C
C     This is the optimal transfer stage.
C
C     Each point is re-allocated, if necessary, to the cluster that
C     will induce a maximum reduction in the within-cluster sum of
C     squares.
C
      INTEGER IC1(M), IC2(M), NC(K), NCP(K), ITRAN(K), LIVE(K)
      REAL    A(M,N), D(M), C(K,N), AN1(K), AN2(K), ZERO, ONE
C
C     Define BIG to be a very large positive number.
C
      DATA BIG /1.0E30/, ZERO /0.0/, ONE/1.0/
C
C     If cluster L is updated in the last quick-transfer stage, it
C     belongs to the live set throughout this stage.   Otherwise, at
C     each step, it is not in the live set if it has not been updated
C     in the last M optimal transfer steps.
C
      DO 10 L = 1, K
      IF (ITRAN(L) .EQ. 1) LIVE(L) = M + 1
   10 CONTINUE
      DO 100 I = 1, M
      INDX = INDX + 1
      L1 = IC1(I)
      L2 = IC2(I)
      LL = L2
C
C     If point I is the only member of cluster L1, no transfer.
C
      IF (NC(L1) .EQ. 1) GO TO 90
C
C     If L1 has not yet been updated in this stage, no need to
C     re-compute D(I).
C
      IF (NCP(L1) .EQ. 0) GO TO 30
      DE = ZERO
      DO 20 J = 1, N
        DF = A(I,J) - C(L1,J)
        DE = DE + DF*DF
   20   CONTINUE
      D(I) = DE * AN1(L1)
C
C     Find the cluster with minimum R2.
C
   30   DA = ZERO
      DO 40 J = 1, N
        DB = A(I,J) - C(L2,J)
        DA = DA + DB*DB
   40   CONTINUE
      R2 = DA * AN2(L2)
      DO 60 L = 1, K
C
C     If I >= LIVE(L1), then L1 is not in the live set.   If this is
C     true, we only need to consider clusters that are in the live set
C     for possible transfer of point I.   Otherwise, we need to consider
C     all possible clusters.
C
        IF (I .GE. LIVE(L1) .AND. I .GE. LIVE(L) .OR. L .EQ. L1 .OR.
     *        L .EQ. LL) GO TO 60
        RR = R2 / AN2(L)
        DC = ZERO
        DO 50 J = 1, N
          DD = A(I,J) - C(L,J)
          DC = DC + DD*DD
          IF (DC .GE. RR) GO TO 60
   50     CONTINUE
        R2 = DC * AN2(L)
        L2 = L
   60     CONTINUE
        IF (R2 .LT. D(I)) GO TO 70
C
C     If no transfer is necessary, L2 is the new IC2(I).
C
        IC2(I) = L2
        GO TO 90
C
C     Update cluster centres, LIVE, NCP, AN1 & AN2 for clusters L1 and
C     L2, and update IC1(I) & IC2(I).
C
   70     INDX = 0
        LIVE(L1) = M + I
        LIVE(L2) = M + I
        NCP(L1) = I
        NCP(L2) = I
        AL1 = NC(L1)
        ALW = AL1 - ONE
        AL2 = NC(L2)
        ALT = AL2 + ONE
        DO 80 J = 1, N
          C(L1,J) = (C(L1,J) * AL1 - A(I,J)) / ALW
          C(L2,J) = (C(L2,J) * AL2 + A(I,J)) / ALT
   80     CONTINUE
        NC(L1) = NC(L1) - 1
        NC(L2) = NC(L2) + 1
        AN2(L1) = ALW / AL1
        AN1(L1) = BIG
        IF (ALW .GT. ONE) AN1(L1) = ALW / (ALW - ONE)
        AN1(L2) = ALT / AL2
        AN2(L2) = ALT / (ALT + ONE)
        IC1(I) = L2
        IC2(I) = L1
   90   CONTINUE
      IF (INDX .EQ. M) RETURN
  100 CONTINUE
      DO 110 L = 1, K
C
C     ITRAN(L) = 0 before entering QTRAN.   Also, LIVE(L) has to be
C     decreased by M before re-entering OPTRA.
C
      ITRAN(L) = 0
      LIVE(L) = LIVE(L) - M
  110 CONTINUE
C
      RETURN
      END
C
C
      SUBROUTINE QTRAN(A, M, N, C, K, IC1, IC2, NC, AN1, AN2, NCP, D,
     *    ITRAN, INDX)
C
C     ALGORITHM AS 136.2  APPL. STATIST. (1979) VOL.28, NO.1
C
C     This is the quick transfer stage.
C     IC1(I) is the cluster which point I belongs to.
C     IC2(I) is the cluster which point I is most likely to be
C         transferred to.
C     For each point I, IC1(I) & IC2(I) are switched, if necessary, to
C     reduce within-cluster sum of squares.  The cluster centres are
C     updated after each step.
C
      INTEGER IC1(M), IC2(M), NC(K), NCP(K), ITRAN(K)
      REAL    A(M,N), D(M), C(K,N), AN1(K), AN2(K), ZERO, ONE
C
C     Define BIG to be a very large positive number
C
      DATA BIG /1.0E30/, ZERO /0.0/, ONE /1.0/
C
C     In the optimal transfer stage, NCP(L) indicates the step at which
C     cluster L is last updated.   In the quick transfer stage, NCP(L)
C     is equal to the step at which cluster L is last updated plus M.
C
      ICOUN = 0
      ISTEP = 0
   10 DO 70 I = 1, M
      ICOUN = ICOUN + 1
      ISTEP = ISTEP + 1
      L1 = IC1(I)
      L2 = IC2(I)
C
C     If point I is the only member of cluster L1, no transfer.
C
      IF (NC(L1) .EQ. 1) GO TO 60
C
C     If ISTEP > NCP(L1), no need to re-compute distance from point I to
C     cluster L1.   Note that if cluster L1 is last updated exactly M
C     steps ago, we still need to compute the distance from point I to
C     cluster L1.
C
      IF (ISTEP .GT. NCP(L1)) GO TO 30
      DA = ZERO
      DO 20 J = 1, N
        DB = A(I,J) - C(L1,J)
        DA = DA + DB*DB
   20   CONTINUE
      D(I) = DA * AN1(L1)
C
C     If ISTEP >= both NCP(L1) & NCP(L2) there will be no transfer of
C     point I at this step.
C
   30   IF (ISTEP .GE. NCP(L1) .AND. ISTEP .GE. NCP(L2)) GO TO 60
      R2 = D(I) / AN2(L2)
      DD = ZERO
      DO 40 J = 1, N
        DE = A(I,J) - C(L2,J)
        DD = DD + DE*DE
        IF (DD .GE. R2) GO TO 60
   40   CONTINUE
C
C     Update cluster centres, NCP, NC, ITRAN, AN1 & AN2 for clusters
C     L1 & L2.   Also update IC1(I) & IC2(I).   Note that if any
C     updating occurs in this stage, INDX is set back to 0.
C
      ICOUN = 0
      INDX = 0
      ITRAN(L1) = 1
      ITRAN(L2) = 1
      NCP(L1) = ISTEP + M
      NCP(L2) = ISTEP + M
      AL1 = NC(L1)
      ALW = AL1 - ONE
      AL2 = NC(L2)
      ALT = AL2 + ONE
      DO 50 J = 1, N
        C(L1,J) = (C(L1,J) * AL1 - A(I,J)) / ALW
        C(L2,J) = (C(L2,J) * AL2 + A(I,J)) / ALT
   50   CONTINUE
      NC(L1) = NC(L1) - 1
      NC(L2) = NC(L2) + 1
      AN2(L1) = ALW / AL1
      AN1(L1) = BIG
      IF (ALW .GT. ONE) AN1(L1) = ALW / (ALW - ONE)
      AN1(L2) = ALT / AL2
      AN2(L2) = ALT / (ALT + ONE)
      IC1(I) = L2
      IC2(I) = L1
C
C     If no re-allocation took place in the last M steps, return.
C
   60   IF (ICOUN .EQ. M) RETURN
   70 CONTINUE
      GO TO 10
      END
