C===================================================== CDFWAK.FOR
      DOUBLE PRECISION FUNCTION CDFWAK(X,PARA)
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
C  CUMULATIVE DISTRIBUTION FUNCTION OF THE WAKEBY DISTRIBUTION
C
C  OTHER ROUTINES USED: QUAWAK
C
C  METHOD: THE EQUATION X=G(Z), WHERE G(Z) IS THE WAKEBY QUANTILE
C  EXPRESSED AS A FUNCTION OF Z=-LOG(1-F), IS SOLVED USING HALLEY'S
C  METHOD (THE 2ND-ORDER ANALOGUE OF NEWTON-RAPHSON ITERATION).
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(5)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/
      DATA P1/0.1D0/,P7/0.7D0/,P99/0.99D0/
C
C         EPS,MAXIT CONTROL THE TEST FOR CONVERGENCE OF THE ITERATION
C         ZINCMX IS THE LARGEST PERMITTED ITERATIVE STEP
C         ZMULT CONTROLS WHAT HAPPENS WHEN THE ITERATION STEPS BELOW ZERO
C         UFL SHOULD BE CHOSEN SO THAT DEXP(UFL) JUST DOES NOT CAUSE
C           UNDERFLOW
C
      DATA EPS/1D-8/,MAXIT/20/,ZINCMX/3D0/,ZMULT/0.2D0/
      DATA UFL/-170D0/
C
      XI=PARA(1)
      A=PARA(2)
      B=PARA(3)
      C=PARA(4)
      D=PARA(5)
C
C         TEST FOR VALID PARAMETERS
C
      IF(B+D.LE.ZERO.AND.(B.NE.ZERO.OR.C.NE.ZERO.OR.D.NE.ZERO))GOTO 1000
      IF(A.EQ.ZERO.AND.B.NE.ZERO)GOTO 1000
      IF(C.EQ.ZERO.AND.D.NE.ZERO)GOTO 1000
      IF(C.LT.ZERO.OR.A+C.LT.ZERO)GOTO 1000
      IF(A.EQ.ZERO.AND.C.EQ.ZERO)GOTO 1000
C
      CDFWAK=ZERO
      IF(X.LE.XI)RETURN
C
C         TEST FOR SPECIAL CASES
C
      IF(B.EQ.ZERO.AND.C.EQ.ZERO.AND.D.EQ.ZERO)GOTO 100
      IF(C.EQ.ZERO)GOTO 110
      IF(A.EQ.ZERO)GOTO 120
C
C         GENERAL CASE
C
      CDFWAK=ONE
      IF(D.LT.ZERO.AND.X.GE.XI+A/B-C/D)RETURN
C
C         INITIAL VALUES FOR ITERATION:
C         IF X IS IN THE LOWEST DECILE OF THE DISTRIBUTION, START AT Z=0
C           (F=0);
C         IF X IS IN THE HIGHEST PERCENTILE OF THE DISTRIBUTION,
C           STARTING VALUE IS OBTAINED FROM ASYMPTOTIC FORM OF THE
C           DISTRIBUTION FOR LARGE Z (F NEAR 1);
C         OTHERWISE START AT Z=0.7 (CLOSE TO F=0.5).
C
      Z=P7
      IF(X.LT.QUAWAK(P1,PARA))Z=ZERO
      IF(X.LT.QUAWAK(P99,PARA))GOTO 10
      IF(D.LT.ZERO)Z=DLOG((X-XI-A/B)*D/C+ONE)/D
      IF(D.EQ.ZERO)Z=(X-XI-A/B)/C
      IF(D.GT.ZERO)Z=DLOG((X-XI)*D/C+ONE)/D
   10 CONTINUE
C
C         HALLEY'S METHOD, WITH MODIFICATIONS:
C         IF HALLEY ITERATION WOULD MOVE IN WRONG DIRECTION
C           (TEMP.LE.ZERO), USE ORDINARY NEWTON-RAPHSON INSTEAD;
C         IF STEP GOES TOO FAR (ZINC.GT.ZINCMX OR ZNEW.LE.ZERO),
C            LIMIT ITS LENGTH.
C
      DO 30 IT=1,MAXIT
      EB=ZERO
      BZ=-B*Z
      IF(BZ.GE.UFL)EB=DEXP(BZ)
      GB=Z
      IF(DABS(B).GT.EPS)GB=(ONE-EB)/B
      ED=DEXP(D*Z)
      GD=-Z
      IF(DABS(D).GT.EPS)GD=(ONE-ED)/D
      XEST=XI+A*GB-C*GD
      FUNC=X-XEST
      DERIV1=A*EB+C*ED
      DERIV2=-A*B*EB+C*D*ED
      TEMP=DERIV1+HALF*FUNC*DERIV2/DERIV1
      IF(TEMP.LE.ZERO)TEMP=DERIV1
      ZINC=FUNC/TEMP
      IF(ZINC.GT.ZINCMX)ZINC=ZINCMX
      ZNEW=Z+ZINC
      IF(ZNEW.LE.ZERO)GOTO 20
      Z=ZNEW
      IF(DABS(ZINC).LE.EPS)GOTO 200
      GOTO 30
   20 Z=Z*ZMULT
   30 CONTINUE
C
C         NOT CONVERGED
C
      WRITE(6,7010)
      GOTO 200
C
C         SPECIAL CASE B=C=D=0: WAKEBY IS EXPONENTIAL
C
  100 CONTINUE
      Z=(X-XI)/A
      GOTO 200
C
C         SPECIAL CASE C=0: WAKEBY IS GENERALIZED PARETO, BOUNDED ABOVE
C
  110 CONTINUE
      CDFWAK=ONE
      IF(X.GE.XI+A/B)RETURN
      Z=-DLOG(ONE-(X-XI)*B/A)/B
      GOTO 200
C
C         SPECIAL CASE A=0: WAKEBY IS GENERALIZED PARETO, NO UPPER BOUND
C
  120 CONTINUE
      Z=DLOG(ONE+(X-XI)*D/C)/D
      GOTO 200
C
C         CONVERT Z VALUE TO PROBABILITY
C
  200 CDFWAK=ONE
      IF(-Z.LT.UFL)RETURN
      CDFWAK=ONE-DEXP(-Z)
      RETURN
C
 1000 WRITE(6,7000)
      CDFWAK=ZERO
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE CDFWAK : PARAMETERS INVALID')
 7010 FORMAT(' ** WARNING ** ROUTINE CDFWAK :',
     *  ' ITERATION HAS NOT CONVERGED. RESULT MAY BE UNRELIABLE.')
      END

C===================================================== QUAWAK.FOR
      DOUBLE PRECISION FUNCTION QUAWAK(F,PARA)
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
C  QUANTILE FUNCTION OF THE WAKEBY DISTRIBUTION
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(5)
      DATA ZERO/0D0/,ONE/1D0/
C
C         UFL SHOULD BE CHOSEN SO THAT EXP(UFL) JUST DOES NOT CAUSE
C         UNDERFLOW
C
      DATA UFL/-170D0/
C
      XI=PARA(1)
      A=PARA(2)
      B=PARA(3)
      C=PARA(4)
      D=PARA(5)
C
C         TEST FOR VALID PARAMETERS
C
      IF(B+D.LE.ZERO.AND.(B.NE.ZERO.OR.C.NE.ZERO.OR.D.NE.ZERO))GOTO 1000
      IF(A.EQ.ZERO.AND.B.NE.ZERO)GOTO 1000
      IF(C.EQ.ZERO.AND.D.NE.ZERO)GOTO 1000
      IF(C.LT.ZERO.OR.A+C.LT.ZERO)GOTO 1000
      IF(A.EQ.ZERO.AND.C.EQ.ZERO)GOTO 1000
C
      IF(F.LE.ZERO.OR.F.GE.ONE)GOTO 10
      Z=-DLOG(ONE-F)
      Y1=Z
      IF(B.EQ.ZERO)GOTO 5
      TEMP=-B*Z
      IF(TEMP.LT.UFL)Y1=ONE/B
      IF(TEMP.GE.UFL)Y1=(ONE-DEXP(TEMP))/B
    5 CONTINUE
      Y2=Z
      IF(D.NE.ZERO)Y2=(ONE-DEXP(D*Y2))/(-D)
      QUAWAK=XI+A*Y1+C*Y2
      RETURN
C
   10 IF(F.EQ.ZERO)GOTO 20
      IF(F.EQ.ONE)GOTO 30
      GOTO 1010
   20 QUAWAK=XI
      RETURN
   30 IF(D.GT.ZERO)GOTO 1010
      IF(D.LT.ZERO)QUAWAK=XI+A/B-C/D
      IF(D.EQ.ZERO.AND.C.GT.ZERO)GOTO 1010
      IF(D.EQ.ZERO.AND.C.EQ.ZERO.AND.B.EQ.ZERO)GOTO 1010
      IF(D.EQ.ZERO.AND.C.EQ.ZERO.AND.B.GT.ZERO)QUAWAK=XI+A/B
      RETURN
C
 1000 WRITE(6,7000)
      QUAWAK=ZERO
      RETURN
 1010 WRITE(6,7010)
      QUAWAK=ZERO
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE QUAWAK : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE QUAWAK :',
     *  ' ARGUMENT OF FUNCTION INVALID')
      END