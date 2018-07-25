C===================================================== PELPE3.FOR
      SUBROUTINE PELPE3(XMOM,PARA)
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
C  PARAMETER ESTIMATION VIA L-MOMENTS FOR THE PEARSON TYPE 3 DISTRIBUTION
C
C  PARAMETERS OF ROUTINE:
C  XMOM   * INPUT* ARRAY OF LENGTH 3. CONTAINS THE L-MOMENTS LAMBDA-1,
C                  LAMBDA-2 AND TAU-3.
C  PARA   *OUTPUT* ARRAY OF LENGTH 3. ON EXIT, CONTAINS THE PARAMETERS
C                  IN THE ORDER MU, SIGMA, GAMMA (MEAN, S.D., SKEWNESS).
C
C  OTHER ROUTINES USED: DLGAMA
C
C  METHOD: RATIONAL APPROXIMATION IS USED TO EXPRESS ALPHA, THE SHAPE
C  PARAMETER OF THE GAMMA DISTRIBUTION, AS A FUNCTION OF TAU-3.
C  RELATIVE ACCURACY OF THE APPROXIMATION IS BETTER THAN 3E-5.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION XMOM(3),PARA(3)

Cf2py intent(in) xmom
Cf2py intent(out) para
      
      DATA ZERO/0D0/,THIRD/0.33333333D0/,HALF/0.5D0/,ONE/1D0/,TWO/2D0/
C
C         SMALL IS USED TO TEST WHETHER SKEWNESS IS EFFECTIVELY ZERO
C
      DATA SMALL/1D-6/
C
C         CONSTANTS USED IN MINIMAX APPROXIMATIONS
C
      DATA C1,C2,C3/ 0.2906D0,  0.1882D0,  0.0442D0/
      DATA D1,D2,D3/ 0.36067D0,-0.59567D0, 0.25361D0/
      DATA D4,D5,D6/-2.78861D0, 2.56096D0,-0.77045D0/
      DATA PI3,ROOTPI/9.4247780D0,1.7724539D0/
C
      T3=DABS(XMOM(3))
      IF(XMOM(2).LE.ZERO.OR.T3.GE.ONE)GOTO 1000
      IF(T3.LE.SMALL)GOTO 100
      IF(T3.GE.THIRD)GOTO 10
      T=PI3*T3*T3
      ALPHA=(ONE+C1*T)/(T*(ONE+T*(C2+T*C3)))
      GOTO 20
   10 CONTINUE
      T=ONE-T3
      ALPHA=T*(D1+T*(D2+T*D3))/(ONE+T*(D4+T*(D5+T*D6)))
   20 CONTINUE
      RTALPH=DSQRT(ALPHA)
      BETA=ROOTPI*XMOM(2)*DEXP(DLGAMA(ALPHA)-DLGAMA(ALPHA+HALF))
      PARA(1)=XMOM(1)
      PARA(2)=BETA*RTALPH
      PARA(3)=TWO/RTALPH
      IF(XMOM(3).LT.ZERO)PARA(3)=-PARA(3)
      RETURN
C
C         ZERO SKEWNESS
C
  100 CONTINUE
      PARA(1)=XMOM(1)
      PARA(2)=XMOM(2)*ROOTPI
      PARA(3)=ZERO
      RETURN
C
 1000 WRITE(6,7000)
      DO 1010 I=1,3
 1010 PARA(I)=ZERO
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE PELPE3 : L-MOMENTS INVALID')
      END

C===================================================== DLGAMA.FOR
      DOUBLE PRECISION FUNCTION DLGAMA(X)
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
C  LOGARITHM OF GAMMA FUNCTION
C
C  BASED ON ALGORITHM ACM291, COMMUN. ASSOC. COMPUT. MACH. (1966)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DATA SMALL,CRIT,BIG,TOOBIG/1D-7,13D0,1D9,2D36/
C
C         C0 IS 0.5*LOG(2*PI)
C         C1...C7 ARE THE COEFFTS OF THE ASYMPTOTIC EXPANSION OF DLGAMA
C
      DATA C0,C1,C2,C3,C4,C5,C6,C7/
     *   0.91893 85332 04672 742D 0,  0.83333 33333 33333 333D-1,
     *  -0.27777 77777 77777 778D-2,  0.79365 07936 50793 651D-3,
     *  -0.59523 80952 38095 238D-3,  0.84175 08417 50841 751D-3,
     *  -0.19175 26917 52691 753D-2,  0.64102 56410 25641 026D-2/
C
C         S1 IS -(EULER'S CONSTANT), S2 IS PI**2/12
C
      DATA S1/-0.57721 56649 01532 861D 0/
      DATA S2/ 0.82246 70334 24113 218D 0/
C
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/,TWO/2D0/
      DLGAMA=ZERO
      IF(X.LE.ZERO)GOTO 1000
      IF(X.GT.TOOBIG)GOTO 1000
C
C         USE SMALL-X APPROXIMATION IF X IS NEAR 0, 1 OR 2
C
      IF(DABS(X-TWO).GT.SMALL)GOTO 10
      DLGAMA=DLOG(X-ONE)
      XX=X-TWO
      GOTO 20
   10 IF(DABS(X-ONE).GT.SMALL)GOTO 30
      XX=X-ONE
   20 DLGAMA=DLGAMA+XX*(S1+XX*S2)
      RETURN
   30 IF(X.GT.SMALL)GOTO 40
      DLGAMA=-DLOG(X)+S1*X
      RETURN
C
C         REDUCE TO DLGAMA(X+N) WHERE X+N.GE.CRIT
C
   40 SUM1=ZERO
      Y=X
      IF(Y.GE.CRIT)GOTO 60
      Z=ONE
   50 Z=Z*Y
      Y=Y+ONE
      IF(Y.LT.CRIT)GOTO 50
      SUM1=SUM1-DLOG(Z)
C
C         USE ASYMPTOTIC EXPANSION IF Y.GE.CRIT
C
   60 SUM1=SUM1+(Y-HALF)*DLOG(Y)-Y+C0
      SUM2=ZERO
      IF(Y.GE.BIG)GOTO 70
      Z=ONE/(Y*Y)
      SUM2=((((((C7*Z+C6)*Z+C5)*Z+C4)*Z+C3)*Z+C2)*Z+C1)/Y
   70 DLGAMA=SUM1+SUM2
      RETURN
C
 1000 WRITE(6,7000)X
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE DLGAMA :',
     *  ' ARGUMENT OUT OF RANGE :',D24.16)
      END
