C===================================================== LMRGAM.FOR
      SUBROUTINE LMRGAM(PARA,XMOM,NMOM)
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
C  L-MOMENT RATIOS FOR THE GAMMA DISTRIBUTION
C
C  PARAMETERS OF ROUTINE:
C  PARA   * INPUT* ARRAY OF LENGTH 2. CONTAINS THE PARAMETERS OF THE
C                  DISTRIBUTION, IN THE ORDER ALPHA,BETA (SHAPE,SCALE).
C  XMOM   *OUTPUT* ARRAY OF LENGTH NMOM. ON EXIT, CONTAINS UP TO 4 OF
C                  THE L-MOMENTS LAMBDA-1, LAMBDA-2, TAU-3, TAU-4.
C  NMOM   * INPUT* NUMBER OF L-MOMENTS TO BE FOUND. AT MOST 4.
C
C  OTHER ROUTINES USED: DLGAMA
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(2),XMOM(NMOM)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/
C
C         CONST IS 1/SQRT(PI)
C
      DATA CONST/0.56418 95835 47756 287D0/
C
C         COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS
C         A0 IS 1/SQRT(3*PI)
C         C0 IS TAU-4 FOR THE NORMAL DISTRIBUTION
C
      DATA A0      / 0.32573501D+00/
      DATA A1,A2,A3/ 0.16869150D+00, 0.78327243D-01,-0.29120539D-02/
      DATA B1,B2   / 0.46697102D+00, 0.24255406D+00/
      DATA C0      / 0.12260172D+00/
      DATA C1,C2,C3/ 0.53730130D-01, 0.43384378D-01, 0.11101277D-01/
      DATA D1,D2   / 0.18324466D+00, 0.20166036D+00/
      DATA E1,E2,E3/ 0.23807576D+01, 0.15931792D+01, 0.11618371D+00/
      DATA F1,F2,F3/ 0.51533299D+01, 0.71425260D+01, 0.19745056D+01/
      DATA G1,G2,G3/ 0.21235833D+01, 0.41670213D+01, 0.31925299D+01/
      DATA H1,H2,H3/ 0.90551443D+01, 0.26649995D+02, 0.26193668D+02/
C
      ALPHA=PARA(1)
      BETA=PARA(2)
      IF(ALPHA.LE.ZERO.OR.BETA.LE.ZERO)GOTO 1000
      IF(NMOM.GT.4)GOTO 1010
C
C         LAMBDA-1
C
      XMOM(1)=ALPHA*BETA
      IF(NMOM.EQ.1)RETURN
C
C         LAMBDA-2
C
      XMOM(2)=BETA*CONST*DEXP(DLGAMA(ALPHA+HALF)-DLGAMA(ALPHA))
      IF(NMOM.EQ.2)RETURN
C
C         HIGHER MOMENTS
C
      IF(ALPHA.LT.ONE)GOTO 10
      Z=ONE/ALPHA
      XMOM(3)=DSQRT(Z)*(((A3*Z+A2)*Z+A1)*Z+A0)/((B2*Z+B1)*Z+ONE)
      IF(NMOM.EQ.3)RETURN
      XMOM(4)=(((C3*Z+C2)*Z+C1)*Z+C0)/((D2*Z+D1)*Z+ONE)
      IF(NMOM.GT.4)WRITE(6,7010)
      RETURN
C
   10 Z=ALPHA
      XMOM(3)=(((E3*Z+E2)*Z+E1)*Z+ONE)/(((F3*Z+F2)*Z+F1)*Z+ONE)
      IF(NMOM.EQ.3)RETURN
      XMOM(4)=(((G3*Z+G2)*Z+G1)*Z+ONE)/(((H3*Z+H2)*Z+H1)*Z+ONE)
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
 1010 WRITE(6,7010)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE LMRGAM : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE LMRGAM : PARAMETER NMOM TOO LARGE')
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