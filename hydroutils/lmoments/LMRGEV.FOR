C===================================================== LMRGEV.FOR
      SUBROUTINE LMRGEV(PARA,XMOM,NMOM)
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
C  L-MOMENT RATIOS FOR THE GENERALIZED EXTREME-VALUE DISTRIBUTION
C
C  PARAMETERS OF ROUTINE:
C  PARA   * INPUT* ARRAY OF LENGTH 3. CONTAINS THE PARAMETERS OF THE
C                  DISTRIBUTION, IN THE ORDER XI, ALPHA, K (LOCATION,
C                  SCALE, SHAPE).
C  XMOM   *OUTPUT* ARRAY OF LENGTH NMOM. ON EXIT, CONTAINS THE L-MOMENTS
C                  LAMBDA-1, LAMBDA-2, TAU-3, TAU-4, ... .
C  NMOM   * INPUT* NUMBER OF L-MOMENTS TO BE FOUND. AT MOST 20.
C
C  OTHER ROUTINES USED: DLGAMA
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(3),XMOM(NMOM),ZMOM(20)
      DATA ZERO/0D0/,ONE/1D0/,TWO/2D0/,THREE/3D0/,FOUR/4D0/,SIX/6D0/
C
C         ARRAY ZMOM CONTAINS THE L-MOMENT RATIOS OF THE STANDARD
C         GUMBEL DISTRIBUTION (XI=0, ALPHA=1).
C         ZMOM(1) IS EULER'S CONSTANT, ZMOM(2) IS LOG(2).
C
      DATA ZMOM/
     *  0.57721 56649 01532 861D 0,  0.69314 71805 59945 309D 0,
     *  0.16992 50014 42312 363D 0,  0.15037 49927 88438 185D 0,
     *  0.55868 35005 77583 138D-1,  0.58110 02399 99710 876D-1,
     *  0.27624 25842 97309 125D-1,  0.30556 37665 79053 126D-1,
     *  0.16465 02822 58328 802D-1,  0.18784 66242 98170 912D-1,
     *  0.10932 82150 63027 148D-1,  0.12697 31266 76329 530D-1,
     *  0.77898 28180 57231 804D-2,  0.91483 61796 21999 726D-2,
     *  0.58333 23893 28363 588D-2,  0.69010 42875 90348 154D-2,
     *  0.45326 79701 80679 549D-2,  0.53891 68113 26595 459D-2,
     *  0.36240 77677 72368 790D-2,  0.43238 76086 05538 096D-2/
C
C         SMALL IS USED TO TEST WHETHER K IS EFFECTIVELY ZERO
C
      DATA SMALL/1D-6/
C
      U=PARA(1)
      A=PARA(2)
      G=PARA(3)
      IF(A.LE.ZERO.OR.G.LE.-ONE)GOTO 1000
      IF(NMOM.GT.20)GOTO 1010
C
C         TEST FOR K=0
C
      IF(DABS(G).GT.SMALL)GOTO 5
      XMOM(1)=U
      IF(NMOM.EQ.1)RETURN
      XMOM(2)=A*ZMOM(2)
      IF(NMOM.EQ.2)RETURN
      DO 2 I=3,NMOM
    2 XMOM(I)=ZMOM(I)
      RETURN
    5 CONTINUE
C
C         FIRST 2 MOMENTS
C
   10 CONTINUE
      GAM=DEXP(DLGAMA(ONE+G))
      XMOM(1)=U+A*(ONE-GAM)/G
      IF(NMOM.EQ.1)RETURN
      XX2=ONE-TWO**(-G)
      XMOM(2)=A*XX2*GAM/G
      IF(NMOM.EQ.2)RETURN
C
C         HIGHER MOMENTS
C
      Z0=ONE
      DO 50 J=3,NMOM
      DJ=J
      BETA=(ONE-DJ**(-G))/XX2
      Z0=Z0*(FOUR*DJ-SIX)/DJ
      Z=Z0*THREE*(DJ-ONE)/(DJ+ONE)
      SUM=Z0*BETA-Z
      IF(J.EQ.3)GOTO 40
      DO 30 I=2,J-2
      DI=I
      Z=Z*(DI+DI+ONE)*(DJ-DI)/((DI+DI-ONE)*(DJ+DI))
      SUM=SUM-Z*XMOM(I+1)
   30 CONTINUE
   40 XMOM(J)=SUM
   50 CONTINUE
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
 1010 WRITE(6,7010)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE LMRGEV : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE LMRGEV : PARAMETER NMOM TOO LARGE')
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