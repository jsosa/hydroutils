C===================================================== LMRKAP.FOR
      SUBROUTINE LMRKAP(PARA,XMOM,NMOM)
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
C  L-MOMENT RATIOS FOR THE KAPPA DISTRIBUTION
C
C  PARAMETERS OF ROUTINE:
C  PARA   * INPUT* ARRAY OF LENGTH 4. CONTAINS THE PARAMETERS OF THE
C                  DISTRIBUTION, IN THE ORDER XI, ALPHA, K, H.
C  XMOM   *OUTPUT* ARRAY OF LENGTH NMOM. ON EXIT, CONTAINS THE L-MOMENTS
C                  LAMBDA-1, LAMBDA-2, TAU-3, TAU-4, ... .
C  NMOM   * INPUT* NUMBER OF L-MOMENTS TO BE FOUND. AT MOST 20.
C
C  OTHER ROUTINES USED: DLGAMA,DIGAMD
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(4),XMOM(NMOM),BETA(20)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/,THREE/3D0/,FOUR/4D0/,SIX/6D0/
C
C         EU  IS EULER'S CONSTANT
C
      DATA EU/0.577215664901532861D0/
C
C         SMALL IS USED TO TEST WHETHER H IS EFFECTIVELY ZERO
C         OFL SHOULD BE CHOSEN SO THAT EXP(OFL) JUST DOES NOT CAUSE
C         OVERFLOW
C
      DATA SMALL/1D-8/,OFL/170D0/
C
      U=PARA(1)
      A=PARA(2)
      G=PARA(3)
      H=PARA(4)
C
C         TEST FOR FEASIBLE PARAMETERS
C
      IF(A.LE.ZERO)GOTO 1000
      IF(G.LE.-ONE)GOTO 1000
      IF(H.LT.ZERO.AND.G*H.LE.-ONE)GOTO 1000
      IF(NMOM.GT.20)GOTO 1010
C
C         CALCULATE FUNCTIONS OCCURRING IN THE PWM'S BETA-SUB-R
C
      DLGAM=DLGAMA(ONE+G)
      ICASE=1
      IF(H.GT.ZERO)ICASE=3
      IF(DABS(H).LT.SMALL)ICASE=2
      IF(G.EQ.ZERO)ICASE=ICASE+3
      GOTO(10,30,50,70,90,110),ICASE
C
C         - CASE H<0, G NONZERO
C
   10 DO 20 IR=1,NMOM
      R=IR
      ARG=DLGAM+DLGAMA(-R/H-G)-DLGAMA(-R/H)-G*DLOG(-H)
      IF(DABS(ARG).GT.OFL)GOTO 1020
   20 BETA(IR)=DEXP(ARG)
      GOTO 130
C
C         - CASE H SMALL, G NONZERO
C
   30 DO 40 IR=1,NMOM
      R=IR
   40 BETA(IR)=DEXP(DLGAM-G*DLOG(R))*(ONE-HALF*H*G*(ONE+G)/R)
      GOTO 130
C
C         - CASE H>0, G NONZERO
C
   50 DO 60 IR=1,NMOM
      R=IR
      ARG=DLGAM+DLGAMA(ONE+R/H)-DLGAMA(ONE+G+R/H)-G*DLOG(H)
      IF(DABS(ARG).GT.OFL)GOTO 1020
   60 BETA(IR)=DEXP(ARG)
      GOTO 130
C
C         - CASE H<0, G=0
C
   70 DO 80 IR=1,NMOM
      R=IR
   80 BETA(IR)=EU+DLOG(-H)+DIGAMD(-R/H)
      GOTO 130
C
C         - CASE H SMALL, G=0
C
   90 DO 100 IR=1,NMOM
      R=IR
  100 BETA(IR)=EU+DLOG(R)
      GOTO 130
C
C         - CASE H>0, G=0
C
  110 DO 120 IR=1,NMOM
      R=IR
  120 BETA(IR)=EU+DLOG(H)+DIGAMD(ONE+R/H)
      GOTO 130
C
C         LAMBDA-1
C
  130 CONTINUE
      IF(G.EQ.ZERO)XMOM(1)=U+A*BETA(1)
      IF(G.NE.ZERO)XMOM(1)=U+A*(ONE-BETA(1))/G
      IF(NMOM.EQ.1)RETURN
C
C         LAMBDA-2
C
      ALAM2=BETA(2)-BETA(1)
      IF(G.EQ.ZERO)XMOM(2)=A*ALAM2
      IF(G.NE.ZERO)XMOM(2)=A*ALAM2/(-G)
      IF(NMOM.EQ.2)RETURN
C
C         HIGHER MOMENTS
C
      Z0=ONE
      DO 170 J=3,NMOM
      DJ=J
      Z0=Z0*(FOUR*DJ-SIX)/DJ
      Z=Z0*THREE*(DJ-ONE)/(DJ+ONE)
      SUM=Z0*(BETA(J)-BETA(1))/ALAM2-Z
      IF(J.EQ.3)GOTO 160
      DO 150 I=2,J-2
      DI=I
      Z=Z*(DI+DI+ONE)*(DJ-DI)/((DI+DI-ONE)*(DJ+DI))
      SUM=SUM-Z*XMOM(I+1)
  150 CONTINUE
  160 XMOM(J)=SUM
  170 CONTINUE
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
 1010 WRITE(6,7010)
      RETURN
 1020 WRITE(6,7020)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE LMRKAP : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE LMRKAP : PARAMETER NMOM TOO LARGE')
 7020 FORMAT(' *** ERROR *** ROUTINE LMRKAP :',
     *  ' CALCULATIONS OF L-MOMENTS HAVE BROKEN DOWN')
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

C===================================================== DIGAMD.FOR
      DOUBLE PRECISION FUNCTION DIGAMD(X)
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
C  DIGAMMA FUNCTION (EULER'S PSI FUNCTION) - THE FIRST DERIVATIVE OF
C  LOG(GAMMA(X))
C
C  BASED ON ALGORITHM AS103, APPL. STATIST. (1976) VOL.25 NO.3
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/
      DATA SMALL/1D-9/,CRIT/13D0/
C
C         C1...C7 ARE THE COEFFTS OF THE ASYMPTOTIC EXPANSION OF DIGAMD
C         D1 IS  -(EULER'S CONSTANT)
C
      DATA C1,C2,C3,C4,C5,C6,C7,D1/
     *  0.83333 33333 33333 333D-1,  -0.83333 33333 33333 333D-2,
     *  0.39682 53968 25396 825D-2,  -0.41666 66666 66666 666D-2,
     *  0.75757 57575 75757 575D-2,  -0.21092 79609 27960 928D-1,
     *  0.83333 33333 33333 333D-1,  -0.57721 56649 01532 861D 0/
      DIGAMD=ZERO
      IF(X.LE.ZERO)GOTO 1000
C
C         USE SMALL-X APPROXIMATION IF X.LE.SMALL
C
      IF(X.GT.SMALL)GOTO 10
      DIGAMD=D1-ONE/X
      RETURN
C
C         REDUCE TO DIGAMD(X+N) WHERE X+N.GE.CRIT
C
   10 Y=X
   20 IF(Y.GE.CRIT)GOTO 30
      DIGAMD=DIGAMD-ONE/Y
      Y=Y+ONE
      GOTO 20
C
C         USE ASYMPTOTIC EXPANSION IF Y.GE.CRIT
C
   30 DIGAMD=DIGAMD+DLOG(Y)-HALF/Y
      Y=ONE/(Y*Y)
      SUM=((((((C7*Y+C6)*Y+C5)*Y+C4)*Y+C3)*Y+C2)*Y+C1)*Y
      DIGAMD=DIGAMD-SUM
      RETURN
C
 1000 WRITE(6,7000)X
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE DIGAMD :',
     *  ' ARGUMENT OUT OF RANGE :',D24.16)
      END