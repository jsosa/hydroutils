C===================================================== CDFPE3.FOR
      DOUBLE PRECISION FUNCTION CDFPE3(X,PARA)
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
C  DISTRIBUTION FUNCTION OF THE PEARSON TYPE 3 DISTRIBUTION
C
C  OTHER ROUTINES USED: DERF,DLGAMA,GAMIND
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(3)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/,TWO/2D0/,FOUR/4D0/
      DATA RTHALF/0.70710 67811 86547 524D0/
C
C         SMALL IS USED TO TEST WHETHER SKEWNESS IS EFFECTIVELY ZERO
C
      DATA SMALL/1D-6/
C
      CDFPE3=ZERO
      IF(PARA(2).LE.ZERO)GOTO 1000
      GAMMA=PARA(3)
      IF(DABS(GAMMA).LE.SMALL)GOTO 10
      ALPHA=FOUR/(GAMMA*GAMMA)
      Z=TWO*(X-PARA(1))/(PARA(2)*GAMMA)+ALPHA
      IF(Z.GT.ZERO)CDFPE3=GAMIND(Z,ALPHA,DLGAMA(ALPHA))
      IF(GAMMA.LT.ZERO)CDFPE3=ONE-CDFPE3
      RETURN
C
C         ZERO SKEWNESS
C
   10 Z=(X-PARA(1))/PARA(2)
      CDFPE3=HALF+HALF*DERF(Z*RTHALF)
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE CDFPE3 : PARAMETERS INVALID')
      END

C===================================================== DERF.FOR
      DOUBLE PRECISION FUNCTION DERF(X)
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
C  ERROR FUNCTION
C
C  BASED ON ALGORITHM 5666, J.F.HART ET AL. (1968) 'COMPUTER
C  APPROXIMATIONS'
C
C  ACCURATE TO 15 DECIMAL PLACES
C
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA ZERO/0D0/,ONE/1D0/,TWO/2D0/,THREE/3D0/,FOUR/4D0/,P65/0.65D0/
C
C         COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATION
C
      DATA P0,P1,P2,P3,P4,P5,P6/
     *  0.22020 68679 12376 1D3,    0.22121 35961 69931 1D3,
     *  0.11207 92914 97870 9D3,    0.33912 86607 83830 0D2,
     *  0.63739 62203 53165 0D1,    0.70038 30644 43688 1D0,
     *  0.35262 49659 98910 9D-1/
      DATA Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7/
     *  0.44041 37358 24752 2D3,   0.79382 65125 19948 4D3,
     *  0.63733 36333 78831 1D3,   0.29656 42487 79673 7D3,
     *  0.86780 73220 29460 8D2,   0.16064 17757 92069 5D2,
     *  0.17556 67163 18264 2D1,   0.88388 34764 83184 4D-1/
C
C         C1 IS SQRT(2), C2 IS SQRT(2/PI)
C         BIG IS THE POINT AT WHICH DERF=1 TO MACHINE PRECISION
C
      DATA C1/1.4142 13562 37309 5D0/
      DATA C2/7.9788 45608 02865 4D-1/
      DATA BIG/6.25D0/,CRIT/5D0/
C
      DERF=ZERO
      IF(X.EQ.ZERO)RETURN
      XX=DABS(X)
      IF(XX.GT.BIG)GOTO 20
      EXPNTL=DEXP(-X*X)
      ZZ=DABS(X*C1)
      IF(XX.GT.CRIT)GOTO 10
      DERF=EXPNTL*((((((P6*ZZ+P5)*ZZ+P4)*ZZ+P3)*ZZ+P2)*ZZ+P1)*ZZ+P0)/
     *  (((((((Q7*ZZ+Q6)*ZZ+Q5)*ZZ+Q4)*ZZ+Q3)*ZZ+Q2)*ZZ+Q1)*ZZ+Q0)
      IF(X.GT.ZERO)DERF=ONE-TWO*DERF
      IF(X.LT.ZERO)DERF=TWO*DERF-ONE
      RETURN
C
   10 DERF=EXPNTL*C2/(ZZ+ONE/(ZZ+TWO/(ZZ+THREE/(ZZ+FOUR/(ZZ+P65)))))
      IF(X.GT.ZERO)DERF=ONE-DERF
      IF(X.LT.ZERO)DERF=DERF-ONE
      RETURN
C
   20 DERF=ONE
      IF(X.LT.ZERO)DERF=-ONE
      RETURN
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

C===================================================== GAMIND.FOR
      DOUBLE PRECISION FUNCTION GAMIND(X,ALPHA,G)
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
C  THE INCOMPLETE GAMMA INTEGRAL
C
C  BASED ON ALGORITHM AS239, APPL. STATIST. (1988) VOL.37 NO.3
C
C  PARAMETERS OF ROUTINE:
C  X      * INPUT* ARGUMENT OF FUNCTION (UPPER LIMIT OF INTEGRATION)
C  ALPHA  * INPUT* SHAPE PARAMETER
C  G      * INPUT* LOG(GAMMA(ALPHA)). MUST BE SUPPLIED BY THE PROGRAM,
C                  E.G. AS DLGAMA(ALPHA).
C
C  OTHER ROUTINES USED: DERF
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/,TWO/2D0/,THREE/3D0/,X13/13D0/,
     *  X36/36D0/,X42/42D0/,X119/119D0/,X1620/1620D0/,X38880/38880D0/,
     *  RTHALF/0.70710 67811 86547 524D0/
C
C         EPS,MAXIT CONTROL THE TEST FOR CONVERGENCE OF THE SERIES AND
C           CONTINUED-FRACTION EXPANSIONS.
C         OFL IS A LARGE NUMBER, USED TO RESCALE THE CONTINUED FRACTION.
C         UFL IS SUCH THAT EXP(UFL) IS JUST .GT. ZERO.
C         AHILL CONTROLS THE SWITCH TO HILL'S APPROXIMATION.
C
      DATA EPS/1D-12/,MAXIT/100000/,OFL/1D30/,UFL/-180D0/,AHILL/1D4/
      GAMIND=ZERO
      IF(ALPHA.LE.ZERO)GOTO 1000
      IF(X.LT.ZERO)GOTO 1010
      IF(X.EQ.ZERO)RETURN
C
      IF(ALPHA.GT.AHILL)GOTO 100
      IF(X.GT.ONE.AND.X.GE.ALPHA)GOTO 50
C
C         SERIES EXPANSION
C
      SUM=ONE
      TERM=ONE
      A=ALPHA
      DO 10 IT=1,MAXIT
      A=A+ONE
      TERM=TERM*X/A
      SUM=SUM+TERM
      IF(TERM.LE.EPS)GOTO 20
   10 CONTINUE
      WRITE(6,7020)
   20 ARG=ALPHA*DLOG(X)-X-G+DLOG(SUM/ALPHA)
      GAMIND=ZERO
      IF(ARG.GE.UFL)GAMIND=DEXP(ARG)
      RETURN
C
C         CONTINUED-FRACTION EXPANSION
C
   50 CONTINUE
      A=ONE-ALPHA
      B=A+X+ONE
      TERM=ZERO
      PN1=ONE
      PN2=X
      PN3=X+ONE
      PN4=X*B
      RATIO=PN3/PN4
      DO 70 IT=1,MAXIT
      A=A+ONE
      B=B+TWO
      TERM=TERM+ONE
      AN=A*TERM
      PN5=B*PN3-AN*PN1
      PN6=B*PN4-AN*PN2
      IF(PN6.EQ.ZERO)GOTO 60
      RN=PN5/PN6
      DIFF=DABS(RATIO-RN)
      IF(DIFF.LE.EPS.AND.DIFF.LE.EPS*RN)GOTO 80
      RATIO=RN
   60 PN1=PN3
      PN2=PN4
      PN3=PN5
      PN4=PN6
      IF(DABS(PN5).LT.OFL)GOTO 70
      PN1=PN1/OFL
      PN2=PN2/OFL
      PN3=PN3/OFL
      PN4=PN4/OFL
   70 CONTINUE
      WRITE(6,7020)
   80 ARG=ALPHA*DLOG(X)-X-G+DLOG(RATIO)
      GAMIND=ONE
      IF(ARG.GE.UFL)GAMIND=ONE-DEXP(ARG)
      RETURN
C
C         ALPHA IS LARGE: USE HILL'S APPROXIMATION (N.L. JOHNSON AND
C         S. KOTZ, 1970, 'CONTINUOUS UNIVARIATE DISTRIBUTIONS 1', P.180)
C
C         THE 'DO 110' LOOP CALCULATES 2*(X-ALPHA-ALPHA*DLOG(X/ALPHA)),
C         USING POWER-SERIES EXPANSION TO AVOID ROUNDING ERROR
C
  100 CONTINUE
      R=ONE/DSQRT(ALPHA)
      Z=(X-ALPHA)*R
      TERM=Z*Z
      SUM=HALF*TERM
      DO 110 I=1,12
      TERM=-TERM*Z*R
      SUM=SUM+TERM/(I+TWO)
      IF(DABS(TERM).LT.EPS)GOTO 120
  110 CONTINUE
  120 WW=TWO*SUM
      W=DSQRT(WW)
      IF(X.LT.ALPHA)W=-W
      H1=ONE/THREE
      H2=-W/X36
      H3=(-WW+X13)/X1620
      H4=(X42*WW+X119)*W/X38880
      Z=(((H4*R+H3)*R+H2)*R+H1)*R+W
      GAMIND=HALF+HALF*DERF(Z*RTHALF)
      RETURN
C
 1000 WRITE(6,7000)ALPHA
      RETURN
 1010 WRITE(6,7010)X
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE GAMIND :',
     *  ' SHAPE PARAMETER OUT OF RANGE :',D16.8)
 7010 FORMAT(' *** ERROR *** ROUTINE GAMIND :',
     *  ' ARGUMENT OF FUNCTION OUT OF RANGE :',D16.8)
 7020 FORMAT(' ** WARNING ** ROUTINE GAMIND :',
     *  ' ITERATION HAS NOT CONVERGED. RESULT MAY BE UNRELIABLE.')
      END