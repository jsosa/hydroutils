C===================================================== QUAGAM.FOR
      DOUBLE PRECISION FUNCTION QUAGAM(F,PARA)
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
C  QUANTILE FUNCTION OF THE GAMMA DISTRIBUTION
C
C  OTHER ROUTINES USED: DERF,DLGAMA,GAMIND,QUASTN
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(2)
      DATA ZERO/0D0/,P01/0.01D0/,ONE/1D0/,NINE/9D0/
C
C         EPS,MAXIT CONTROL THE TEST FOR CONVERGENCE OF N-R ITERATION
C
      DATA EPS/1D-10/,MAXIT/30/
C
      QUAGAM=ZERO
      ALPHA=PARA(1)
      BETA=PARA(2)
      IF(ALPHA.LE.ZERO.OR.BETA.LE.ZERO)GOTO 1000
      IF(F.LT.ZERO.OR.F.GE.ONE)GOTO 1010
      IF(F.EQ.ZERO)RETURN
      AM1=ALPHA-ONE
      IF(AM1.NE.ZERO)GOTO 10
C
C         CASE ALPHA.EQ.1 - GAMMA IS EXPONENTIAL
C
      QUAGAM=(-DLOG(ONE-F))*BETA
      RETURN
C
C         INITIAL ESTIMATE OF ROOT OF EQUATION GAMIND(X)=F:
C         - IF ALPHA.GT.1, USE WILSON-HILFERTY APPROXIMATION IF IT'S
C           POSITIVE AND NOT TOO CLOSE TO ZERO;
C         - IF ALPHA.LT.1, OR IF W-H APPROX. ISN'T POSITIVE ENOUGH,
C           USE THE SMALL-X APPROXIMATION OF IGNORING THE EXP(-T) TERM
C           IN THE INTEGRAL DEFINING GAMIND(X)
C
   10 DLOGG=DLGAMA(ALPHA)
      IF(AM1.LE.ZERO)GOTO 20
      ROOT=ALPHA*(ONE-ONE/(NINE*ALPHA)+QUASTN(F)/DSQRT(NINE*ALPHA))**3
      IF(ROOT.GT.P01*ALPHA)GOTO 30
   20 ROOT=DEXP((DLOG(ALPHA*F)+DLOGG)/ALPHA)
   30 CONTINUE
C
C         REFINE INITIAL ESTIMATE BY NEWTON-RAPHSON ITERATION
C
      DO 40 IT=1,MAXIT
      FUNC=GAMIND(ROOT,ALPHA,DLOGG)-F
      RINC=FUNC*DEXP(DLOGG+ROOT-AM1*DLOG(ROOT))
      ROOT=ROOT-RINC
      IF(DABS(FUNC).LE.EPS)GOTO 50
   40 CONTINUE
      WRITE(6,7020)
C
C         SCALE SOLUTION
C
   50 QUAGAM=ROOT*BETA
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
 1010 WRITE(6,7010)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE QUAGAM : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE QUAGAM :',
     *  ' ARGUMENT OF FUNCTION INVALID')
 7020 FORMAT(' ** WARNING ** ROUTINE QUAGAM :',
     *  ' ITERATION HAS NOT CONVERGED. RESULT MAY BE UNRELIABLE')
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

C===================================================== QUASTN.FOR
      DOUBLE PRECISION FUNCTION QUASTN(F)
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
C*  * Fixed: WRITE(6,7000) and FORMAT statement 7000 incompatible      *
C*                                                                     *
C***********************************************************************
C
C  QUANTILE FUNCTION OF THE STANDARD NORMAL DISTRIBUTION
C
C  BASED ON ALGORITHM AS241, APPL. STATIST. (1988) VOL.37 NO.3
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/
      DATA SPLIT1/0.425D0/,SPLIT2/5D0/,CONST1/0.180625D0/,CONST2/1.6D0/
C
C         COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS
C
      DATA A0,A1,A2,A3,A4,A5,A6,A7,B1,B2,B3,B4,B5,B6,B7/
     *                                0.33871 32872 79636 661D  1,
     *  0.13314 16678 91784 377D  3,  0.19715 90950 30655 144D  4,
     *  0.13731 69376 55094 611D  5,  0.45921 95393 15498 715D  5,
     *  0.67265 77092 70087 009D  5,  0.33430 57558 35881 281D  5,
     *  0.25090 80928 73012 267D  4,  0.42313 33070 16009 113D  2,
     *  0.68718 70074 92057 908D  3,  0.53941 96021 42475 111D  4,
     *  0.21213 79430 15865 959D  5,  0.39307 89580 00927 106D  5,
     *  0.28729 08573 57219 427D  5,  0.52264 95278 85285 456D  4/
      DATA C0,C1,C2,C3,C4,C5,C6,C7,D1,D2,D3,D4,D5,D6,D7/
     *                                0.14234 37110 74968 358D  1,
     *  0.46303 37846 15654 530D  1,  0.57694 97221 46069 141D  1,
     *  0.36478 48324 76320 461D  1,  0.12704 58252 45236 838D  1,
     *  0.24178 07251 77450 612D  0,  0.22723 84498 92691 846D -1,
     *  0.77454 50142 78341 408D -3,  0.20531 91626 63775 882D  1,
     *  0.16763 84830 18380 385D  1,  0.68976 73349 85100 005D  0,
     *  0.14810 39764 27480 075D  0,  0.15198 66656 36164 572D -1,
     *  0.54759 38084 99534 495D -3,  0.10507 50071 64441 684D -8/
      DATA E0,E1,E2,E3,E4,E5,E6,E7,F1,F2,F3,F4,F5,F6,F7/
     *                                0.66579 04643 50110 378D  1,
     *  0.54637 84911 16411 437D  1,  0.17848 26539 91729 133D  1,
     *  0.29656 05718 28504 891D  0,  0.26532 18952 65761 230D -1,
     *  0.12426 60947 38807 844D -2,  0.27115 55568 74348 758D -4,
     *  0.20103 34399 29228 813D -6,  0.59983 22065 55887 938D  0,
     *  0.13692 98809 22735 805D  0,  0.14875 36129 08506 149D -1,
     *  0.78686 91311 45613 259D -3,  0.18463 18317 51005 468D -4,
     *  0.14215 11758 31644 589D -6,  0.20442 63103 38993 979D-14/
C
      Q=F-HALF
      IF(DABS(Q).GT.SPLIT1)GOTO 10
      R=CONST1-Q*Q
      QUASTN=Q*(((((((A7*R+A6)*R+A5)*R+A4)*R+A3)*R+A2)*R+A1)*R+A0)
     *        /(((((((B7*R+B6)*R+B5)*R+B4)*R+B3)*R+B2)*R+B1)*R+ONE)
      RETURN
   10 R=F
      IF(Q.GE.ZERO)R=ONE-F
      IF(R.LE.ZERO)GOTO 1000
      R=DSQRT(-DLOG(R))
      IF(R.GT.SPLIT2)GOTO 20
      R=R-CONST2
      QUASTN=(((((((C7*R+C6)*R+C5)*R+C4)*R+C3)*R+C2)*R+C1)*R+C0)
     *      /(((((((D7*R+D6)*R+D5)*R+D4)*R+D3)*R+D2)*R+D1)*R+ONE)
      GOTO 30
   20 R=R-SPLIT2
      QUASTN=(((((((E7*R+E6)*R+E5)*R+E4)*R+E3)*R+E2)*R+E1)*R+E0)
     *      /(((((((F7*R+F6)*R+F5)*R+F4)*R+F3)*R+F2)*R+F1)*R+ONE)
   30 IF(Q.LT.ZERO)QUASTN=-QUASTN
      RETURN
C
 1000 WRITE(6,7000)
      QUASTN=ZERO
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE QUASTN :',
     *  ' ARGUMENT OF FUNCTION INVALID')
      END