C===================================================== LMRGNO.FOR
      SUBROUTINE LMRGNO(PARA,XMOM,NMOM)
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
C  L-MOMENT RATIOS FOR THE GENERALIZED NORMAL DISTRIBUTION
C
C  PARAMETERS OF ROUTINE:
C  PARA   * INPUT* ARRAY OF LENGTH 3. CONTAINS THE PARAMETERS OF THE
C                  DISTRIBUTION, IN THE ORDER XI, ALPHA, K (LOCATION,
C                  SCALE, SHAPE).
C  XMOM   *OUTPUT* ARRAY OF LENGTH NMOM. ON EXIT, CONTAINS THE L-MOMENTS
C                  LAMBDA-1, LAMBDA-2, TAU-3, TAU-4, ... .
C  NMOM   * INPUT* NUMBER OF L-MOMENTS TO BE FOUND. AT MOST 20.
C
C  OTHER ROUTINES USED: DERF
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(3),XMOM(NMOM),EST(20),ESTX(20),SUM(20),
     *  ZMOM(20)
      DATA ZERO/0D0/,HALF/0.5D0/,ONE/1D0/
C
C         ARRAY ZMOM CONTAINS L-MOMENTS OF THE STANDARD NORMAL DIST.
C
      DATA ZMOM/
     *  0D0,   0.56418 95835 47756 287D 0,
     *  0D0,   0.12260 17195 40890 947D 0,
     *  0D0,   0.43661 15389 50024 944D-1,
     *  0D0,   0.21843 13603 32508 776D-1,
     *  0D0,   0.12963 50158 01507 746D-1,
     *  0D0,   0.85296 21241 91705 402D-2,
     *  0D0,   0.60138 90151 79323 333D-2,
     *  0D0,   0.44555 82586 47650 150D-2,
     *  0D0,   0.34264 32435 78076 985D-2,
     *  0D0,   0.27126 79630 48139 365D-2/
C
C         RRT2 IS 1/SQRT(2), RRTPI IS 1/SQRT(PI)
C
      DATA RRT2 /0.70710 67811 86547 524D0/
      DATA RRTPI/0.56418 95835 47756 287D0/
C
C         RANGE,EPS,MAXIT CONTROL THE ITERATIVE PROCEDURE FOR NUMERICAL
C         INTEGRATION
C
      DATA RANGE/5D0/,EPS/1D-8/,MAXIT/10/
C
      U=PARA(1)
      A=PARA(2)
      G=PARA(3)
      IF(A.LE.ZERO)GOTO 1000
      IF(NMOM.GT.20)GOTO 1010
C
C         TEST FOR K=0
C
      IF(DABS(G).GT.EPS)GOTO 5
      XMOM(1)=U
      IF(NMOM.EQ.1)RETURN
      XMOM(2)=A*ZMOM(2)
      IF(NMOM.EQ.2)RETURN
      DO 2 I=3,NMOM
    2 XMOM(I)=ZMOM(I)
      RETURN
    5 CONTINUE
C
C         LAMBDA-1
C
      EGG=DEXP(HALF*G*G)
      ALAM1=(ONE-EGG)/G
      XMOM(1)=U+A*ALAM1
      IF(NMOM.EQ.1)RETURN
C
C         LAMBDA-2
C
      ALAM2=EGG*DERF(HALF*G)/G
      XMOM(2)=A*ALAM2
      IF(NMOM.EQ.2)RETURN
C
C         HIGHER MOMENTS. THE INTEGRAL DEFINING LAMBDA-R IS EVALUATED
C         BY ITERATIVE APPLICATION OF THE TRAPEZIUM RULE.
C
C         - INITIAL ESTIMATE, USING 16 ORDINATES  (THE 'DO 20' LOOP
C           CALCULATES LEGENDRE POLYNOMIALS RECURSIVELY)
C
      CC=-G*RRT2
      XMIN=CC-RANGE
      XMAX=CC+RANGE
      DO 10 M=3,NMOM
   10 SUM(M)=ZERO
      N=16
      XINC=(XMAX-XMIN)/N
      DO 30 I=1,N-1
      X=XMIN+I*XINC
      E=DEXP(-((X-CC)**2))
      D=DERF(X)
      P1=ONE
      P=D
      DO 20 M=3,NMOM
      C1=M+M-3
      C2=M-2
      C3=M-1
      P2=P1
      P1=P
      P=(C1*D*P1-C2*P2)/C3
   20 SUM(M)=SUM(M)+E*P
   30 CONTINUE
      DO 40 M=3,NMOM
   40 EST(M)=SUM(M)*XINC
C
C         - DOUBLE THE NUMBER OF ORDINATES UNTIL CONVERGED
C
      DO 90 IT=1,MAXIT
      DO 50 M=3,NMOM
   50 ESTX(M)=EST(M)
      N=N*2
      XINC=(XMAX-XMIN)/N
      DO 70 I=1,N-1,2
      X=XMIN+I*XINC
      E=DEXP(-((X-CC)**2))
      D=DERF(X)
      P1=ONE
      P=D
      DO 60 M=3,NMOM
      C1=M+M-3
      C2=M-2
      C3=M-1
      P2=P1
      P1=P
      P=(C1*D*P1-C2*P2)/C3
   60 SUM(M)=SUM(M)+E*P
   70 CONTINUE
C
C         --- TEST FOR CONVERGENCE
C
      NOTCGD=0
      DO 80 M=NMOM,3,-1
      EST(M)=SUM(M)*XINC
      IF(DABS(EST(M)-ESTX(M)).GT.EPS*DABS(EST(M)))NOTCGD=M
   80 CONTINUE
      IF(NOTCGD.EQ.0)GOTO 100
   90 CONTINUE
C
      WRITE(6,7020)NOTCGD-1
  100 CONTINUE
      CONST=-DEXP(CC*CC)*RRTPI/(ALAM2*G)
      DO 110 M=3,NMOM
  110 XMOM(M)=CONST*EST(M)
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
 1010 WRITE(6,7010)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE LMRGNO : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE LMRGNO : PARAMETER NMOM TOO LARGE')
 7020 FORMAT(' ** WARNING ** ROUTINE LMRGNO :',
     *  ' ITERATION HAS NOT CONVERGED. ONLY THE FIRST',I3,
     *  ' L-MOMENTS ARE RELIABLE.')
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