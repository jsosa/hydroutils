C===================================================== LMRGLO.FOR
      SUBROUTINE LMRGLO(PARA,XMOM,NMOM)
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
C  L-MOMENT RATIOS FOR THE GENERALIZED LOGISTIC DISTRIBUTION
C
C  PARAMETERS OF ROUTINE:
C  PARA   * INPUT* ARRAY OF LENGTH 3. CONTAINS THE PARAMETERS OF THE
C                  DISTRIBUTION, IN THE ORDER XI, ALPHA, K (LOCATION,
C                  SCALE, SHAPE).
C  XMOM   *OUTPUT* ARRAY OF LENGTH NMOM. ON EXIT, CONTAINS THE L-MOMENTS
C                  LAMBDA-1, LAMBDA-2, TAU-3, TAU-4, ... .
C  NMOM   * INPUT* NUMBER OF L-MOMENTS TO BE FOUND. AT MOST 20.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION PARA(3),XMOM(NMOM),Z(10,20)
      DATA ZERO/0D0/,ONE/1D0/
      DATA PI/3.141592653589793238D0/
C
C         SMALL IS USED TO DECIDE WHETHER TO APPROXIMATE THE FIRST 2
C         L-MOMENTS BY A POWER-SERIES EXPANSION WHEN G IS NEAR ZERO.
C         C1,C2 ARE COEFFICIENTS OF THIS POWER-SERIES EXPANSION.
C         C1 IS PI**2/6, C2 IS 7*PI**4/360.
C
      DATA SMALL/1D-4/
      DATA C1,C2/
     *  0.16449 34066 84822 644D 1,  0.18940 65658 99449 184D 1/
C
C         Z-ARRAY CONTAINS COEFFICIENTS OF THE REPRESENTATIONS OF
C         L-MOMENT RATIOS AS POLYNOMIALS IN THE SHAPE PARAMETER K
C
      DATA Z(1,3)/1D0/
      DATA (Z(I, 4),I=1, 2)/
     *  0.16666 66666 66666 667D 0,  0.83333 33333 33333 333D 0/
      DATA (Z(I, 5),I=1, 2)/
     *  0.41666 66666 66666 667D 0,  0.58333 33333 33333 333D 0/
      DATA (Z(I, 6),I=1, 3)/
     *  0.66666 66666 66666 667D-1,  0.58333 33333 33333 333D 0,
     *  0.35000 00000 00000 000D 0/
      DATA (Z(I, 7),I=1, 3)/
     *  0.23333 33333 33333 333D 0,  0.58333 33333 33333 333D 0,
     *  0.18333 33333 33333 333D 0/
      DATA (Z(I, 8),I=1, 4)/
     *  0.35714 28571 42857 143D-1,  0.42083 33333 33333 333D 0,
     *  0.45833 33333 33333 333D 0,  0.85119 04761 90476 190D-1/
      DATA (Z(I, 9),I=1, 4)/
     *  0.15099 20634 92063 492D 0,  0.51562 50000 00000 000D 0,
     *  0.29791 66666 66666 667D 0,  0.35466 26984 12698 413D-1/
      DATA (Z(I,10),I=1, 5)/
     *  0.22222 22222 22222 222D-1,  0.31889 32980 59964 727D 0,
     *  0.47997 68518 51851 852D 0,  0.16550 92592 59259 259D 0,
     *  0.13398 36860 67019 400D-1/
      DATA (Z(I,11),I=1, 5)/
     *  0.10650 79365 07936 508D 0,  0.44766 31393 29805 996D 0,
     *  0.36081 01851 85185 185D 0,  0.80390 21164 02116 402D-1,
     *  0.46285 27336 86067 019D-2/
      DATA (Z(I,12),I=1, 6)/
     *  0.15151 51515 15151 515D-1,  0.25131 61375 66137 566D 0,
     *  0.46969 52160 49382 716D 0,  0.22765 04629 62962 963D 0,
     *  0.34713 95502 64550 265D-1,  0.14727 13243 54657 688D-2/
      DATA (Z(I,13),I=1, 6)/
     *  0.79569 50456 95045 695D-1,  0.38976 59465 02057 613D 0,
     *  0.39291 73096 70781 893D 0,  0.12381 31062 61022 928D 0,
     *  0.13499 87139 91769 547D-1,  0.43426 15974 56041 900D-3/
      DATA (Z(I,14),I=1, 7)/
     *  0.10989 01098 90109 890D-1,  0.20413 29966 32996 633D 0,
     *  0.44773 66255 14403 292D 0,  0.27305 34428 27748 383D 0,
     *  0.59191 74382 71604 938D-1,  0.47768 77572 01646 091D-2,
     *  0.11930 26366 63747 775D-3/
      DATA (Z(I,15),I=1, 7)/
     *  0.61934 52050 59490 774D-1,  0.34203 17593 92870 504D 0,
     *  0.40701 37051 73427 396D 0,  0.16218 91928 06752 331D 0,
     *  0.25249 21002 35155 791D-1,  0.15509 34276 62872 107D-2,
     *  0.30677 82085 63922 850D-4/
      DATA (Z(I,16),I=1, 8)/
     *  0.83333 33333 33333 333D-2,  0.16976 83649 02293 474D 0,
     *  0.42219 12828 68366 202D 0,  0.30542 71728 94620 811D 0,
     *  0.84082 79399 72285 210D-1,  0.97243 57914 46208 113D-2,
     *  0.46528 02829 88616 322D-3,  0.74138 06706 96146 887D-5/
      DATA (Z(I,17),I=1, 8)/
     *  0.49716 60284 16028 416D-1,  0.30276 58385 89871 328D 0,
     *  0.41047 33000 89185 506D 0,  0.19483 90265 03251 764D 0,
     *  0.38659 80637 04648 526D-1,  0.34139 94076 42897 226D-2,
     *  0.12974 16173 71825 705D-3,  0.16899 11822 91033 482D-5/
      DATA (Z(I,18),I=1, 9)/
     *  0.65359 47712 41830 065D-2,  0.14387 48475 95085 690D 0,
     *  0.39643 28537 10259 464D 0,  0.32808 41807 20899 471D 0,
     *  0.10797 13931 65194 318D 0,  0.15965 33699 32077 769D-1,
     *  0.11012 77375 69143 819D-2,  0.33798 23645 82066 963D-4,
     *  0.36449 07853 33601 627D-6/
      DATA (Z(I,19),I=1, 9)/
     *  0.40878 45705 49276 431D-1,  0.27024 42907 25441 519D 0,
     *  0.40759 95245 14551 521D 0,  0.22211 14264 89320 008D 0,
     *  0.52846 38846 29533 398D-1,  0.59829 82392 72872 761D-2,
     *  0.32859 39655 65898 436D-3,  0.82617 91134 22830 354D-5,
     *  0.74603 37711 50646 605D-7/
      DATA (Z(I,20),I=1,10)/
     *  0.52631 57894 73684 211D-2,  0.12381 76557 53054 913D 0,
     *  0.37185 92914 44794 917D 0,  0.34356 87476 70189 607D 0,
     *  0.13019 86628 12524 058D 0,  0.23147 43648 99477 023D-1,
     *  0.20519 25194 79869 981D-2,  0.91205 82581 07571 930D-4,
     *  0.19023 86116 43414 884D-5,  0.14528 02606 97757 497D-7/
C
      U=PARA(1)
      A=PARA(2)
      G=PARA(3)
      IF(A.LE.ZERO.OR.DABS(G).GE.ONE)GOTO 1000
      IF(NMOM.GT.20)GOTO 1010
C
C         FIRST 2 MOMENTS
C
      GG=G*G
      ALAM1=-G*(C1+GG*C2)
      ALAM2=ONE+GG*(C1+GG*C2)
      IF(DABS(G).GT.SMALL)ALAM2=G*PI/DSIN(G*PI)
      IF(DABS(G).GT.SMALL)ALAM1=(ONE-ALAM2)/G
      XMOM(1)=U+A*ALAM1
      IF(NMOM.EQ.1)RETURN
      XMOM(2)=A*ALAM2
      IF(NMOM.EQ.2)RETURN
C
C         HIGHER MOMENTS
C
      DO 20 M=3,NMOM
      KMAX=M/2
      SUM=Z(KMAX,M)
      DO 10 K=KMAX-1,1,-1
   10 SUM=SUM*GG+Z(K,M)
      IF(M.NE.M/2*2)SUM=-G*SUM
      XMOM(M)=SUM
   20 CONTINUE
      RETURN
C
 1000 WRITE(6,7000)
      RETURN
 1010 WRITE(6,7010)
      RETURN
C
 7000 FORMAT(' *** ERROR *** ROUTINE LMRGLO : PARAMETERS INVALID')
 7010 FORMAT(' *** ERROR *** ROUTINE LMRGLO : PARAMETER NMOM TOO LARGE')
      END