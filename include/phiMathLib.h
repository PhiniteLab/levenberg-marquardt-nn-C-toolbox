#ifndef __PHI_MATH_LIB__
#define __PHI_MATH_LIB__

// in order to generalize our code, we need to use
#ifdef __cplusplus
extern "C"
{
#endif

    ////////////////////////////////////////////////////////////
    // including some libraries for using input/output functions

#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#include "phiNNLibSettings.h"

    // including some libraries for using input/output functions
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // define constants

#define phi_PI acos(-1.0)

    // define constants
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // function prototype decleration

    double **phiVectorMatrixMultiplication(double **firstTerm,
                                           double **SecondTerm,
                                           int row1,
                                           int col1,
                                           int row2,
                                           int col2); /*
    * creating dynamically matrices produced by multiplication
    * output -> address of multiplied matrices
    * input  -> address of first Matrices
    *           address of second Matrices
    *           row of first Matrices
    *           col of first Matrices
    *           row of second Matrices
    *           col of second Matrices
    * */

    double **phiSkalarMatrixMultiplication(double skalarTerm,
                                           double **SecondTerm,
                                           int row,
                                           int col); /*
    * creating dynamically matrices produced by skalar multiplication
    * output -> address of multiplied matrices
    * input  -> value of skalar term
    *           address of second Matrices
    *           row of second Matrices
    *           col of second Matrices
    * */

    double **phiMatrixSummation(double **firstTerm,
                                double **SecondTerm,
                                int row,
                                int col); /*
    * creating dynamically matrices produced by summation
    * output -> address of summed matrices
    * input  -> address of first Matrices
    *           address of second Matrices
    *           row of second Matrices
    *           col of second Matrices
    * */

    double phiDeterminationCalculation(double **phiMatrices,
                                       int row,
                                       int col); /*
    * creating dynamically matrices resulting with determinant
    * output -> address of summed matrices
    * input  -> address of first Matrices
    *           row of second Matrices
    *           col of second Matrices
    * */

    void phiTranspose(double **phiMatrices, int row, int col, double **phiTransMatrices); /*
    * creating inverse matrices of phiMatrices
    * output -> address of summed matrices
    * input  -> address of first Matrices
    *           value of row
    *           value of col
    *           address of transpose matrices
    * */

    void phiMatrixAssignment(double **assignedTerm,
                             double **SecondTerm,
                             int row,
                             int col); /*
    * creating dynamically matrices to equalize these terms
    * output -> address of summed matrices
    * input  -> address of first Matrices
    *           address of second Matrices
    *           row of second Matrices
    *           col of second Matrices
    * */

    double phiRand(double min, double max); /*
    * generating random number in float number for a given min/max
    * output -> value of float number
    * input  -> value of min value
    *           value of max value
    * */

    double phiSumMatrices(double **pd, int row, int col); /*
    * summation value of matrices
    * output -> value of sum number
    * input  -> address of matrices
    *           value of row
    *           value of column
    * */

    double phiMeanMatrices(double **pd, int row, int col); /*
    * mean value of matrices
    * output -> value of sum number
    * input  -> address of matrices
    *           value of row
    *           value of column
    * */

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////

    double **phiEyeCreationMatrices(int row); /*
   * creating eyematrices for a given row value
   * output -> address of this vector
   * input -> value of row
   */

    ////////////////////////////////////////////////////////////
    // inverse operation
    double **phiInverseMatrices(double **phiMatrices, int row, int col); /*
    * creating inverse matrices of phiMatrices
    * output -> address inverseMatrices
    * input  -> address of first Matrices
    *           value of row
    *           value of col
    * */

    void PrintMatrix(double **ar, int n, int m);

    void PrintInverse(double **ar, int n, int m);

    double **InverseOfMatrix(double **matrix, int order);

    // function prototype decleration
    ////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif