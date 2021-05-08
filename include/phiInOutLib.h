#ifndef __PHI_IN_OUT_LIB_H__
#define __PHI_IN_OUT_LIB_H__

#ifdef __cplusplus
extern "C"
{
#endif

    ////////////////////////////////////////////////////////////
    // including some libraries for using input/output functions

// standard library usage
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"

    // phinite Library usage
#include "phiNNLibSettings.h"

    // including some libraries for using input/output functions
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // define error types

#define ALLOCATION_ERROR 1
#define INCONSISTENT_ROW_COLUMN 2
#define FILE_OPEN_ERROR 3
#define SAMPLING_RATE_ERROR 4

    // define error types
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // input output data structure to store every variables

    typedef struct phiinoutparameters
    {
        ////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////
        // input file information

        FILE *inputFile; /*
                          * creating an inputfile to read the file
                          */
        char *fileName;  /*
                          * storing the name of file
                          */

        int lengthNNDataset; /*
                           * row value of NN dataset
                           */

        int numberNNInputDataset;  /*
                                   * column value of input NN dataset
                                   */
        int numberNNOutputDataset; /*
                                   * column value of output NN dataset
                                   */

        // input file information
        ////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////
        // storing the whole elements in one pointer

        double **inputDataArray; /*
                                 * storing the whole input elements in one dynamic array
                                 */

        double **outputDataArray; /*
                                 * storing the whole output elements in one dynamic array
                                 */

        // storing the whole elements in one pointer
        ////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////

    } phiInOutParameters;

    typedef phiInOutParameters *phiInOutParametersPtr;

    // input output data structure to store every variables
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // type decleration for usage of input output section library

    /*
    * This structure represents the basic variables of input/output
    sessions in the other files such as excel, text, etc.
    */

    void printPhi(); /*
                      * simple test code to laod the whole library to the project
                      * output -> returns nothing
                      * input -> taking nothing
                      */

    void readFileFromText(phiInOutParametersPtr ptrInOut); /*
                        * reading the whole data for a given text file
                        * output -> returns nothing
                        * input -> address of InOutparameters structure 
                        */

    void readFileFromCSV(phiInOutParametersPtr ptrInOut); /*
                        * reading the whole data for a given CSV file
                        * output -> returns nothing
                        * input -> address of InOutparameters structure 
                        */

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////

    double **creatingEmptyInputMatrices(phiInOutParametersPtr ptrInOut); /*
    * creating dynamically empty input matrices for data storage
    * output -> address of matrices (float **)
    * input  -> address of phinioutparameters
    * */

    double **creatingEmptyOutputMatrices(phiInOutParametersPtr ptrInOut); /*
    * creating dynamically empty output matrices for data storage
    * output -> address of matrices (float **)
    * input  -> address of phinioutparameters
    * */

    void fillInputAndOutputMatrices(phiInOutParametersPtr ptrInOut); /*
    * filling the whole data to the matrices
    * output -> returns nothing
    * input  -> address of phinioutparameters
    * */

    void writeDataSetMatrices(phiInOutParametersPtr ptrInOut); /*
    * writing the whole data to the matrices
    * output -> returns nothing
    * input  -> address of phinioutparameters
    * */

    double **creatingEmptyMatrices(int rows,
                                   int cols); /*
    * creating dynamically empty matrices for general usage
    * output -> address of empty matrices
    * input  -> rows and columns values of matrices
    * */

    int **creatingEmptyMatricesIntegralType(int rows,
                                            int cols); /*
    * creating dynamically empty matrices for general usage
    * output -> address of empty matrices
    * input  -> rows and columns values of matrices
    * */

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////

    void phiFree(double **pd,
                 int row,
                 int col); /*
    * free the whole memory allocation
    * output -> return nothing
    * input  -> address of memory
    *           row value of matrices
    *           column value of matrices
    * */

    void phiFreeIntegralType(int **pd,
                             int row,
                             int col); /*
    * free the whole memory allocation
    * output -> return nothing
    * input  -> address of memory
    *           row value of matrices
    *           column value of matrices
    * */

    void phiExitInOut(phiInOutParametersPtr ptrInOut); /*
                      * exiting the whole variables and functions
                      * output -> returns nothing
                      * input -> address of ptrInOut
                      */

    void phiErrorHandler(int errorType); /*
    * notifying the error result
    * output -> return nothing
    * input  -> type of error
    * */

    //////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////

    void readingDemo1(); /*
                        * reading the data from csv or text file
                        * */

    //////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////

    // type decleration for usage of NN Library
    ////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif