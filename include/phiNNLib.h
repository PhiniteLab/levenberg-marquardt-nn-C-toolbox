#ifndef __PHI_NN_LIB_H__
#define __PHI_NN_LIB_H__

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
    // define error types

    // define error types
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // input output data structure to store every variables

    typedef struct phinnlibparameters
    {
        /////////////////////////////////////////////////
        // standard NN parameters

        int trainingNumber; /*
                             * length of dataset in training process
                             */

        int numberOfInputLayerNode; /*
                                     * number of input layer node
                                     */

        int numberOfHiddenLayerNode; /*
                                     * number of hidden layer node
                                     */

        int numberOfOutputLayerNode; /*
                                     * number of output layer node
                                     */

        int I, H, K; // for shorthining code

        double devInvMatrices; /*
                             * Jacobian check condition
                             */

        double mu; /*
                   * learning rate for Levenberg Marquardt algorithm
                   */

        int nnTrainingCondition; /*
                                  * control variable to check NN training loop
                                  * it is used to terminate loop if the conditions are
                                  * satisfied!
                                  */

        FILE *nnOutputFileId; /*
                               * storage variable to keep the whole output of LM algorithm
                               */

        // standard NN parameters
        /////////////////////////////////////////////////

        /////////////////////////////////////////////////
        // neural network activation functions

        double **zActFunc; /*
                           * hidden activation function values
                           */

        double **yActFunc; /*
                           * output activation function values
                           */

        // neural network activation functions
        /////////////////////////////////////////////////

        /////////////////////////////////////////////////
        // neural network coefficients

        double **phiWmatrices; /*
                               * input to hidden vector coefficients
                               */

        double **phiVmatrices; /*
                               * hidden to output vector coefficients
                               */

        // neural network coefficients
        /////////////////////////////////////////////////

        /////////////////////////////////////////////////
        // vector parameters

        double **errorNow; /*
                           * error vector for the present values
                           */

        double **errorNowJac; /*
                           * error Jacobian vector for the present values
                           */

        double **errorPre; /*
                           * error vector for the past values
                           */

        double errorNowValue; /*
                              * the sum value of errorNow parameters
                              */

        double errorPreValue; /*
                              * the sum value of errorPre parameters
                              */

        // vector parameters
        /////////////////////////////////////////////////

        /////////////////////////////////////////////////
        // NN training termination parameters

        int iterationMax; /*
                           * maximum iteration number of loop
                           */

        double epsilonError; /*
                             * minimum value of error function
                             */

        double muMin; /*
                      * minimum value of mu parameter
                      */

        double maxDetInvMatrices; /*
                                  * maximum value of Jacobian determinant value
                                  */

        // NN training termination parameters
        /////////////////////////////////////////////////

        /////////////////////////////////////////////////
        // NN training Matrices

        double **JacobianTerm; /*
         * storing the total Jacobian matrices to update the coefficients
         */

        // NN training Matrices
        /////////////////////////////////////////////////

    } phiNNLibParameters;

    typedef phiNNLibParameters *phiNNLibParameterPtr;

    // input output data structure to store every variables
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // type decleration for usage of NN library

    /*
    * This structure represents the basic variables in terms of
    * neural network parameters.
    * 
    * There are only pointer types of object, the rest of them is
    * created in the nn blocks!
    * 
    */

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    void phiInitializeNeuralNetwork(phiInOutParametersPtr ptrInOut,
                                    phiNNLibParameterPtr ptrNN,
                                    int inputLayerNode,
                                    int hiddenLayerNode,
                                    int outputLayerNode); /*
    * initializing neural network for Levenberg marquadt structure
    * output -> returns nothing
    * input  -> address of phiInOutParameter
    *        -> address of phiNNLibParameterPtr
    *        -> value of input layer node
    *        -> value of hidden layer node
    *        -> value of output layer node
    * */

    void creatingActivationFunctions(phiNNLibParameterPtr ptrNN); /*
    * initializing NN activation functions
    * output -> returns nothing
    * input  -> address of phi NN lib structure
    * */

    double **phiRandInitialization(double min, double max, int firstNodeNumber, int secondNodeNumber); /*
    * initializing random NN coefficients
    * output -> returns address of W,V
    * input  -> value of minimum value
    *        -> value of maximum value
    *        -> value of first node number
    *        -> value of second node number
    * */

    void writeCoefficientMatrices(phiNNLibParameterPtr ptrNN); /*
    * write the coefficient matrices W,V
    * output -> returns nothings
    * input  -> address of phiNNlibParameters
    * */

    void phiTrainingParametersSettings(phiNNLibParameterPtr ptrNN,
                                       double epsilon,
                                       int iterationMax,
                                       double maxDetInvMatrices,
                                       double muMin); /*
    * parameters settings of training Neural Network
    * output -> returns nothings
    * input  -> address of phiNNlibParameters
    *           max error value
    *           max iteration step
    *           max Jacobian check
    *           max learning rate
    * */

    void phiTrainingNeuralNetworkWithLM(phiInOutParametersPtr ptrInOut,
                                        phiNNLibParameterPtr ptrNN); /*
    * training Neural Network with Levenberg-Marquardt optimization technique
    * output -> returns nothings
    * input  -> address of phiInOutParametersPtr
    *           address of phiNNlibParameters
    * */

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    // training internal functions

    void phiActivationCalculation(phiInOutParametersPtr ptrInOut,
                                  phiNNLibParameterPtr ptrNN, int index); /*
    * calculation of hidden layer outputs
    * output -> returns nothings
    * input  -> address of phiInOutParametersPtr
    *           address of phiNNlibParameters
    * */

    void phiOutputFunctionCalculation(phiInOutParametersPtr ptrInOut,
                                      phiNNLibParameterPtr ptrNN, int index); /*
    * calculation of output layer outputs
    * output -> returns nothings
    * input  -> address of phiInOutParametersPtr
    *           address of phiNNlibParameters
    * */

    void phiLmAlgorithm(phiInOutParametersPtr ptrInOut,
                        phiNNLibParameterPtr ptrNN, int index); /*
    * calculation total deflection in NN coefficient with Levenberg Marquardt
    * output -> returns nothings
    * input  -> address of phiInOutParametersPtr
    *           address of phiNNlibParameters
    * */

    void costFunctionCalculation(phiInOutParametersPtr ptrInOut,
                                 phiNNLibParameterPtr ptrNN, int index); /*
    * calculation of total error
    * output -> returns nothings
    * input  -> address of phiInOutParametersPtr
    *           address of phiNNlibParameters
    * */

    void costFunctionCalculationJacobian(phiInOutParametersPtr ptrInOut,
                                         phiNNLibParameterPtr ptrNN, int index);
    /*
    * calculation of single error
    * output -> returns nothings
    * input  -> address of phiInOutParametersPtr
    *           address of phiNNlibParameters
    * */

    // training internal functions
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    void printPhi(); /*
                      * simple test code to laod the whole library to the project
                      * output -> returns nothing
                      * input -> taking nothing
                      */

    void phiExitNNLib(); /*
                      * exiting the whole variables and functions
                      * output -> returns nothing
                      * input -> taking nothing
                      */

    // type decleration for usage of NN Library
    ////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif