#include "..\include\phiNNLibSettings.h"

////////////////////////////////////////////////////////////////
// neural network structure

void phiInitializeNeuralNetwork(phiInOutParametersPtr ptrInOut,
                                phiNNLibParameterPtr ptrNN,
                                int inputLayerNode,
                                int hiddenLayerNode,
                                int outputLayerNode)
{
    // set training number value
    ptrNN->trainingNumber = ptrInOut->lengthNNDataset;

    // set the ptrNN parameters to use in other sections
    ptrNN->numberOfHiddenLayerNode = hiddenLayerNode;
    ptrNN->numberOfInputLayerNode = ptrInOut->numberNNInputDataset;
    ptrNN->numberOfOutputLayerNode = ptrInOut->numberNNOutputDataset;

    // for summary operation NN library

    ptrNN->I = ptrNN->numberOfInputLayerNode;
    ptrNN->H = ptrNN->numberOfHiddenLayerNode;
    ptrNN->K = ptrNN->numberOfOutputLayerNode;

    // initialize NN coefficients
    ptrNN->phiWmatrices = phiRandInitialization(-10, 10, ptrNN->H, ptrNN->I);
    ptrNN->phiVmatrices = phiRandInitialization(-10, 10, ptrNN->K, ptrNN->H + 1);
    printf("NN coefficients are created with random values\n");

    //writeCoefficientMatrices(ptrNN);

    // initialize error matrices
    ptrNN->errorNow = phiRandInitialization(0, 1, ptrNN->trainingNumber, ptrNN->K);
    ptrNN->errorPre = phiRandInitialization(0, 1, ptrNN->trainingNumber, ptrNN->K);
    ptrNN->errorNowJac = phiRandInitialization(0, 1, ptrNN->trainingNumber, ptrNN->K);
    printf("Error vectors are created with random values\n");

    ptrNN->errorNowValue = phiMeanMatrices(ptrNN->errorNow, ptrNN->trainingNumber, ptrNN->K);
    ptrNN->errorPreValue = phiMeanMatrices(ptrNN->errorPre, ptrNN->trainingNumber, ptrNN->K);

    // creating NN stucture with proper activation functions
    creatingActivationFunctions(ptrNN);

    // training condition is set to the start position!
    ptrNN->nnTrainingCondition = 1;

    ptrNN->mu = 0.08;

    // Jacobian update function is created
    ptrNN->JacobianTerm = creatingEmptyMatrices(ptrNN->trainingNumber,
                                                (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    printf("Activation functions are created with random values\n\n\n");

    printf("CONGRATULATIONS, NEURAL NETWORK STRUCTURE HAS BEEN ESTABLISHED!\n\n");

    writeCoefficientMatrices(ptrNN);
}

void creatingActivationFunctions(phiNNLibParameterPtr ptrNN)
{
    ptrNN->zActFunc = phiRandInitialization(0, 0, ptrNN->H + 1, ptrNN->trainingNumber);
    ptrNN->yActFunc = phiRandInitialization(0, 0, ptrNN->K, ptrNN->trainingNumber);
}

void phiTrainingParametersSettings(phiNNLibParameterPtr ptrNN,
                                   double epsilon,
                                   int iterationMax,
                                   double maxDetInvMatrices,
                                   double muMin)
{
    ptrNN->epsilonError = epsilon;
    ptrNN->iterationMax = iterationMax;
    ptrNN->maxDetInvMatrices = maxDetInvMatrices;
    ptrNN->muMin = muMin;

    printf("Neural Network Training Parameters:\n");
    printf("Minimum Error : %lf, Maximum Iteration : %d, Jacobian Check : %g, Minimum Mu : %lf",
           epsilon,
           iterationMax,
           maxDetInvMatrices,
           muMin);

    printf("\n\nThe whole parameters settings is done!\n\n");
}

void phiTrainingNeuralNetworkWithLM(phiInOutParametersPtr ptrInOut,
                                    phiNNLibParameterPtr ptrNN)
{
    // storing iteration number
    int iteration = 0;

    // transpose storing Jacobian terms / indices are changed!
    double **JacobianTermTrans = creatingEmptyMatrices((ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
                                                       ptrNN->trainingNumber);

    // multiplication of JacobianTerms
    double **JacobianMultiplicationResult = creatingEmptyMatrices(
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    double **eyeMatrices = creatingEmptyMatrices(
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    double **muEyeMatrices = creatingEmptyMatrices(
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    double **JacobianFullTerm = creatingEmptyMatrices(
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    double **InvJacobianFullTerm = creatingEmptyMatrices(
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    double **coeffIntUpdate = creatingEmptyMatrices(ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                    ptrNN->trainingNumber);

    double **coeffUpdate = creatingEmptyMatrices(
        (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
        1);

    while (ptrNN->nnTrainingCondition != 0)
    {
        // increasing iteration number
        iteration++;
        phiMatrixAssignment(ptrNN->errorPre, ptrNN->errorNow, ptrNN->trainingNumber, ptrNN->K);

        for (int i = 0; i < ptrNN->trainingNumber; i++)
        {
            phiActivationCalculation(ptrInOut, ptrNN, i);

            phiOutputFunctionCalculation(ptrInOut, ptrNN, i);

            phiLmAlgorithm(ptrInOut, ptrNN, i);

            costFunctionCalculation(ptrInOut, ptrNN, i);

            costFunctionCalculationJacobian(ptrInOut, ptrNN, i);
        }

        // determinant calculation for devIntMatrices
        phiTranspose(ptrNN->JacobianTerm, ptrNN->trainingNumber,
                     (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
                     JacobianTermTrans);

        //printf("Phitranspose is done!\n\n");
        JacobianMultiplicationResult = phiVectorMatrixMultiplication(JacobianTermTrans,
                                                                     ptrNN->JacobianTerm,
                                                                     (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
                                                                     ptrNN->trainingNumber,
                                                                     ptrNN->trainingNumber,
                                                                     (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));
        //printf("Jacobian multiplication is done!\n\n");

        eyeMatrices = phiEyeCreationMatrices(ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1));

        //printf("Eye matrices is done!\n\n");

        muEyeMatrices = phiSkalarMatrixMultiplication(ptrNN->mu, eyeMatrices,
                                                      ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                      ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1));

        //printf("Mu Eye matrices is done!\n\n");

        JacobianFullTerm = phiMatrixSummation(JacobianMultiplicationResult,
                                              muEyeMatrices,
                                              ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                              ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1));

        //printf("Jacobian Full Term is done!\n\n");

        //printf("Inv matrices\n\n");
        InvJacobianFullTerm = phiInverseMatrices(JacobianFullTerm,
                                                 ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                 ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1));

        //printf("Internal update coeff is being calculated..\n");

        coeffIntUpdate = phiVectorMatrixMultiplication(InvJacobianFullTerm,
                                                       JacobianTermTrans,
                                                       ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                       ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                       (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
                                                       ptrNN->trainingNumber);

        coeffUpdate = phiVectorMatrixMultiplication(coeffIntUpdate, ptrNN->errorNowJac,
                                                    ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                    ptrNN->trainingNumber,
                                                    ptrNN->trainingNumber, 1);

        // internal terms
        phiFree(coeffIntUpdate, ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1), ptrNN->trainingNumber);
        //printf("Total update coeff is being calculated..\n");

        ptrNN->devInvMatrices = phiDeterminationCalculation(InvJacobianFullTerm,
                                                            ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1),
                                                            ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1));
        /*
        for (int i = 0; i < ptrNN->trainingNumber; i++)
        {
            for (int j = 0; j < ptrNN->K; j++)
            {
                printf("%lf ", ptrNN->errorNow[i][j]);
            }
            printf("\n");
        }*/

        /////////////////////////////////////////////////////////////
        // LM update to coefficients
        int counter = 0;

        for (int k = 0; k < ptrNN->K; k++)
        {
            for (int h = 0; h < ptrNN->H + 1; h++)
            {
                ptrNN->phiVmatrices[k][h] -= coeffUpdate[counter][0];
                counter++;
            }
        }

        for (int h = 0; h < ptrNN->H; h++)
        {
            for (int in = 0; in < ptrNN->I; in++)
            {
                ptrNN->phiWmatrices[h][in] -= coeffUpdate[counter][0];

                counter++;
            }
        }

        // LM update to coefficients
        /////////////////////////////////////////////////////////////

        ptrNN->errorNowValue = phiMeanMatrices(ptrNN->errorNow, ptrNN->trainingNumber, ptrNN->K);
        ptrNN->errorPreValue = phiMeanMatrices(ptrNN->errorPre, ptrNN->trainingNumber, ptrNN->K);

        if ((ptrNN->errorPreValue - ptrNN->errorNowValue) > 0)
        {
            double internalAssesmentMu = (ptrNN->errorPreValue - ptrNN->errorNowValue);

            if (fabs(internalAssesmentMu) > (1e-4 / (ptrNN->trainingNumber)))
            {
                ptrNN->mu = ptrNN->mu - ptrNN->mu * 0.01;
            }
            else
            {
                ptrNN->mu = ptrNN->mu + ptrNN->mu * 0.01;
            }
        }

        if ((iteration % 1) == 0)
        {
            printf("\nIteration : %d, error: %e, Jac: %e, mu: %e\n",
                   iteration,
                   ptrNN->errorNowValue,
                   ptrNN->devInvMatrices,
                   ptrNN->mu);
            //writeCoefficientMatrices(ptrNN);
            /*
            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1); j++)
                {
                    printf("%e ", coeffUpdate[j][i]);
                }
                printf("\n");
            }*/
        }

        if ((ptrNN->errorPreValue - ptrNN->errorNowValue) >= 0)
        {
            double internalAssesmentMu = (ptrNN->errorPreValue - ptrNN->errorNowValue);

            if (fabs(internalAssesmentMu) > ((1e-1) / (ptrNN->trainingNumber)))
            {
                ptrNN->mu = ptrNN->mu + ptrNN->mu * 0.1;
            }
            else
            {
                ptrNN->mu = ptrNN->mu - ptrNN->mu * 0.1;
            }
        }

        ptrNN->nnTrainingCondition = (iteration < ptrNN->iterationMax) &&
                                     (ptrNN->errorNowValue > ptrNN->epsilonError);

        double coeffControl = phiSumMatrices(coeffUpdate, ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1), 1);

        if ((ptrNN->errorPreValue - ptrNN->errorNowValue) < (ptrNN->epsilonError * ptrNN->epsilonError) &&
                (iteration > 1000) ||
            (_isnan(coeffControl) != 0) || (ptrNN->errorNowValue > 500 * ptrNN->H))
        {
            phiFree(ptrNN->phiWmatrices, ptrNN->H, ptrNN->I);
            phiFree(ptrNN->phiVmatrices, ptrNN->K, ptrNN->H + 1);
            phiFree(ptrNN->errorNow, ptrNN->trainingNumber, ptrNN->K);
            phiFree(ptrNN->errorPre, ptrNN->trainingNumber, ptrNN->K);
            phiFree(ptrNN->errorNowJac, ptrNN->trainingNumber, ptrNN->K);

            phiFree(ptrNN->zActFunc, ptrNN->H + 1, ptrNN->trainingNumber);
            phiFree(ptrNN->yActFunc, ptrNN->K, ptrNN->trainingNumber);

            iteration = 0;

            phiInitializeNeuralNetwork(ptrInOut, ptrNN, ptrNN->I, ptrNN->H, ptrNN->K);
            printf("\n\nReinitialize!!!\n\n");
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // memory free functions

    // full terms
    phiFree(coeffUpdate, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)), 1);
    phiFree(coeffIntUpdate, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)), ptrNN->trainingNumber);

    phiFree(InvJacobianFullTerm, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
            (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    phiFree(JacobianFullTerm, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
            (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    phiFree(muEyeMatrices, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
            (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    phiFree(eyeMatrices, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
            (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    phiFree(JacobianMultiplicationResult, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
            (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));

    phiFree(JacobianTermTrans, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)),
            ptrNN->trainingNumber);
}

void phiActivationCalculation(phiInOutParametersPtr ptrInOut,
                              phiNNLibParameterPtr ptrNN, int index)
{
    // creating a loop to search the whole hidden node
    for (int h = 0; h < ptrNN->H + 1; h++)
    {
        // assign 1 if the hiddes node equals to bias term
        if (h == ptrNN->H)
        {
            ptrNN->zActFunc[h][index] = 1;
        }
        else
        {
            // calculatin input times coefficients
            double internalSum = 0.0;

            for (int i = 0; i < ptrNN->I; i++)
            {
                // the whole inputs are summed with the terms of W matrices * inputMatrices
                internalSum = internalSum + ptrNN->phiWmatrices[h][i] * ptrInOut->inputDataArray[i][index];
            }
            // sigmoid activation function is used in here!
            ptrNN->zActFunc[h][index] = 1.0 / (1.0 + exp(-internalSum));
            /*
            if (internalSum > 20)
            {

                ptrNN->zActFunc[h][index] = 1.0;
            }
            else
            {
                if (internalSum < -20)
                {

                    ptrNN->zActFunc[h][index] = 0.0;
                }
                else
                {
                    ptrNN->zActFunc[h][index] = 1.0 / (1.0 + exp(-internalSum));
                }
            }*/
        }
    }
}

void phiOutputFunctionCalculation(phiInOutParametersPtr ptrInOut,
                                  phiNNLibParameterPtr ptrNN, int index)
{
    // search for the whole outputs
    for (int k = 0; k < ptrNN->K; k++)
    {
        double internalSum = 0.0;

        for (int h = 0; h < ptrNN->H + 1; h++)
        {
            // multiplying the whole terms with the hidden output values
            // linear neuron is used in here!
            internalSum = internalSum + ptrNN->phiVmatrices[k][h] * ptrNN->zActFunc[h][index];
        }
        ptrNN->yActFunc[k][index] = internalSum;
        /*
        // assigned internalSum to the output function
        if (internalSum > 15)
        {
            ptrNN->yActFunc[k][index] = 15;
        }
        else
        {
            if (internalSum < -15)
            {
                ptrNN->yActFunc[k][index] = -15;
            }
            else
            {
                ptrNN->yActFunc[k][index] = internalSum;
            }
        }*/
    }
}

void phiLmAlgorithm(phiInOutParametersPtr ptrInOut,
                    phiNNLibParameterPtr ptrNN, int index)
{
    int jacobianCounter = 0;

    // calculation of increment values in V matrices
    for (int k = 0; k < ptrNN->K; k++)
    {
        for (int h = 0; h < ptrNN->H + 1; h++)
        {
            // Jacobian terms is assigned to dV value since it is updated in the next session!
            ptrNN->JacobianTerm[index][jacobianCounter] = ptrNN->zActFunc[h][index] * (-1.0);

            jacobianCounter++;
        }
    }

    for (int k = 0; k < ptrNN->K; k++)
    {
        for (int h = 0; h < ptrNN->H; h++)
        {
            for (int in = 0; in < ptrNN->I; in++)
            {

                ptrNN->JacobianTerm[index][jacobianCounter] = ptrNN->phiVmatrices[k][h] *
                                                              ptrNN->zActFunc[h][index] *
                                                              (1.0 - ptrNN->zActFunc[h][index]) *
                                                              ptrInOut->inputDataArray[in][index] *
                                                              (-1.0);

                jacobianCounter++;
            }
        }
    }
}

void costFunctionCalculation(phiInOutParametersPtr ptrInOut,
                             phiNNLibParameterPtr ptrNN, int index)
{

    // calculate the total error value on training/output dataset
    for (int k = 0; k < ptrNN->K; k++)
    {
        // 1/2*(y_tr - y)^2
        ptrNN->errorNow[index][k] = fabs(1.0 / 2 *
                                         (ptrInOut->outputDataArray[k][index] - ptrNN->yActFunc[k][index]) *
                                         (ptrInOut->outputDataArray[k][index] - ptrNN->yActFunc[k][index]));
    }
}

void costFunctionCalculationJacobian(phiInOutParametersPtr ptrInOut,
                                     phiNNLibParameterPtr ptrNN, int index)
{

    // calculate the total error value on training/output dataset
    for (int k = 0; k < ptrNN->K; k++)
    {
        // 1/2*(y_tr - y)^2
        ptrNN->errorNowJac[index][k] = (ptrInOut->outputDataArray[k][index] - ptrNN->yActFunc[k][index]);
    }
}

// neural network structure
////////////////////////////////////////////////////////////////
void writeCoefficientMatrices(phiNNLibParameterPtr ptrNN)
{
    printf("\n\n");
    printf("W input to hidden NN coefficients are written!\n");
    printf("\n");

    for (int i = 0; i < ptrNN->H; i++)
    {
        for (int j = 0; j < ptrNN->I; j++)
        {
            printf("%lf ", ptrNN->phiWmatrices[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("V hidden to output NN coefficients are written!\n");
    printf("\n");

    for (int i = 0; i < ptrNN->K; i++)
    {
        for (int j = 0; j < ptrNN->H + 1; j++)
        {
            printf("%lf ", ptrNN->phiVmatrices[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

double **phiRandInitialization(double min, double max, int firstNodeNumber, int secondNodeNumber)
{
    double **pd = creatingEmptyMatrices(firstNodeNumber, secondNodeNumber);

    for (int i = 0; i < firstNodeNumber; i++)
    {
        for (int j = 0; j < secondNodeNumber; j++)
        {
            pd[i][j] = (int)phiRand(min, max);
            //pd[i][j] = phiRand(min, max);
        }
    }

    return pd;
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

void phiExitNNLib(phiNNLibParameterPtr ptrNN)
{
    phiFree(ptrNN->phiWmatrices, ptrNN->H, ptrNN->I);
    phiFree(ptrNN->phiVmatrices, ptrNN->K, ptrNN->H + 1);
    phiFree(ptrNN->errorNow, ptrNN->trainingNumber, 1);
    phiFree(ptrNN->errorPre, ptrNN->trainingNumber, 1);
    phiFree(ptrNN->errorNowJac, ptrNN->trainingNumber, 1);
    phiFree(ptrNN->zActFunc, ptrNN->H + 1, ptrNN->trainingNumber);
    phiFree(ptrNN->yActFunc, ptrNN->K, ptrNN->trainingNumber);
    phiFree(ptrNN->JacobianTerm, ptrNN->trainingNumber, (ptrNN->H * ptrNN->I + ptrNN->K * (ptrNN->H + 1)));
}