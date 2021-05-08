#include "..\..\include\phiNNLibSettings.h"

int main()
{
    phiInOutParameters pInOut;

    pInOut.fileName = "nnInputOutputFile.txt";

    readFileFromText(&pInOut);

    writeDataSetMatrices(&pInOut);

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////
    // NN lib is started!

    printf("\nPress enter to start phiLM!\n");
    getchar();

    // randomize the whole process!!
    srand((unsigned int)time(NULL));

    phiNNLibParameters pNN;

    phiInitializeNeuralNetwork(&pInOut, &pNN, 3, 4, 1);

    phiTrainingParametersSettings(&pNN, 1e-9, 100000, 1e200, 1e-12);

    phiTrainingNeuralNetworkWithLM(&pInOut, &pNN);

    printf("%lf ", pNN.errorNowValue);

    writeCoefficientMatrices(&pNN);

    printf("Press enter to exit!\n");
    getchar();

    // NN lib is started!
    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////
    // exit functions

    phiExitInOut(&pInOut);

    phiExitNNLib(&pNN);

    // exit functions
    ////////////////////////////////////////////////////////

    return 0;
}
/*
int main()
{
    double **p1 = creatingEmptyMatrices(3, 2);
    double **p2 = creatingEmptyMatrices(2, 3);

    p1[0][0] = 1;
    p1[0][1] = 2;

    p1[1][0] = 4;
    p1[1][1] = 5;

    p1[2][0] = 9;
    p1[2][1] = 13;

    phiTranspose(p1, 3, 2, p2);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("%lf ", p2[i][j]);
        }
        printf("\n");
    }

    phiFree(p1, 3, 2);
    phiFree(p2, 2, 3);

    return 0;
}
*/