#include "..\..\include\phiNNLibSettings.h"

int main()
{
    printf("Hello Phinite\n");

    //readingDemo1();

    phiInOutParameters pInOut;

    //pInOut.fileName = "nnInputOutputFile.csv";

    //readFileFromCSV(&pInOut);

    pInOut.fileName = "nnInputOutputFile.txt";

    readFileFromText(&pInOut);

    writeDataSetMatrices(&pInOut);

    phiExitInOut(&pInOut);

    return 0;
}