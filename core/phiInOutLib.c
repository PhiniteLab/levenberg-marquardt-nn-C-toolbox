#include "..\include\phiNNLibSettings.h"

void printPhi()
{
    printf("\n\nPhi Libraries are loaded to the project!\n");
    printf("Process is started...\n\n");
    Sleep(5);
}

void readFileFromText(phiInOutParametersPtr ptrInOut)
{

    ptrInOut->inputFile = fopen(ptrInOut->fileName, "r");

    if (ptrInOut->inputFile == NULL)
    {
        printf("Text file is not opened!\n");
        exit(EXIT_FAILURE);
    }

    double emptyReading1, emptyReading2, emptyReading3;
    double emptyReading;
    double fillReading;

    char endOfFileControl = 0;

    int varTextInfo = 0;
    int rowCounter = 0;

    while (endOfFileControl != EOF)
    {
        if (varTextInfo == 0)
        {
            varTextInfo = 1;
            fscanf(ptrInOut->inputFile, "%lf %lf %lf", &emptyReading1,
                   &emptyReading2,
                   &emptyReading3);

            printf("Row %d Input Column %d Output Column %d\n", (int)emptyReading1,
                   (int)emptyReading2,
                   (int)emptyReading3);

            ptrInOut->lengthNNDataset = (int)emptyReading1;
            ptrInOut->numberNNInputDataset = (int)emptyReading2;
            ptrInOut->numberNNOutputDataset = (int)emptyReading3;

            for (int j = 0; j < (ptrInOut->numberNNOutputDataset + ptrInOut->numberNNInputDataset - 3); j++)
            {
                fscanf(ptrInOut->inputFile, "%lf", &emptyReading);
            }

            fillInputAndOutputMatrices(ptrInOut);

            endOfFileControl = getc(ptrInOut->inputFile);
        }
        else
        {
            if (rowCounter < ptrInOut->lengthNNDataset)
            {
                for (int j = 0; j < ptrInOut->numberNNInputDataset; j++)
                {
                    fscanf(ptrInOut->inputFile, "%lf ", &fillReading);
                    ptrInOut->inputDataArray[j][rowCounter] = fillReading;
                }

                for (int j = 0; j < ptrInOut->numberNNOutputDataset; j++)
                {
                    fscanf(ptrInOut->inputFile, "%lf", &fillReading);
                    ptrInOut->outputDataArray[j][rowCounter] = fillReading;
                }
            }

            rowCounter++;
            endOfFileControl = getc(ptrInOut->inputFile);
        }
    }

    fclose(ptrInOut->inputFile);
}

void readFileFromCSV(phiInOutParametersPtr ptrInOut)
{

    ptrInOut->inputFile = fopen(ptrInOut->fileName, "r");

    if (ptrInOut->inputFile == NULL)
    {
        printf("CSV file is not opened!\n");
        exit(EXIT_FAILURE);
    }

    double emptyReading1, emptyReading2, emptyReading3;
    double emptyReading;
    double fillReading;

    char endOfFileControl = 0;

    int varTextInfo = 0;
    int rowCounter = 0;

    while (endOfFileControl != EOF)
    {
        if (varTextInfo == 0)
        {
            varTextInfo = 1;
            fscanf(ptrInOut->inputFile, "%lf,%lf,%lf,", &emptyReading1,
                   &emptyReading2,
                   &emptyReading3);

            printf("Row %d Input Column %d Output Column %d\n", (int)emptyReading1,
                   (int)emptyReading2,
                   (int)emptyReading3);

            ptrInOut->lengthNNDataset = (int)emptyReading1;
            ptrInOut->numberNNInputDataset = (int)emptyReading2;
            ptrInOut->numberNNOutputDataset = (int)emptyReading3;

            for (int j = 0; j < (ptrInOut->numberNNOutputDataset + ptrInOut->numberNNInputDataset - 3); j++)
            {
                fscanf(ptrInOut->inputFile, "%lf,", &emptyReading);
            }

            fillInputAndOutputMatrices(ptrInOut);

            endOfFileControl = getc(ptrInOut->inputFile);
        }
        else
        {
            if (rowCounter < ptrInOut->lengthNNDataset)
            {
                for (int j = 0; j < ptrInOut->numberNNInputDataset; j++)
                {
                    fscanf(ptrInOut->inputFile, "%lf,", &fillReading);
                    ptrInOut->inputDataArray[j][rowCounter] = fillReading;
                }

                for (int j = 0; j < ptrInOut->numberNNOutputDataset; j++)
                {
                    fscanf(ptrInOut->inputFile, "%lf,", &fillReading);
                    ptrInOut->outputDataArray[j][rowCounter] = fillReading;
                }
            }

            rowCounter++;
            endOfFileControl = getc(ptrInOut->inputFile);
        }
    }

    fclose(ptrInOut->inputFile);
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

double **creatingEmptyInputMatrices(phiInOutParametersPtr ptrInOut)
{
    double **pd = (double **)malloc(ptrInOut->numberNNInputDataset * sizeof(double *));

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ptrInOut->numberNNInputDataset; i++)
        pd[i] = (double *)malloc(ptrInOut->lengthNNDataset * sizeof(double));

    return pd;
}

double **creatingEmptyOutputMatrices(phiInOutParametersPtr ptrInOut)
{
    double **pd = (double **)malloc(ptrInOut->numberNNOutputDataset * sizeof(double *));

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ptrInOut->numberNNOutputDataset; i++)
        pd[i] = (double *)malloc(ptrInOut->lengthNNDataset * sizeof(double));

    return pd;
}

double **creatingEmptyMatrices(int rows, int cols)
{
    double **pd = (double **)malloc(rows * sizeof(double *));

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++)
        pd[i] = (double *)malloc(cols * sizeof(double));

    // initialize the matrices
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            pd[i][j] = 0.0;
        }
    }

    return pd;
}

int **creatingEmptyMatricesIntegralType(int rows, int cols)
{
    int **pd = (int **)malloc(rows * sizeof(int *));

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++)
        pd[i] = (int *)malloc(cols * sizeof(int));

    // initialize the matrices
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            pd[i][j] = 0;
        }
    }

    return pd;
}

void fillInputAndOutputMatrices(phiInOutParametersPtr ptrInOut)
{
    ptrInOut->inputDataArray = creatingEmptyInputMatrices(ptrInOut);
    ptrInOut->outputDataArray = creatingEmptyOutputMatrices(ptrInOut);
}

void writeDataSetMatrices(phiInOutParametersPtr ptrInOut)
{
    printf("Printing dataset...\n");

    printf("In1 In2 In3 Out1\n");

    for (int i = 0; i < ptrInOut->lengthNNDataset; i++)
    {
        for (int j = 0; j < ptrInOut->numberNNInputDataset; j++)
        {
            printf("%lf ", ptrInOut->inputDataArray[j][i]);
        }

        for (int j = 0; j < ptrInOut->numberNNOutputDataset; j++)
        {
            printf("%lf ", ptrInOut->outputDataArray[j][i]);
        }

        printf("\n");
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void phiFree(double **pd, int row, int col)
{
    for (int i = 0; i < row; i++)
        free(pd[i]);

    free(pd);
}

void phiFreeIntegralType(int **pd, int row, int col)
{
    for (int i = 0; i < row; i++)
        free(pd[i]);

    free(pd);
}

void phiExitInOut(phiInOutParametersPtr ptrInOut)
{
    phiFree(ptrInOut->inputDataArray, ptrInOut->numberNNInputDataset, ptrInOut->lengthNNDataset);
    phiFree(ptrInOut->outputDataArray, ptrInOut->numberNNOutputDataset, ptrInOut->lengthNNDataset);
}

//////////////////////////////////////////////////////////////////////
// error Handler

void phiErrorHandler(int errorType)
{
    switch (errorType)
    {
    case FILE_OPEN_ERROR:
        printf("System Dynamic Parameter files cannot be created!\n");
        break;
    case INCONSISTENT_ROW_COLUMN:
        printf("The rows and columns are not consistent!\n");
        break;
    case ALLOCATION_ERROR:
        printf("Memory allocation cannot be done!\n");
        break;
    case SAMPLING_RATE_ERROR:
        printf("Sampling period cannot be assigned to either negative or zero value!\n");
        break;

    default:
        break;
    }
}

// error Handler
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// demos

void readingDemo1()
{
    phiInOutParameters pInOut;

    pInOut.fileName = "nnInputOutputFile.csv";

    readFileFromCSV(&pInOut);

    writeDataSetMatrices(&pInOut);

    phiExitInOut(&pInOut);
}

// demos
//////////////////////////////////////////////////////////////////////