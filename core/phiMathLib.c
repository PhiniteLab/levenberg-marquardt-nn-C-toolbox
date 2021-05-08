#include "..\include\phiNNLibSettings.h"

////////////////////////////////////////////////////////////////////
// function definition

double **phiVectorMatrixMultiplication(double **firstTerm, double **SecondTerm,
                                       int row1, int col1,
                                       int row2, int col2)
{
    double **pd = creatingEmptyMatrices(row1, col2);
    double sum = 0;

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < row1; i++) //row of first matrix
    {
        for (int j = 0; j < col2; j++) //column of second matrix
        {
            sum = 0;
            for (int k = 0; k < col1; k++)
            {
                sum = sum + firstTerm[i][k] * SecondTerm[k][j];
            }
            pd[i][j] = sum;
        }
    }
    return pd;
}

double **phiSkalarMatrixMultiplication(double skalarTerm, double **SecondTerm,
                                       int row, int col)
{
    double **pd = creatingEmptyMatrices(row, col);

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            pd[i][j] = skalarTerm * SecondTerm[i][j];
        }
    }

    return pd;
}

double **phiMatrixSummation(double **firstTerm, double **SecondTerm, int row, int col)
{
    double **pd = creatingEmptyMatrices(row, col);

    if (pd == NULL)
    {
        phiErrorHandler(ALLOCATION_ERROR);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            pd[i][j] = firstTerm[i][j] + SecondTerm[i][j];
        }
    }

    return pd;
}

void phiMatrixAssignment(double **assignedTerm, double **SecondTerm, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            assignedTerm[i][j] = SecondTerm[i][j];
        }
    }
}

double phiRand(double min, double max)
{
    double scale = rand() / (double)RAND_MAX; /* [0, 1.0] */
    return min + scale * (max - min);         /* [min, max] */
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

double phiSumMatrices(double **pd, int row, int col)
{
    double sumValue = 0.0;

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            sumValue = sumValue + pd[i][j];
        }
    }

    return sumValue;
}

double phiMeanMatrices(double **pd, int row, int col)
{
    double sumValue = 0.0;
    int counter = 0;

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            counter++;
            sumValue = sumValue + pd[i][j];
        }
    }

    return sumValue / (double)counter;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// LM algorithm requirements

double phiDeterminationCalculation(double **phiMatrices,
                                   int row,
                                   int col)
{
    double phiDet = 1.0; // Initialize result
    double ratio = 0.0;

    /* Conversion of matrix to upper triangular */
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < row; j++)
        {
            if (j > i)
            {
                ratio = phiMatrices[j][i] / phiMatrices[i][i];
                for (int k = 0; k < row; k++)
                {
                    phiMatrices[j][k] -= ratio * phiMatrices[i][k];
                }
            }
        }
    }

    for (int i = 0; i < row; i++)
    {
        phiDet *= phiMatrices[i][i];
    }
    //printf("The determinant of matrix is: %e\n\n", phiDet);

    return phiDet;
}

void phiTranspose(double **phiMatrices, int row, int col, double **phiTransMatrices)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            phiTransMatrices[j][i] = phiMatrices[i][j];
        }
    }
}

double **phiEyeCreationMatrices(int row)
{
    double **pd = creatingEmptyMatrices(row, row);

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < row; j++)
        {
            if (i == j)
            {
                pd[i][j] = 1;
            }
            else
            {
                pd[i][j] = 0;
            }
        }
    }

    return pd;
}

// LM algorithm requirements
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// inverse matrices operation

double **phiInverseMatrices(double **phiMatrices, int row, int col)
{
    /*
    double determinantValue = phiDeterminationCalculation(phiMatrices, row, col);

    if (determinantValue == 0)
    {
        printf("Determinant is zero, so there is no inverse matrices!");
    }*/

    //////////////////////////////////////////////////
    // dynamic memory allocation

    // inverse Matrice
    double **inv = creatingEmptyMatrices(row, 2 * col);
    double **realInv = creatingEmptyMatrices(row, col);

    phiMatrixAssignment(inv, phiMatrices, row, col);

    realInv = InverseOfMatrix(inv, row);

    // control for realInv matrices
    /*
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < row; j++)
        {
            if (realInv[i][j] > 1e50)
            {
                realInv[i][j] = 1e50;
            }
            else
            {
                if (realInv[i][j] < -1e50)
                {
                    realInv[i][j] = -1e50;
                }
            }
        }
    }*/

    phiFree(inv, row, 2 * col);

    return realInv;
}

// Function to Print matrix.
void PrintMatrix(double **ar, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%e ", ar[i][j]);
        }
        printf("\n");
    }
    return;
}

// Function to Print inverse matrix
void PrintInverse(double **ar, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = n; j < m; j++)
        {
            printf("%e ", ar[i][j]);
        }
        printf("\n");
    }
    return;
}

// Function to perform the inverse operation on the matrix.
double **InverseOfMatrix(double **matrix, int order)
{
    // Matrix Declaration.

    double temp;

    // PrintMatrix function to print the element
    // of the matrix.
    //printf("=== Matrix ===\n");
    //PrintMatrix(matrix, order, order);

    // Create the augmented matrix
    // Add the identity matrix
    // of order at the end of original matrix.
    for (int i = 0; i < order; i++)
    {

        for (int j = 0; j < 2 * order; j++)
        {

            // Add '1' at the diagonal places of
            // the matrix to create a identity matirx
            if (j == (i + order))
            {
                matrix[i][j] = 1;
            }
        }
    }

    // Interchange the row of matrix,
    // interchanging of row will start from the last row
    for (int i = order - 1; i > 0; i--)
    {

        // Swapping each and every element of the two rows
        // if (matrix[i - 1][0] < matrix[i][0])
        // for (int j = 0; j < 2 * order; j++) {
        //
        //	 // Swapping of the row, if above
        //	 // condition satisfied.
        // temp = matrix[i][j];
        // matrix[i][j] = matrix[i - 1][j];
        // matrix[i - 1][j] = temp;
        // }

        // Directly swapping the rows using pointers saves
        // time

        if (matrix[i - 1][0] < matrix[i][0])
        {
            double *temp = matrix[i];
            matrix[i] = matrix[i - 1];
            matrix[i - 1] = temp;
        }
    }

    // Print matrix after interchange operations.
    //printf("\n=== Augmented Matrix ===\n");
    //PrintMatrix(matrix, order, order * 2);

    // Replace a row by sum of itself and a
    // constant multiple of another row of the matrix
    for (int i = 0; i < order; i++)
    {

        for (int j = 0; j < order; j++)
        {

            if (j != i)
            {
                if (matrix[i][i] == 0)
                {
                    //matrix[i][i] = 1;
                    temp = 1e-100;
                }
                else
                {
                    temp = matrix[j][i] / matrix[i][i];
                }

                for (int k = 0; k < 2 * order; k++)
                {

                    matrix[j][k] -= matrix[i][k] * temp;
                }
            }
        }
    }

    // Multiply each row by a nonzero integer.
    // Divide row element by the diagonal element
    for (int i = 0; i < order; i++)
    {

        temp = matrix[i][i];
        for (int j = 0; j < 2 * order; j++)
        {

            matrix[i][j] = matrix[i][j] / temp;
        }
    }

    double **pdInv = creatingEmptyMatrices(order, order);

    for (int i = 0; i < order; i++)
    {
        for (int j = order; j < 2 * order; j++)
        {
            pdInv[i][j - order] = matrix[i][j];
        }
    }

    //PrintInverse(matrix, order, 2 * order);

    return pdInv;
}

// inverse matrices operation
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

// function definition
////////////////////////////////////////////////////////////////////
