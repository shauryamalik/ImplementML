using System;
namespace ImplementML
{
    class KernelLogisticProgram
    {
        static Random rnd = new Random(0);
        public static void MainFunc()
        {
            Console.WriteLine("\nBegin kernel logistic regression demo \n");
            int numFeatures = 2;

            Console.WriteLine("Goal is binary classification (0/1) \n");
            Console.WriteLine("Setting up 21 training and 4 test data items \n");

            // donut-like data (not linearly separable)
            double[][] trainData = new double[21][];
            trainData[0] = new double[] { 2.0, 3.0, 0 };
            trainData[1] = new double[] { 1.0, 5.0, 0 };
            trainData[2] = new double[] { 2.0, 7.0, 0 };
            trainData[3] = new double[] { 3.0, 2.0, 0 };
            trainData[4] = new double[] { 3.0, 8.0, 0 };
            trainData[5] = new double[] { 4.0, 2.0, 0 };
            trainData[6] = new double[] { 4.0, 8.0, 0 };
            trainData[7] = new double[] { 5.0, 2.0, 0 };
            trainData[8] = new double[] { 5.0, 8.0, 0 };
            trainData[9] = new double[] { 6.0, 3.0, 0 };
            trainData[10] = new double[] { 7.0, 5.0, 0 };
            trainData[11] = new double[] { 6.0, 7.0, 0 };
            trainData[12] = new double[] { 3.0, 4.0, 1 };
            trainData[13] = new double[] { 3.0, 5.0, 1 };
            trainData[14] = new double[] { 3.0, 6.0, 1 };
            trainData[15] = new double[] { 4.0, 4.0, 1 };
            trainData[16] = new double[] { 4.0, 5.0, 1 };
            trainData[17] = new double[] { 4.0, 6.0, 1 };
            trainData[18] = new double[] { 5.0, 4.0, 1 };
            trainData[19] = new double[] { 5.0, 5.0, 1 };
            trainData[20] = new double[] { 5.0, 6.0, 1 };

            // test data
            double[][] testData = new double[4][];
            testData[0] = new double[] { 1.5, 4.5, 0 };
            testData[1] = new double[] { 7.0, 6.5, 0 };
            testData[2] = new double[] { 3.5, 4.5, +1 };
            testData[3] = new double[] { 5.5, 5.5, +1 };

            Console.WriteLine("training [0] =  (2.0, 3.0, 0)");
            Console.WriteLine("training [1] =  (1.0, 5.0, 0)");
            Console.WriteLine(" . . . ");
            Console.WriteLine("training [20] = (5.0, 6.0, 1)");

            //// a super-tiny dataset 
            //double[][] trainData = new double[4][];
            //trainData[0] = new double[] { 2.0, 4.0, 0 };
            //trainData[1] = new double[] { 4.0, 1.0, 1 };
            //trainData[2] = new double[] { 5.0, 3.0, 0 };
            //trainData[3] = new double[] { 6.0, 7.0, 1 };
            //double[][] testData = new double[1][];
            //testData[0] = new double[] { 3.0, 5.0, 0 };

            int numTrain = trainData.Length;
            int numTest = testData.Length;

            double[] alphas = new double[numTrain + 1];  // one alpha weight for each train item, plus bias at end
            for (int i = 0; i < alphas.Length; ++i)
                alphas[i] = 0.0;

            // pre-compute all kernels - only viable if numTrain is not huge
            double[][] kernelMatrix = new double[numTrain][]; // item-item similarity
            for (int i = 0; i < kernelMatrix.Length; ++i)
                kernelMatrix[i] = new double[numTrain];

            double sigma = 1.0;
            for (int i = 0; i < numTrain; ++i)  // pre-compute all Kernel
            {
                for (int j = 0; j < numTrain; ++j)
                {
                    double k = Kernel(trainData[i], trainData[j], sigma);
                    kernelMatrix[i][j] = kernelMatrix[j][i] = k;
                }
            }

            //// display kernel matrix
            //Console.WriteLine("\n-------------\n");
            //for (int i = 0; i < numTrain; ++i)  // pre-compute all Kernel
            //{
            //  for (int j = 0; j < numTrain; ++j)
            //  {
            //    Console.Write(kernelMatrix[i][j].ToString("F8") + " ");
            //   }
            //  Console.WriteLine("");
            //}
            //Console.WriteLine("\n-------------\n");

            // train. aj = aj + eta(t - y) * K(i,j)
            double eta = 0.001;  // aka learning-rate
            int iter = 0;
            int maxIter = 1000;
            int[] indices = new int[numTrain];
            for (int i = 0; i < indices.Length; ++i)
                indices[i] = i;

            Console.WriteLine("\nStarting training");
            Console.WriteLine("Using RBF kernel() with sigma = " + sigma.ToString("F1"));
            Console.WriteLine("Using SGD with eta = " + eta + " and maxIter = " + maxIter);
            while (iter < maxIter)
            {
                Shuffle(indices);  // visit train data in random order
                for (int idx = 0; idx < indices.Length; ++idx)  // each 'from' train data
                {
                    int i = indices[idx];  // current train data index

                    double sum = 0.0;  // sum of alpha-i * kernel-i
                    for (int j = 0; j < alphas.Length - 1; ++j)  // not the bias
                        sum += alphas[j] * kernelMatrix[i][j];
                    sum += alphas[alphas.Length - 1];  // add bias (last alpha) -- 'input' is dummy 1.0

                    double y = 1.0 / (1.0 + Math.Exp(-sum));
                    double t = trainData[i][numFeatures];  // last col holds target value

                    // update each alpha
                    for (int j = 0; j < alphas.Length - 1; ++j)
                        alphas[j] = alphas[j] + (eta * (t - y) * kernelMatrix[i][j]);
                    // update the bias
                    alphas[alphas.Length - 1] = alphas[alphas.Length - 1] +
                      (eta * (t - y)) * 1;  // dummy input
                }
                ++iter;
            } // while (train)

            Console.WriteLine("\nTraining complete");
            Console.WriteLine("\nTrained model alpha values: \n");
            for (int i = 0; i < 3; ++i)
                Console.WriteLine(" [" + i + "]  " + alphas[i].ToString("F4"));
            Console.WriteLine(" . . .");
            for (int i = alphas.Length - 3; i < alphas.Length - 1; ++i)
                Console.WriteLine(" [" + i + "]  " + alphas[i].ToString("F4"));
            Console.WriteLine(" [" + (alphas.Length - 1) + "] (bias) " +
              alphas[alphas.Length - 1].ToString("F4"));
            Console.WriteLine("");

            Console.WriteLine("Evalating model accuracy on train data");
            double accTrain = Accuracy(trainData, trainData, alphas, sigma, false);
            Console.WriteLine("accuracy = " + accTrain.ToString("F4") + "\n");
            Console.WriteLine("Evalating model accuracy on test data");
            double accTest = Accuracy(testData, trainData, alphas, sigma, true);  // verbose
                                                                                  //Console.WriteLine("accuracy = " + accTest.ToString("F4"));

            Console.WriteLine("\nEnd kernel logistic regression demo ");
            Console.ReadLine();
        } // Main

        static double Accuracy(double[][] data, double[][] trainData,
          double[] alphas, double sigma, bool verbose)
        {
            int numCorrect = 0;
            int numWrong = 0;
            int numTrain = trainData.Length;
            int numFeatures = trainData[0].Length - 1;

            for (int i = 0; i < data.Length; ++i)  // i index into data to predict
            {
                // compare currr against all trainData
                double sum = 0.0;
                for (int j = 0; j < alphas.Length - 1; ++j)
                {
                    double k = Kernel(data[i], trainData[j], sigma);
                    sum += alphas[j] * k;  // (cannot pre-compute)
                }
                sum += alphas[alphas.Length - 1] * 1;  // add the bias

                double y = 1.0 / (1.0 + Math.Exp(-sum));
                double t = data[i][numFeatures];
                double pred = 0;
                if (y > 0.5) pred = 1;

                if (verbose)
                {
                    Console.Write(" input = (");
                    for (int j = 0; j < data[i].Length - 2; ++j)
                    {
                        Console.Write(data[i][j].ToString("F1") + ", ");
                    }
                    Console.Write(data[i][data[i].Length - 2].ToString("F1"));
                    Console.Write(")");

                    Console.Write(" actual = " + t + "  calc y = " +
                      y.ToString("F4") +
                      "  pred = " + pred.ToString("F0"));
                    if (y <= 0.5 && t == 0.0 || y > 0.5 && t == 1.0)
                    {
                        ++numCorrect;
                        Console.WriteLine("  correct");
                    }
                    else
                    {
                        ++numWrong;
                        Console.WriteLine("  WRONG");
                    }
                }
                else // not verbose
                {
                    if (y <= 0.5 && t == 0.0 || y > 0.5 && t == 1.0)
                        ++numCorrect;
                    else
                        ++numWrong;
                }
            } // each test data

            if (verbose)
                Console.WriteLine("numCorrect = " + numCorrect +
                  "   numWrong = " + numWrong);
            return (1.0 * numCorrect) / (numCorrect + numWrong);
        }

        static double Kernel(double[] v1, double[] v2, double sigma)
        {
            // RBF kernel. v1 & v2 have class label in last cell
            double num = 0.0;
            for (int i = 0; i < v1.Length - 1; ++i)  // not last cell
                num += (v1[i] - v2[i]) * (v1[i] - v2[i]);
            double denom = 2.0 * sigma * sigma;
            double z = num / denom;
            return Math.Exp(-z);
        }

        static void Shuffle(int[] indices)
        {
            // assumes class-scope Random object rnd
            for (int i = 0; i < indices.Length; ++i)
            {
                int ri = rnd.Next(i, indices.Length);
                int tmp = indices[i];
                indices[i] = indices[ri];
                indices[ri] = tmp;
            }
        }

    } // Program
} // ns

// a bit more interesting data:

//double[][] trainData = new double[100][];
//trainData[0] = new double[] { -7.5, 5.5, 0 };
//trainData[1] = new double[] { 3.0, 9.5, 0 };
//trainData[2] = new double[] { 2.5, -1.0, 1 };
//trainData[3] = new double[] { -5.0, 9.0, 0 };
//trainData[4] = new double[] { 0.0, 2.5, 1 };
//trainData[5] = new double[] { -9.5, 1.5, 0 };
//trainData[6] = new double[] { 3.0, 1.0, 1 };
//trainData[7] = new double[] { 2.5, -1.5, 1 };
//trainData[8] = new double[] { 4.5, -9.0, 0 };
//trainData[9] = new double[] { 8.0, -3.0, 0 };
//trainData[10] = new double[] { 6.0, -7.0, 0 };
//trainData[11] = new double[] { 9.5, 3.0, 0 };
//trainData[12] = new double[] { 2.0, -1.5, 1 };
//trainData[13] = new double[] { -10.0, -1.5, 0 };
//trainData[14] = new double[] { 0.0, 10.0, 0 };
//trainData[15] = new double[] { -0.5, -9.5, 0 };
//trainData[16] = new double[] { -1.5, 9.5, 0 };
//trainData[17] = new double[] { -7.5, -4.5, 0 };
//trainData[18] = new double[] { -3.0, -9.0, 0 };
//trainData[19] = new double[] { 2.0, 1.5, 1 };
//trainData[20] = new double[] { -1.0, -1.5, 1 };
//trainData[21] = new double[] { 9.0, 4.5, 0 };
//trainData[22] = new double[] { -2.0, -1.5, 1 };
//trainData[23] = new double[] { -2.0, -0.5, 1 };
//trainData[24] = new double[] { -1.5, -1.0, 1 };
//trainData[25] = new double[] { 1.0, 2.5, 1 };
//trainData[26] = new double[] { 8.5, 5.5, 0 };
//trainData[27] = new double[] { -1.5, 1.0, 1 };
//trainData[28] = new double[] { -2.5, 1.0, 1 };
//trainData[29] = new double[] { -6.5, 6.0, 0 };
//trainData[30] = new double[] { -0.5, 3.0, 1 };
//trainData[31] = new double[] { 0.5, 2.0, 1 };
//trainData[32] = new double[] { -9.5, -3.0, 0 };
//trainData[33] = new double[] { -2.5, 0.0, 1 };
//trainData[34] = new double[] { 1.0, 2.5, 1 };
//trainData[35] = new double[] { -2.0, -0.5, 1 };
//trainData[36] = new double[] { -6.5, 6.0, 0 };
//trainData[37] = new double[] { 0.0, -3.0, 1 };
//trainData[38] = new double[] { 6.0, 7.0, 0 };
//trainData[39] = new double[] { 1.5, -10.0, 0 };
//trainData[40] = new double[] { 6.5, 6.0, 0 };
//trainData[41] = new double[] { -3.0, -1.5, 1 };
//trainData[42] = new double[] { 1.5, 9.0, 0 };
//trainData[43] = new double[] { 1.0, 2.0, 1 };
//trainData[44] = new double[] { -9.0, 3.5, 0 };
//trainData[45] = new double[] { 2.0, -2.5, 1 };
//trainData[46] = new double[] { -8.5, 2.0, 0 };
//trainData[47] = new double[] { 0.5, 9.5, 0 };
//trainData[48] = new double[] { 2.0, -2.0, 1 };
//trainData[49] = new double[] { -10.5, 0.0, 0 };
//trainData[50] = new double[] { -1.5, -2.5, 1 };
//trainData[51] = new double[] { 9.0, -2.5, 0 };
//trainData[52] = new double[] { -5.5, 8.5, 0 };
//trainData[53] = new double[] { -1.0, 2.0, 1 };
//trainData[54] = new double[] { -7.0, -6.0, 0 };
//trainData[55] = new double[] { -4.0, -9.5, 0 };
//trainData[56] = new double[] { -3.0, 10.0, 0 };
//trainData[57] = new double[] { 2.5, 0.0, 1 };
//trainData[58] = new double[] { -0.5, -3.5, 1 };
//trainData[59] = new double[] { -2.0, -2.5, 1 };
//trainData[60] = new double[] { -9.0, -3.0, 0 };
//trainData[61] = new double[] { -0.5, 2.5, 1 };
//trainData[62] = new double[] { -2.0, 3.0, 1 };
//trainData[63] = new double[] { -5.5, -8.0, 0 };
//trainData[64] = new double[] { 8.0, -4.5, 0 };
//trainData[65] = new double[] { 5.5, -9.0, 0 };
//trainData[66] = new double[] { 3.5, -9.5, 0 };
//trainData[67] = new double[] { -7.0, -8.0, 0 };
//trainData[68] = new double[] { 1.5, -2.5, 1 };
//trainData[69] = new double[] { -2.5, 1.0, 1 };
//trainData[70] = new double[] { 10.0, -0.5, 0 };
//trainData[71] = new double[] { 5.5, 7.5, 0 };
//trainData[72] = new double[] { -7.5, -6.0, 0 };
//trainData[73] = new double[] { 3.5, 0.0, 1 };
//trainData[74] = new double[] { -1.5, 2.5, 1 };
//trainData[75] = new double[] { 2.0, 1.5, 1 };
//trainData[76] = new double[] { -0.5, -2.5, 1 };
//trainData[77] = new double[] { -3.0, 0.5, 1 };
//trainData[78] = new double[] { -2.5, -0.5, 1 };
//trainData[79] = new double[] { -8.0, 4.0, 0 };
//trainData[80] = new double[] { 0.5, -3.5, 1 };
//trainData[81] = new double[] { 9.5, 0.0, 0 };
//trainData[82] = new double[] { 9.0, 3.5, 0 };
//trainData[83] = new double[] { 1.5, 1.0, 1 };
//trainData[84] = new double[] { 0.0, -9.5, 0 };
//trainData[85] = new double[] { 7.0, -6.5, 0 };
//trainData[86] = new double[] { 1.5, -1.5, 1 };
//trainData[87] = new double[] { -1.5, -9.5, 0 };
//trainData[88] = new double[] { 1.5, 2.0, 1 };
//trainData[89] = new double[] { 1.0, -2.0, 1 };
//trainData[90] = new double[] { 1.5, -0.5, 1 };
//trainData[91] = new double[] { 2.0, 2.0, 1 };
//trainData[92] = new double[] { 4.0, 9.5, 0 };
//trainData[93] = new double[] { 2.5, 0.0, 1 };
//trainData[94] = new double[] { -3.0, 1.5, 1 };
//trainData[95] = new double[] { 0.0, -2.0, 1 };
//trainData[96] = new double[] { 9.5, 1.0, 0 };
//trainData[97] = new double[] { -2.0, 1.5, 1 };
//trainData[98] = new double[] { 2.5, 1.0, 1 };
//trainData[99] = new double[] { 9.5, -4.5, 0 };

//double[][] testData = new double[20][];
//testData[0] = new double[] { 2.5, 8.5, 0 };
//testData[1] = new double[] { -9.5, -0.5, 0 };
//testData[2] = new double[] { 0.0, -2.0, 1 };
//testData[3] = new double[] { 1.5, -2.0, 1 };
//testData[4] = new double[] { 0.5, 2.0, 1 };
//testData[5] = new double[] { 7.5, 5.5, 0 };
//testData[6] = new double[] { 2.5, 1.0, 1 };
////testData[7] = new double[] { 4.0, 0.0, 1 };
//testData[7] = new double[] { 4.5, 0.0, 1 };  // to get an error!
//testData[8] = new double[] { -0.5, 2.5, 1 };
//testData[9] = new double[] { 1.0, -2.5, 1 };
//testData[10] = new double[] { -2.0, -9.5, 0 };
//testData[11] = new double[] { -7.0, 6.0, 0 };
//testData[12] = new double[] { 2.5, -9.5, 0 };
//testData[13] = new double[] { 8.5, -5.5, 0 };
//testData[14] = new double[] { 10.5, 0.0, 0 };
//testData[15] = new double[] { -2.5, -2.0, 1 };
//testData[16] = new double[] { -8.0, -6.0, 0 };
//testData[17] = new double[] { -3.0, -0.5, 1 };
//testData[18] = new double[] { -2.0, 1.0, 1 };
//testData[19] = new double[] { -2.0, 8.5, 0 };
