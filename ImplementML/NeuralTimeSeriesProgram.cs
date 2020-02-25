using System;
namespace ImplementML
{
  class NeuralTimeSeriesProgram
  {
    public static void MainFunc()//string[] args)
    {
      Console.WriteLine("\nBegin neural network times series demo");
      Console.WriteLine("Goal is to predict airline passengers over time ");
      Console.WriteLine("Data from January 1949 to December 1960 \n");

      double[][] trainData = GetAirlineData();
      trainData = Normalize(trainData);
      Console.WriteLine("Normalized training data:");
      ShowMatrix(trainData, 5, 2, true);  // first 5 rows, 2 decimals, show indices

      int numInput = 4; // number predictors
      int numHidden = 12;
      int numOutput = 1; // regression

      Console.WriteLine("Creating a " + numInput + "-" + numHidden +
        "-" + numOutput + " neural network");
      NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

      int maxEpochs = 10000;
      double learnRate = 0.01;
      Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));

      Console.WriteLine("\nStarting training");
      double[] weights = nn.Train(trainData, maxEpochs, learnRate);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      ShowVector(weights, 2, 10, true);

      double trainAcc = nn.Accuracy(trainData, 0.30);  // within 30
      Console.WriteLine("\nModel accuracy (+/- 30) on training data = " +
        trainAcc.ToString("F4"));

      double[] predictors = new double[] { 5.08, 4.61, 3.90, 4.32 };
      double[] forecast = nn.ComputeOutputs(predictors);  // 4.33362252510741
      Console.WriteLine("\nPredicted passengers for January 1961 (t=145): ");
      Console.WriteLine((forecast[0] * 100).ToString("F0"));

      //double[] predictors = new double[] { 4.61, 3.90, 4.32, 4.33362252510741 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 4.33933519590564
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 3.90, 4.32, 4.33362252510741, 4.33933519590564 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 4.69036205766231
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 4.32, 4.33362252510741, 4.33933519590564, 4.69036205766231 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 4.83360378041341
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 4.33362252510741, 4.33933519590564, 4.69036205766231, 4.83360378041341 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 5.50703476366623
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 4.33933519590564, 4.69036205766231, 4.83360378041341, 5.50703476366623 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 6.39605763609294
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 4.69036205766231, 4.83360378041341, 5.50703476366623, 6.39605763609294 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 6.06664881070054
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 4.83360378041341, 5.50703476366623, 6.39605763609294, 6.06664881070054 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 4.95781531728514
      //Console.WriteLine(forecast[0]);

      //double[] predictors = new double[] { 5.50703476366623, 6.39605763609294, 6.06664881070054, 4.95781531728514 };
      //double[] forecast = nn.ComputeOutputs(predictors);  // 4.45837470369601
      //Console.WriteLine(forecast[0]);


      Console.WriteLine("\nEnd time series demo\n");
      Console.ReadLine();
    } // Main

    static double[][] Normalize(double[][] data)
    {
      // divide all by 100.0
      int rows = data.Length;
      int cols = data[0].Length;
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];

      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = data[i][j] / 100.0;
      return result;
    }

    //static double[][] MakeSineData()
    //{
    //  double[][] sineData = new double[17][];
    //  sineData[0] = new double[] { 0.0000, 0.841470985, 0.909297427, 0.141120008 };
    //  sineData[1] = new double[] { 0.841470985, 0.909297427, 0.141120008, -0.756802495 };
    //  sineData[2] = new double[] { 0.909297427, 0.141120008, -0.756802495, -0.958924275 };
    //  sineData[3] = new double[] { 0.141120008, -0.756802495, -0.958924275, -0.279415498 };
    //  sineData[4] = new double[] { -0.756802495, -0.958924275, -0.279415498, 0.656986599 };
    //  sineData[5] = new double[] { -0.958924275, -0.279415498, 0.656986599, 0.989358247 };
    //  sineData[6] = new double[] { -0.279415498, 0.656986599, 0.989358247, 0.412118485 };
    //  sineData[7] = new double[] { 0.656986599, 0.989358247, 0.412118485, -0.544021111 };
    //  sineData[8] = new double[] { 0.989358247, 0.412118485, -0.544021111, -0.999990207 };
    //  sineData[9] = new double[] { 0.412118485, -0.544021111, -0.999990207, -0.536572918 };
    //  sineData[10] = new double[] { -0.544021111, -0.999990207, -0.536572918, 0.420167037 };
    //  sineData[11] = new double[] { -0.999990207, -0.536572918, 0.420167037, 0.990607356 };
    //  sineData[12] = new double[] { -0.536572918, 0.420167037, 0.990607356, 0.65028784 };
    //  sineData[13] = new double[] { 0.420167037, 0.990607356, 0.65028784, -0.287903317 };
    //  sineData[14] = new double[] { 0.990607356, 0.65028784, -0.287903317, -0.961397492 };
    //  sineData[15] = new double[] { 0.65028784, -0.287903317, -0.961397492, -0.750987247 };
    //  sineData[16] = new double[] { -0.287903317, -0.961397492, -0.750987247, 0.14987721 };
    //  return sineData;
    //} // MakeSineData

    static double[][] GetAirlineData()
    {
      double[][] airData = new double[140][];
      airData[0] = new double[] { 112, 118, 132, 129, 121 };
      airData[1] = new double[] { 118, 132, 129, 121, 135 };
      airData[2] = new double[] { 132, 129, 121, 135, 148 };
      airData[3] = new double[] { 129, 121, 135, 148, 148 };
      airData[4] = new double[] { 121, 135, 148, 148, 136 };
      airData[5] = new double[] { 135, 148, 148, 136, 119 };
      airData[6] = new double[] { 148, 148, 136, 119, 104 };
      airData[7] = new double[] { 148, 136, 119, 104, 118 };
      airData[8] = new double[] { 136, 119, 104, 118, 115 };
      airData[9] = new double[] { 119, 104, 118, 115, 126 };
      airData[10] = new double[] { 104, 118, 115, 126, 141 };
      airData[11] = new double[] { 118, 115, 126, 141, 135 };
      airData[12] = new double[] { 115, 126, 141, 135, 125 };
      airData[13] = new double[] { 126, 141, 135, 125, 149 };
      airData[14] = new double[] { 141, 135, 125, 149, 170 };
      airData[15] = new double[] { 135, 125, 149, 170, 170 };
      airData[16] = new double[] { 125, 149, 170, 170, 158 };
      airData[17] = new double[] { 149, 170, 170, 158, 133 };
      airData[18] = new double[] { 170, 170, 158, 133, 114 };
      airData[19] = new double[] { 170, 158, 133, 114, 140 };
      airData[20] = new double[] { 158, 133, 114, 140, 145 };
      airData[21] = new double[] { 133, 114, 140, 145, 150 };
      airData[22] = new double[] { 114, 140, 145, 150, 178 };
      airData[23] = new double[] { 140, 145, 150, 178, 163 };
      airData[24] = new double[] { 145, 150, 178, 163, 172 };
      airData[25] = new double[] { 150, 178, 163, 172, 178 };
      airData[26] = new double[] { 178, 163, 172, 178, 199 };
      airData[27] = new double[] { 163, 172, 178, 199, 199 };
      airData[28] = new double[] { 172, 178, 199, 199, 184 };
      airData[29] = new double[] { 178, 199, 199, 184, 162 };
      airData[30] = new double[] { 199, 199, 184, 162, 146 };
      airData[31] = new double[] { 199, 184, 162, 146, 166 };
      airData[32] = new double[] { 184, 162, 146, 166, 171 };
      airData[33] = new double[] { 162, 146, 166, 171, 180 };
      airData[34] = new double[] { 146, 166, 171, 180, 193 };
      airData[35] = new double[] { 166, 171, 180, 193, 181 };
      airData[36] = new double[] { 171, 180, 193, 181, 183 };
      airData[37] = new double[] { 180, 193, 181, 183, 218 };
      airData[38] = new double[] { 193, 181, 183, 218, 230 };
      airData[39] = new double[] { 181, 183, 218, 230, 242 };
      airData[40] = new double[] { 183, 218, 230, 242, 209 };
      airData[41] = new double[] { 218, 230, 242, 209, 191 };
      airData[42] = new double[] { 230, 242, 209, 191, 172 };
      airData[43] = new double[] { 242, 209, 191, 172, 194 };
      airData[44] = new double[] { 209, 191, 172, 194, 196 };
      airData[45] = new double[] { 191, 172, 194, 196, 196 };
      airData[46] = new double[] { 172, 194, 196, 196, 236 };
      airData[47] = new double[] { 194, 196, 196, 236, 235 };
      airData[48] = new double[] { 196, 196, 236, 235, 229 };
      airData[49] = new double[] { 196, 236, 235, 229, 243 };
      airData[50] = new double[] { 236, 235, 229, 243, 264 };
      airData[51] = new double[] { 235, 229, 243, 264, 272 };
      airData[52] = new double[] { 229, 243, 264, 272, 237 };
      airData[53] = new double[] { 243, 264, 272, 237, 211 };
      airData[54] = new double[] { 264, 272, 237, 211, 180 };
      airData[55] = new double[] { 272, 237, 211, 180, 201 };
      airData[56] = new double[] { 237, 211, 180, 201, 204 };
      airData[57] = new double[] { 211, 180, 201, 204, 188 };
      airData[58] = new double[] { 180, 201, 204, 188, 235 };
      airData[59] = new double[] { 201, 204, 188, 235, 227 };
      airData[60] = new double[] { 204, 188, 235, 227, 234 };
      airData[61] = new double[] { 188, 235, 227, 234, 264 };
      airData[62] = new double[] { 235, 227, 234, 264, 302 };
      airData[63] = new double[] { 227, 234, 264, 302, 293 };
      airData[64] = new double[] { 234, 264, 302, 293, 259 };
      airData[65] = new double[] { 264, 302, 293, 259, 229 };
      airData[66] = new double[] { 302, 293, 259, 229, 203 };
      airData[67] = new double[] { 293, 259, 229, 203, 229 };
      airData[68] = new double[] { 259, 229, 203, 229, 242 };
      airData[69] = new double[] { 229, 203, 229, 242, 233 };
      airData[70] = new double[] { 203, 229, 242, 233, 267 };
      airData[71] = new double[] { 229, 242, 233, 267, 269 };
      airData[72] = new double[] { 242, 233, 267, 269, 270 };
      airData[73] = new double[] { 233, 267, 269, 270, 315 };
      airData[74] = new double[] { 267, 269, 270, 315, 364 };
      airData[75] = new double[] { 269, 270, 315, 364, 347 };
      airData[76] = new double[] { 270, 315, 364, 347, 312 };
      airData[77] = new double[] { 315, 364, 347, 312, 274 };
      airData[78] = new double[] { 364, 347, 312, 274, 237 };
      airData[79] = new double[] { 347, 312, 274, 237, 278 };
      airData[80] = new double[] { 312, 274, 237, 278, 284 };
      airData[81] = new double[] { 274, 237, 278, 284, 277 };
      airData[82] = new double[] { 237, 278, 284, 277, 317 };
      airData[83] = new double[] { 278, 284, 277, 317, 313 };
      airData[84] = new double[] { 284, 277, 317, 313, 318 };
      airData[85] = new double[] { 277, 317, 313, 318, 374 };
      airData[86] = new double[] { 317, 313, 318, 374, 413 };
      airData[87] = new double[] { 313, 318, 374, 413, 405 };
      airData[88] = new double[] { 318, 374, 413, 405, 355 };
      airData[89] = new double[] { 374, 413, 405, 355, 306 };
      airData[90] = new double[] { 413, 405, 355, 306, 271 };
      airData[91] = new double[] { 405, 355, 306, 271, 306 };
      airData[92] = new double[] { 355, 306, 271, 306, 315 };
      airData[93] = new double[] { 306, 271, 306, 315, 301 };
      airData[94] = new double[] { 271, 306, 315, 301, 356 };
      airData[95] = new double[] { 306, 315, 301, 356, 348 };
      airData[96] = new double[] { 315, 301, 356, 348, 355 };
      airData[97] = new double[] { 301, 356, 348, 355, 422 };
      airData[98] = new double[] { 356, 348, 355, 422, 465 };
      airData[99] = new double[] { 348, 355, 422, 465, 467 };
      airData[100] = new double[] { 355, 422, 465, 467, 404 };
      airData[101] = new double[] { 422, 465, 467, 404, 347 };
      airData[102] = new double[] { 465, 467, 404, 347, 305 };
      airData[103] = new double[] { 467, 404, 347, 305, 336 };
      airData[104] = new double[] { 404, 347, 305, 336, 340 };
      airData[105] = new double[] { 347, 305, 336, 340, 318 };
      airData[106] = new double[] { 305, 336, 340, 318, 362 };
      airData[107] = new double[] { 336, 340, 318, 362, 348 };
      airData[108] = new double[] { 340, 318, 362, 348, 363 };
      airData[109] = new double[] { 318, 362, 348, 363, 435 };
      airData[110] = new double[] { 362, 348, 363, 435, 491 };
      airData[111] = new double[] { 348, 363, 435, 491, 505 };
      airData[112] = new double[] { 363, 435, 491, 505, 404 };
      airData[113] = new double[] { 435, 491, 505, 404, 359 };
      airData[114] = new double[] { 491, 505, 404, 359, 310 };
      airData[115] = new double[] { 505, 404, 359, 310, 337 };
      airData[116] = new double[] { 404, 359, 310, 337, 360 };
      airData[117] = new double[] { 359, 310, 337, 360, 342 };
      airData[118] = new double[] { 310, 337, 360, 342, 406 };
      airData[119] = new double[] { 337, 360, 342, 406, 396 };
      airData[120] = new double[] { 360, 342, 406, 396, 420 };
      airData[121] = new double[] { 342, 406, 396, 420, 472 };
      airData[122] = new double[] { 406, 396, 420, 472, 548 };
      airData[123] = new double[] { 396, 420, 472, 548, 559 };
      airData[124] = new double[] { 420, 472, 548, 559, 463 };
      airData[125] = new double[] { 472, 548, 559, 463, 407 };
      airData[126] = new double[] { 548, 559, 463, 407, 362 };
      airData[127] = new double[] { 559, 463, 407, 362, 405 };
      airData[128] = new double[] { 463, 407, 362, 405, 417 };
      airData[129] = new double[] { 407, 362, 405, 417, 391 };
      airData[130] = new double[] { 362, 405, 417, 391, 419 };
      airData[131] = new double[] { 405, 417, 391, 419, 461 };
      airData[132] = new double[] { 417, 391, 419, 461, 472 };
      airData[133] = new double[] { 391, 419, 461, 472, 535 };
      airData[134] = new double[] { 419, 461, 472, 535, 622 };
      airData[135] = new double[] { 461, 472, 535, 622, 606 };
      airData[136] = new double[] { 472, 535, 622, 606, 508 };
      airData[137] = new double[] { 535, 622, 606, 508, 461 };
      airData[138] = new double[] { 622, 606, 508, 461, 390 };
      airData[139] = new double[] { 606, 508, 461, 390, 432 };
      return airData;
    }

    static void ShowMatrix(double[][] matrix, int numRows,
      int decimals, bool indices)
    {
      int len = matrix.Length.ToString().Length;
      for (int i = 0; i < numRows; ++i)
      {
        if (indices == true)
          Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          double v = matrix[i][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
        Console.WriteLine("");
      }

      if (numRows < matrix.Length)
      {
        Console.WriteLine(". . .");
        int lastRow = matrix.Length - 1;
        if (indices == true)
          Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[lastRow].Length; ++j)
        {
          double v = matrix[lastRow][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
      }
      Console.WriteLine("\n");
    }

    static void ShowVector(double[] vector, int decimals,
      int lineLen, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % lineLen == 0) Console.WriteLine("");
        if (vector[i] >= 0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " ");
      }
      if (newLine == true)
        Console.WriteLine("");
    }


  } // Program

  public class NeuralNetwork
  {
    private int numInput; // number input nodes
    private int numHidden;
    private int numOutput;

    private double[] iNodes;
    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hNodes;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;
    private double[] oNodes;

    private Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.iNodes = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      this.hBiases = new double[numHidden];
      this.hNodes = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      this.oBiases = new double[numOutput];
      this.oNodes = new double[numOutput];

      this.rnd = new Random(0);
      this.InitializeWeights(); // all weights and biases
    } // ctor

    private static double[][] MakeMatrix(int rows,
      int cols, double v) // helper for ctor, Train
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;
      return result;
    }

    private void InitializeWeights() // helper for ctor
    {
      // initialize weights and biases to small random values
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
      this.SetWeights(initialWeights);
    }

    public void SetWeights(double[] weights)
    {
      // copy serialized weights and biases in weights[] array
      // to i-h weights, i-h biases, h-o weights, h-o biases
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array in SetWeights");

      int k = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    }

    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
      double[] oSums = new double[numOutput]; // output nodes sums

      for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
        this.iNodes[i] = xValues[i];

      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.iNodes[i] * this.ihWeights[i][j]; // note +=

      for (int i = 0; i < numHidden; ++i)  // add biases to hidden sums
        hSums[i] += this.hBiases[i];

      for (int i = 0; i < numHidden; ++i)   // apply activation
        this.hNodes[i] = HyperTan(hSums[i]); // hard-coded

      for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hNodes[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to output sums
        oSums[i] += oBiases[i];

      Array.Copy(oSums, this.oNodes, oSums.Length);  // really only 1 value

      double[] retResult = new double[numOutput]; // could define a GetOutputs 
      Array.Copy(this.oNodes, retResult, retResult.Length);
      return retResult;
    }

    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    private static double LogSig(double x)
    {
      if (x < -20.0) return 0.0; // approximation
      else if (x > 20.0) return 1.0;
      else return 1.0 / (1.0 + Math.Exp(x));
    }

    public double[] Train(double[][] trainData, int maxEpochs,
      double learnRate)
    {
      // train using back-prop
      // back-prop specific arrays
      double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
      double[] obGrads = new double[numOutput];                   // output bias gradients

      double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
      double[] hbGrads = new double[numHidden];                   // hidden bias gradients

      double[] oSignals = new double[numOutput];                  // local gradient output signals
      double[] hSignals = new double[numHidden];                  // local gradient hidden node signals

      int epoch = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // target values
      double derivative = 0.0;
      double errorSignal = 0.0;

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / 5; // interval to check error
      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = Error(trainData);
          Console.WriteLine("epoch = " + epoch + "  error = " +
            trainErr.ToString("F4"));
        }

        Shuffle(sequence); // visit each training data in random order
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues); // copy xValues in, compute outputs 

          // indices: i = inputs, j = hiddens, k = outputs

          // 1. compute output node signals (assumes softmax)
          for (int k = 0; k < numOutput; ++k)
          {
            errorSignal = tValues[k] - oNodes[k];  // Wikipedia uses (o-t)
            derivative = 1.0;  // for Identity activation
            oSignals[k] = errorSignal * derivative;
          }

          // 2. compute hidden-to-output weight gradients using output signals
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
              hoGrads[j][k] = oSignals[k] * hNodes[j];

          // 2b. compute output bias gradients using output signals
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

          // 3. compute hidden node signals
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hNodes[j]) * (1 - hNodes[j]); // for tanh
            double sum = 0.0; // need sums of output signals times hidden-to-output weights
            for (int k = 0; k < numOutput; ++k)
            {
              sum += oSignals[k] * hoWeights[j][k]; // represents error signal
            }
            hSignals[j] = derivative * sum;
          }

          // 4. compute input-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * iNodes[i];

          // 4b. compute hidden node bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

          // == update weights and biases

          // update input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta; // would be -= if (o-t)
            }
          }

          // update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
          }

          // update hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
            }
          }

          // update output node biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
          }

        } // each training item

      } // while
      double[] bestWts = GetWeights();
      return bestWts;
    } // Train

    private void Shuffle(int[] sequence) // instance method
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    } // Shuffle

    private double Error(double[][] trainData)
    {
      // average squared error per training item
      double sumSquaredError = 0.0;
      double[] xValues = new double[numInput]; // first numInput values in trainData
      double[] tValues = new double[numOutput]; // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / trainData.Length;
    } // MeanSquaredError

    public double Accuracy(double[][] testData, double howClose)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput); // get x-values
        Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
        yValues = this.ComputeOutputs(xValues);

        if (Math.Abs(yValues[0] - tValues[0]) < howClose)  // within 30
          ++numCorrect;
        else
          ++numWrong;

      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

  } // class NeuralNetwork

} // ns
