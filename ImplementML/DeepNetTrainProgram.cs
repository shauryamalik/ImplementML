﻿using System;
namespace ImplementML
{
  class DeepNetTrainProgram
  {
    public static void MainFunc()//string[] args)
    {
      Console.WriteLine("\nBegin deep net training demo \n");

      int numInput = 4;
      int[] numHidden = new int[] { 10, 10, 10 };
      int numOutput = 3;

      int numDataItems = 2000;
      Console.WriteLine("Generating " + numDataItems + " artificial training data items ");
      double[][] trainData = MakeData(numDataItems, numInput, numHidden, numOutput, 5);
      Console.WriteLine("\nDone. Training data is: ");
      ShowMatrix(trainData, 3, 2, true);

      Console.WriteLine("\nCreating a 4-(10,10,10)-3 deep neural network (tanh & softmax) \n");
      DeepNet dn = new DeepNet(numInput, numHidden, numOutput);
      //dn.Dump();

      int maxEpochs = 2000;
      double learnRate = 0.001;
      double momentum = 0.01;
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting learnRate = " + learnRate.ToString("F3"));
      Console.WriteLine("Setting momentumm = " + momentum.ToString("F3"));
      Console.WriteLine("\nStart training using back-prop with mean squared error \n");
      double[] wts = dn.Train(trainData, maxEpochs, learnRate, momentum, 10);  // show error every maxEpochs / 10 
      Console.WriteLine("Training complete \n");

      double trainError = dn.Error(trainData, false);
      double trainAcc = dn.Accuracy(trainData, false);
      Console.WriteLine("Final model MS error = " + trainError.ToString("F4"));
      Console.WriteLine("Final model accuracy = " + trainAcc.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();

    } // Main

    static double[][] MakeData(int numItems, int numInput, int[] numHidden, int numOutput, int seed)
    {
      // generate data using a Deep NN (tanh hidden activation)
      DeepNet dn = new DeepNet(numInput, numHidden, numOutput);  // make a DNN generator
      Random rrnd = new Random(seed);  // to make random weights & biases, random input vals
      double wtLo = -9.0;
      double wtHi = 9.0;
      int nw = DeepNet.NumWeights(numInput, numHidden, numOutput);
      double[] wts = new double[nw];

      for (int i = 0; i < nw; ++i)
        wts[i] = (wtHi - wtLo) * rrnd.NextDouble() + wtLo;
      dn.SetWeights(wts);

      double[][] result = new double[numItems][];  // make the result matrix holder
      for (int r = 0; r < numItems; ++r)
        result[r] = new double[numInput + numOutput];  // allocate the cols

      double inLo = -4.0;    // pseudo-Gaussian scaling
      double inHi = 4.0;
      for (int r = 0; r < numItems; ++r)  // each row
      {
        double[] inputs = new double[numInput];  // random input values
        
        for (int i = 0; i < numInput; ++i)
          inputs[i] = (inHi - inLo) * rrnd.NextDouble() + inLo;

        //ShowVector(inputs, 2);

        double[] probs = dn.ComputeOutputs(inputs);  // compute the outputs (as softmax probs) like [0.10, 0.15, 0.55, 0.20]
        //dn.Dump();
        //Console.ReadLine();
        //ShowVector(probs, 4);
         double[] outputs = ProbsToClasses(probs);  // convert to outputs like [0, 0, 1, 0]

        int c = 0;
        for (int i = 0; i < numInput; ++i)
          result[r][c++] = inputs[i];
        for (int i = 0; i < numOutput; ++i)
          result[r][c++] = outputs[i];
        //Console.WriteLine("");
      }
      return result;

    } // MakeData

    static double[] ProbsToClasses(double[] probs)
    {
      double[] result = new double[probs.Length];
      int idx = MaxIndex(probs);
      result[idx] = 1.0;
      return result;
    }

    static int MaxIndex(double[] probs)
    {
      int maxIdx = 0;
      double maxVal = probs[0];

      for (int i = 0; i < probs.Length; ++i)
      {
        if (probs[i] > maxVal)
        {
          maxVal = probs[i];
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    public static void ShowMatrix(double[][] matrix, int numRows,
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

    static void ShowMatrix(double[][] matrix, int numRows, int numDec)
    {
      for (int r = 0; r < numRows; ++r)
      {
        for (int c = 0; c < matrix[r].Length; ++c)
        {
          if (matrix[r][c] >= 0.0) Console.Write(" ");  // '+'
          Console.Write(matrix[r][c].ToString("F" + numDec) + "  ");
        }
        Console.WriteLine("");
      }
      Console.WriteLine("");
    }

    static void ShowVector(double[] vector, int numDec)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] >= 0.0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + numDec) + "  ");
      }
      Console.WriteLine("");
    }

  } // Program

  public class DeepNet
  {
    public static Random rnd;  // weight init and train shuffle

    public int nInput;  // number input nodes
    public int[] nHidden;  // number hidden nodes, each layer
    public int nOutput;  // number output nodes
    public int nLayers;  // number hidden node layers

    public double[] iNodes;  // input nodes
    public double[][] hNodes;
    public double[] oNodes;

    public double[][] ihWeights;  // input- 1st hidden
    public double[][][] hhWeights; // hidden-hidden
    public double[][] hoWeights;  // last hidden-output

    public double[][] hBiases;  // hidden node biases
    public double[] oBiases;  // output node biases

    public double ihGradient00;  // one gradient to monitor

    public DeepNet(int numInput, int[] numHidden, int numOutput)
    {
      rnd = new Random(0);  // seed could be a ctor parameter

      this.nInput = numInput;
      this.nHidden = new int[numHidden.Length];
      for (int i = 0; i < numHidden.Length; ++i)
        this.nHidden[i] = numHidden[i];
      this.nOutput = numOutput;
      this.nLayers = numHidden.Length;

      iNodes = new double[numInput];
      hNodes = MakeJaggedMatrix(numHidden);
      oNodes = new double[numOutput];

      ihWeights = MakeMatrix(numInput, numHidden[0]);
      hoWeights = MakeMatrix(numHidden[nLayers - 1], numOutput);

      hhWeights = new double[nLayers - 1][][];  // if 3 h layer, 2 h-h weights[][]
      for (int h = 0; h < hhWeights.Length; ++h)
      {
        int rows = numHidden[h];
        int cols = numHidden[h + 1];
        hhWeights[h] = MakeMatrix(rows, cols);
      }

      hBiases = MakeJaggedMatrix(numHidden);  // pass an array of lengths
      oBiases = new double[numOutput];

      InitializeWeights();  // small randomm non-zero values
    } // ctor

    public void InitializeWeights()
    {
      // make wts
      double lo = -0.10;
      double hi = +0.10;
      int numWts = DeepNet.NumWeights(this.nInput, this.nHidden, this.nOutput);
      double[] wts = new double[numWts];
      for (int i = 0; i < numWts; ++i)
        wts[i] = (hi - lo) * rnd.NextDouble() + lo;
      this.SetWeights(wts);
    }

    public double[] ComputeOutputs(double[] xValues)
    {
      // 'xValues' might have class label or not
      // copy vals into iNodes
      for (int i = 0; i < nInput; ++i)  // possible trunc
        iNodes[i] = xValues[i];

      // zero-out all hNodes, oNodes
      for (int h = 0; h < nLayers; ++h)
        for (int j = 0; j < nHidden[h]; ++j)
          hNodes[h][j] = 0.0;

      for (int k = 0; k < nOutput; ++k)
        oNodes[k] = 0.0;

      // input to 1st hid layer
      for (int j = 0; j < nHidden[0]; ++j)  // each hidden node, 1st layer
      {
        for (int i = 0; i < nInput; ++i)
          hNodes[0][j] += ihWeights[i][j] * iNodes[i];
        // add the bias
        hNodes[0][j] += hBiases[0][j];
        // apply activation
        hNodes[0][j] = Math.Tanh(hNodes[0][j]);
      }

      // each remaining hidden node
      for (int h = 1; h < nLayers; ++h)
      {
        for (int j = 0; j < nHidden[h]; ++j)  // 'to index'
        {
          for (int jj = 0; jj < nHidden[h - 1]; ++jj)  // 'from index'
          {
            hNodes[h][j] += hhWeights[h - 1][jj][j] * hNodes[h - 1][jj];
          }
          hNodes[h][j] += hBiases[h][j];  // add bias value
          hNodes[h][j] = Math.Tanh(hNodes[h][j]);  // apply activation
        }
      }

      // compute ouput node values
      for (int k = 0; k < nOutput; ++k)
      {
        for (int j = 0; j < nHidden[nLayers - 1]; ++j)
        {
          oNodes[k] += hoWeights[j][k] * hNodes[nLayers - 1][j];
        }
        oNodes[k] += oBiases[k];  // add bias value
        //Console.WriteLine("Pre-softmax output node [" + k + "] value = " + oNodes[k].ToString("F4"));
      }

      double[] retResult = Softmax(oNodes);  // softmax activation all oNodes

      for (int k = 0; k < nOutput; ++k)
        oNodes[k] = retResult[k];
      return retResult;  // calling convenience

    } // ComputeOutputs

    public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum, int showEvery)
    {
      // no momentum right now
      // each weight (and bias) needs a big_delta. big_delta is just learnRate * "a gradient"
      // so goal is to find "a gradient".
      // the gradient (the term can have several meanings) is "a signal" * "an input"
      // the signal 

      // 1. each weight and bias has a 'gradient' (partial dervative)
      double[][] hoGrads = MakeMatrix(nHidden[nLayers-1], nOutput);  // last_hidden layer - output weights grads
      double[][][] hhGrads = new double[nLayers - 1][][];
      for (int h = 0; h < hhGrads.Length; ++h)
      {
        int rows = nHidden[h];
        int cols = nHidden[h + 1];
        hhGrads[h] = MakeMatrix(rows, cols);
      }
      double[][] ihGrads = MakeMatrix(nInput, nHidden[0]);  // input-first_hidden wts gradients
      // biases
      double[] obGrads = new double[nOutput];  // output node bias grads
      double[][] hbGrads = MakeJaggedMatrix(nHidden);  // hidden node bias grads

      // 2. each output node and each hidden node has a 'signal' == gradient without associated input (lower case delta in Wikipedia)
      double[] oSignals = new double[nOutput];
      double[][] hSignals = MakeJaggedMatrix(nHidden);

      // 3. for momentum, each weight and bias needs to store the prev epoch delta
      // the structure for prev deltas is same as for Weights & Biases, which is same as for Grads

      double[][] hoPrevWeightsDelta = MakeMatrix(nHidden[nLayers - 1], nOutput);  // last_hidden layer - output weights momentum term
      double[][][] hhPrevWeightsDelta = new double[nLayers - 1][][];
      for (int h = 0; h < hhPrevWeightsDelta.Length; ++h)
      {
        int rows = nHidden[h];
        int cols = nHidden[h + 1];
        hhPrevWeightsDelta[h] = MakeMatrix(rows, cols);
      }
      double[][] ihPrevWeightsDelta = MakeMatrix(nInput, nHidden[0]);  // input-first_hidden wts gradients
      double[] oPrevBiasesDelta = new double[nOutput];  // output node bias prev deltas
      double[][] hPrevBiasesDelta = MakeJaggedMatrix(nHidden);  // hidden node bias prev deltas

      int epoch = 0;
      double[] xValues = new double[nInput];  // not necessary - could copy direct from data item to iNodes
      double[] tValues = new double[nOutput];  // not necessary
      double derivative = 0.0;  // of activation (softmax or tanh or log-sigmoid or relu)
      double errorSignal = 0.0;  // target - output

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / showEvery; // interval to check & display  error
      while (epoch < maxEpochs)
      {
        ++epoch;
        if (epoch % errInterval == 0 && epoch < maxEpochs)  // display curr MSE
        {
          double trainErr = Error(trainData, false);  // using curr weights & biases
          double trainAcc = this.Accuracy(trainData, false);
          Console.Write("epoch = " + epoch + "  MS error = " +
            trainErr.ToString("F4"));
          Console.WriteLine("  accuracy = " +
            trainAcc.ToString("F4"));

          Console.WriteLine("input-to-hidden [0][0] gradient = " + this.ihGradient00.ToString("F6"));
          Console.WriteLine("");
          //this.Dump();
          // Console.ReadLine();
        }

        Shuffle(sequence); // must visit each training data in random order in stochastic GD

        for (int ii = 0; ii < trainData.Length; ++ii)  // each train data item
        {
          int idx = sequence[ii];  // idx points to a data item
          Array.Copy(trainData[idx], xValues, nInput);
          Array.Copy(trainData[idx], nInput, tValues, 0, nOutput);
          ComputeOutputs(xValues); // copy xValues in, compute outputs using curr weights & biases, ignore return

          // must compute signals from right-to-left
          // weights and bias gradients can be computed left-to-right
          // weights and bias gradients can be updated left-to-right

          // x. compute output node signals (assumes softmax) depends on target values to the right
          for (int k = 0; k < nOutput; ++k)
          {
            errorSignal = tValues[k] - oNodes[k];  // Wikipedia uses (o-t)
            derivative = (1 - oNodes[k]) * oNodes[k]; // for softmax (same as log-sigmoid) with MSE
            //derivative = 1.0;  // for softmax with cross-entropy
            oSignals[k] = errorSignal * derivative;  // we'll use this for ho-gradient and hSignals
          }

          // x. compute signals for last hidden layer (depends on oNodes values to the right)
          int lastLayer = nLayers - 1;
          for (int j = 0; j < nHidden[lastLayer]; ++j)
          {
            derivative = (1 + hNodes[lastLayer][j]) * (1 - hNodes[lastLayer][j]); // for tanh!
            double sum = 0.0; // need sums of output signals times hidden-to-output weights
            for (int k = 0; k < nOutput; ++k)
            {
              sum += oSignals[k] * hoWeights[j][k]; // represents error signal
            }
            hSignals[lastLayer][j] = derivative * sum;
          }

          // x. compute signals for all the non-last layer hidden nodes (depends on layer to the right)
          for (int h = lastLayer - 1; h >= 0; --h)  // each hidden layer, right-to-left
          {
            for (int j = 0; j < nHidden[h]; ++j)  // each node
            {
              derivative = (1 + hNodes[h][j]) * (1 - hNodes[h][j]); // for tanh
              // derivative = hNodes[h][j];

              double sum = 0.0; // need sums of output signals times hidden-to-output weights

              for (int jj = 0; jj < nHidden[h+1]; ++jj)  // layer to right of curr layer
                sum += hSignals[h+1][jj] * hhWeights[h][j][jj]; 

              hSignals[h][j] = derivative * sum;

            } // j
          } // h

          // at this point, all hidden and output node signals have been computed
          // calculate gradients left-to-right

          // x. compute input-to-hidden weights gradients using iNodes & hSignal[0]
          for (int i = 0; i < nInput; ++i)
            for (int j = 0; j < nHidden[0]; ++j)
              ihGrads[i][j] = iNodes[i] * hSignals[0][j];  // "from" input & "to" signal

          // save the special monitored ihGradient00
          this.ihGradient00 = ihGrads[0][0];

          // x. compute hidden-to-hidden gradients
          for (int h = 0; h < nLayers - 1; ++h)
          {
            for (int j = 0; j < nHidden[h]; ++j)
            {
              for (int jj = 0; jj < nHidden[h + 1]; ++jj)
              {
                hhGrads[h][j][jj] = hNodes[h][j] * hSignals[h + 1][jj];
              }
            }
          }

          // x. compute hidden-to-output gradients
          for (int j = 0; j < nHidden[lastLayer]; ++j)
          {
            for (int k = 0; k < nOutput; ++k)
            {
              hoGrads[j][k] = hNodes[lastLayer][j] * oSignals[k];  // from last hidden, to oSignals
            }
          }

          // compute bias gradients
          // a bias is like a weight on the left/before
          // so there's a dummy input of 1.0 and we use the signal of the 'current' layer

          // x. compute all hidden bias gradients
          // a gradient needs the "left/from" input and the "right/to" signal
          // for biases we use a dummy 1.0 input

          for (int h = 0; h < nLayers; ++h)
          {
            for (int j = 0; j < nHidden[h]; ++j)
            {
              hbGrads[h][j] = 1.0 * hSignals[h][j];
            }
          }

          // x. output bias gradients
          for (int k = 0; k < nOutput; ++k)
          {
            obGrads[k] = 1.0 * oSignals[k];
          }

          // at this point all signals have been computed and all gradients have been computed 
          // so can use gradients to update all weights and biases.
          // save each delta for the momentum

          // x. update input-to-first_hidden weights using ihWeights & ihGrads
          for (int i = 0; i < nInput; ++i)
          {
            for (int j = 0; j < nHidden[0]; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta;
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihPrevWeightsDelta[i][j] = delta;
            }
          }

          // other hidden-to-hidden weights using hhWeights & hhGrads
          for (int h = 0; h < nLayers - 1; ++h)
          {
            for (int j = 0; j < nHidden[h]; ++j)
            {
              for (int jj = 0; jj < nHidden[h + 1]; ++jj)
              {
                double delta = hhGrads[h][j][jj] * learnRate;
                hhWeights[h][j][jj] += delta;
                hhWeights[h][j][jj] += hhPrevWeightsDelta[h][j][jj] * momentum;
                hhPrevWeightsDelta[h][j][jj] = delta;
              }
            }
          }

          // update hidden-to-output weights using hoWeights & hoGrads
          for (int j = 0; j < nHidden[lastLayer]; ++j)
          {
            for (int k = 0; k < nOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // update hidden biases using hBiases & hbGrads
          for (int h = 0; h < nLayers; ++h)
          {
            for (int j = 0; j < nHidden[h]; ++j)
            {
              double delta = hbGrads[h][j] * learnRate;
              hBiases[h][j] += delta;
              hBiases[h][j] += hPrevBiasesDelta[h][j] * momentum;
              hPrevBiasesDelta[h][j] = delta;
            }
          }

          // update output biases using oBiases & obGrads
          for (int k = 0; k < nOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }
          
          // Whew!
        }  // for each train data item
      }  // while

      double[] bestWts = this.GetWeights();
      return bestWts;
    } // Train

    public double Error(double[][] data, bool verbose)
    {
      // mean squared error using current weights & biases
      double sumSquaredError = 0.0;
      double[] xValues = new double[nInput]; // first numInput values in trainData
      double[] tValues = new double[nOutput]; // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, nInput);
        Array.Copy(data[i], nInput, tValues, 0, nOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights

        if (verbose == true)
        {
          ShowVector(yValues, 4);
          ShowVector(tValues, 4);
          Console.WriteLine("");
          Console.ReadLine();
        }


        for (int j = 0; j < nOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / (data.Length * nOutput);  // average per item
    } // Error

    public double Error(double[][] data, double[] weights)
    {
      // mean squared error using supplied weights & biases
      this.SetWeights(weights);

      double sumSquaredError = 0.0;
      double[] xValues = new double[nInput]; // first numInput values in trainData
      double[] tValues = new double[nOutput]; // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, nInput);
        Array.Copy(data[i], nInput, tValues, 0, nOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < nOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / data.Length;
    } // Error

    public double Accuracy(double[][] data, bool verbose)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[nInput]; // inputs
      double[] tValues = new double[nOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, nInput); // get x-values
        Array.Copy(data[i], nInput, tValues, 0, nOutput); // get t-values
        yValues = this.ComputeOutputs(xValues);

        if (verbose == true)
        {
          ShowVector(yValues, 4);
          ShowVector(tValues, 4);
          Console.WriteLine("");
          Console.ReadLine();
        }

        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    private static void ShowVector(double[] vector, int dec)
    {
      for (int i = 0; i < vector.Length; ++i)
        Console.Write(vector[i].ToString("F" + dec) + " ");
      Console.WriteLine("");
    }

    public double Accuracy(double[][] data, double[] weights)
    {
      this.SetWeights(weights);
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[nInput]; // inputs
      double[] tValues = new double[nOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, nInput); // get x-values
        Array.Copy(data[i], nInput, tValues, 0, nOutput); // get t-values
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
      // index of largest value in vector[]
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }

    private void Shuffle(int[] sequence) // instance method
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    } // Shuffle

   

    private static double MyTanh(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    //private static double[] Softmax(double[] oNodes) // does all output nodes at once so scale doesn't have to be re-computed each time
    //{
    //  // Softmax using the max trick to avoid overflow
    //  // determine max output node value
    //  double max = oNodes[0];
    //  for (int i = 0; i < oNodes.Length; ++i)
    //    if (oNodes[i] > max) max = oNodes[i];

    //  // determine scaling factor -- sum of exp(each val - max)
    //  double scale = 0.0;
    //  for (int i = 0; i < oNodes.Length; ++i)
    //    scale += Math.Exp(oNodes[i] - max);

    //  double[] result = new double[oNodes.Length];
    //  for (int i = 0; i < oNodes.Length; ++i)
    //    result[i] = Math.Exp(oNodes[i] - max) / scale;

    //  return result; // now scaled so that xi sum to 1.0
    //}

    private static double[] Softmax(double[] oSums)
    {
      // does all output nodes at once so scale
      // doesn't have to be re-computed each time.
      // possible overflow . . . use max trick

      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result; // now scaled so that xi sum to 1.0
    }


    public void SetWeights(double[] wts)
    {
      // order: ihweights - hhWeights[] - hoWeights - hBiases[] - oBiases
      int nw = NumWeights(this.nInput, this.nHidden, this.nOutput);  // total num wts + biases
      if (wts.Length != nw)
        throw new Exception("Bad wts[] length in SetWeights()");
      int ptr = 0;  // pointer into wts[]

      for (int i = 0; i < nInput; ++i)  // input node
        for (int j = 0; j < hNodes[0].Length; ++j)  // 1st hidden layer nodes
          ihWeights[i][j] = wts[ptr++];

      for (int h = 0; h < nLayers - 1; ++h)  // not last h layer
      {
        for (int j = 0; j < nHidden[h]; ++j)  // from node
        {
          for (int jj = 0; jj < nHidden[h + 1]; ++jj)  // to node
          {
            hhWeights[h][j][jj] = wts[ptr++];
          }
        }
      }

      int hi = this.nLayers - 1;  // if 3 hidden layers (0,1,2) last is 3-1 = [2]
      for (int j = 0; j < this.nHidden[hi]; ++j)
      {
        for (int k = 0; k < this.nOutput; ++k)
        {
          hoWeights[j][k] = wts[ptr++];
        }
      }

      for (int h = 0; h < nLayers; ++h)  // hidden node biases
      {
        for (int j = 0; j < this.nHidden[h]; ++j)
        {
          hBiases[h][j] = wts[ptr++];
        }
      }

      for (int k = 0; k < nOutput; ++k)
      {
        oBiases[k] = wts[ptr++];
      }
    } // SetWeights

    public double[] GetWeights()
    {
      // order: ihweights -> hhWeights[] -> hoWeights -> hBiases[] -> oBiases
      int nw = NumWeights(this.nInput, this.nHidden, this.nOutput);  // total num wts + biases
      double[] result = new double[nw];

      int ptr = 0;  // pointer into result[]

      for (int i = 0; i < nInput; ++i)  // input node
        for (int j = 0; j < hNodes[0].Length; ++j)  // 1st hidden layer nodes
          result[ptr++] = ihWeights[i][j];

      for (int h = 0; h < nLayers - 1; ++h)  // not last h layer
      {
        for (int j = 0; j < nHidden[h]; ++j)  // from node
        {
          for (int jj = 0; jj < nHidden[h + 1]; ++jj)  // to node
          {
            result[ptr++] = hhWeights[h][j][jj];
          }
        }
      }

      int hi = this.nLayers - 1;  // if 3 hidden layers (0,1,2) last is 3-1 = [2]
      for (int j = 0; j < this.nHidden[hi]; ++j)
      {
        for (int k = 0; k < this.nOutput; ++k)
        {
          result[ptr++] = hoWeights[j][k];
        }
      }

      for (int h = 0; h < nLayers; ++h)  // hidden node biases
      {
        for (int j = 0; j < this.nHidden[h]; ++j)
        {
          result[ptr++] = hBiases[h][j];
        }
      }

      for (int k = 0; k < nOutput; ++k)
      {
        result[ptr++] = oBiases[k];
      }
      return result;
    }

    public static int NumWeights(int numInput, int[] numHidden, int numOutput)
    {
      // total num weights and biases
      int ihWts = numInput * numHidden[0];

      int hhWts = 0;
      for (int j = 0; j < numHidden.Length - 1; ++j)
      {
        int rows = numHidden[j];
        int cols = numHidden[j + 1];
        hhWts += rows * cols;
      }
      int hoWts = numHidden[numHidden.Length - 1] * numOutput;

      int hbs = 0;
      for (int i = 0; i < numHidden.Length; ++i)
        hbs += numHidden[i];

      int obs = numOutput;
      int nw = ihWts + hhWts + hoWts + hbs + obs;
      return nw;
    }

    public static double[][] MakeJaggedMatrix(int[] cols)
    {
      // array of arrays using size info in cols[]
      int rows = cols.Length;  // num rows inferred by col count
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
      {
        int ncol = cols[i];
        result[i] = new double[ncol];
      }
      return result;
    }

    public static double[][] MakeMatrix(int rows, int cols)
    {
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
      return result;
    }

    public void Dump()
    {
      for (int i = 0; i < nInput; ++i)
      {
        Console.WriteLine("input node [" + i + "] = " + iNodes[i].ToString("F4"));
      }
      for (int h = 0; h < nLayers; ++h)
      {
        Console.WriteLine("");
        for (int j = 0; j < nHidden[h]; ++j)
        {
          Console.WriteLine("hidden layer " + h + " node [" + j + "] = " + hNodes[h][j].ToString("F4"));
        }
      }
      Console.WriteLine("");
      for (int k = 0; k < nOutput; ++k)
      {
        Console.WriteLine("output node [" + k + "] = " + oNodes[k].ToString("F4"));
      }

      Console.WriteLine("");
      for (int i = 0; i < nInput; ++i)
      {
        for (int j = 0; j < nHidden[0]; ++j)
        {
          Console.WriteLine("input-hidden wt [" + i + "][" + j + "] = " + ihWeights[i][j].ToString("F4"));
        }
      }

      for (int h = 0; h < nLayers - 1; ++h)  // note
      {
        Console.WriteLine("");
        for (int j = 0; j < nHidden[h]; ++j)
        {
          for (int jj = 0; jj < nHidden[h + 1]; ++jj)
          {
            Console.WriteLine("hidden-hidden wt layer " + h + " to layer " + (h + 1) + " node [" + j + "] to [" + jj + "] = " + hhWeights[h][j][jj].ToString("F4"));
          }
        }
      }

      Console.WriteLine("");
      for (int j = 0; j < nHidden[nLayers - 1]; ++j)
      {
        for (int k = 0; k < nOutput; ++k)
        {
          Console.WriteLine("hidden-output wt [" + j + "][" + k + "] = " + hoWeights[j][k].ToString("F4"));
        }
      }

      for (int h = 0; h < nLayers; ++h)
      {
        Console.WriteLine("");
        for (int j = 0; j < nHidden[h]; ++j)
        {
          Console.WriteLine("hidden layer " + h + " bias [" + j + "] = " + hBiases[h][j].ToString("F4"));
        }
      }

      Console.WriteLine("");
      for (int k = 0; k < nOutput; ++k)
      {
        Console.WriteLine("output node bias [" + k + "] = " + oBiases[k].ToString("F4"));
      }

    } // Dump

  } // class DeepNet

} // ns
