using System;

namespace ImplementML
{
    class Program
    {
        static void Main()//string[] args)
        {
            Console.WriteLine("Press 1 for KNN Implementation");
            Console.WriteLine("Press 2 for Kernel Logistic Regression Implementation");
            Console.WriteLine("Press 3 for Time Series Regression Using NN Implementation");
            Console.WriteLine("Press 4 for Deep Neural Network Implementation");
            string input = Console.ReadLine();
            switch (input) {
                case "1":
                    KNNProgram.MainFunc();
                    break;
                case "2":
                    KernelLogisticProgram.MainFunc();
                    break;
                case "3":
                    NeuralTimeSeriesProgram.MainFunc();
                    break;
                case "4":
                    DeepNetTrainProgram.MainFunc();
                    break;
            }
        }
    }
}
