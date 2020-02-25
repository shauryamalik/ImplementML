using System;

namespace ImplementML
{
    class Program
    {
        static void Main()//string[] args)
        {
            Console.WriteLine("Press 1 for KNN Implementation");
            Console.WriteLine("Press 2 for Kernel Logistic Regression Implementation");
            string input = Console.ReadLine();
            switch (input) {
                case "1":
                    KNNProgram.MainFunc();
                    break;
                case "2":
                    KernelLogisticProgram.MainFunc();
                    break;
            }
        }
    }
}
