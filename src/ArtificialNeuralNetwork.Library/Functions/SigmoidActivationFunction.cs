using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library.Functions
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private readonly double _coefficient;

        public SigmoidActivationFunction(double coefficient)
        {
            _coefficient = coefficient;
        }

        public double CalculateOutput(double input)
        {
            return 1 / (1 + Math.Exp(-input * _coefficient));
        }
    }
}