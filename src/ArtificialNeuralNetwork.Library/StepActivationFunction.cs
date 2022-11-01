using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library
{
    public class StepActivationFunction : IActivationFunction
    {
        private readonly double _threshold;

        public StepActivationFunction(double threshold)
        {
            _threshold = threshold;
        }

        public double CalculateOutput(double input)
        {
            return input > _threshold ? 1d : 0d;
        }
    }
}
