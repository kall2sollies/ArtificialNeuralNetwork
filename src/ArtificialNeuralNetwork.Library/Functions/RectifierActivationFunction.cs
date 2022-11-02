using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library.Functions;

public class RectifierActivationFunction : IActivationFunction
{
    public double CalculateOutput(double input)
    {
        return input > 0 ? input : 0;
    }
}