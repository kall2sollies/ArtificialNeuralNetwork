using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library.Functions;

public class WeightedSumFunction : IInputFunction
{
    public double CalculateInput(List<ISynapse> inputs)
    {
        return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
    }
}