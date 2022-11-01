namespace ArtificialNeuralNetwork.Abstractions
{
    public interface IInputFunction
    {
        double CalculateInput(List<ISynapse> inputs);
    }
}