namespace ArtificialNeuralNetwork.Abstractions;

public interface ISynapse
{
    double Weight { get; }
    double PreviousWeight { get; }
    double GetOutput();

    bool IsFromNeuron(Guid fromNeuronId);
    public bool IsFromNeuron(INeuron fromNeuron);
    void UpdateWeight(double learningRate, double delta);
}