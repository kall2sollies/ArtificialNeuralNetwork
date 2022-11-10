namespace ArtificialNeuralNetwork.Abstractions;

public interface ISynapse
{
    double Weight { get; }
    double PreviousWeight { get; }
    double GetOutput();

    bool IsFromNeuron(Guid fromNeuronId);
    public bool IsToNeuron(INeuron toNeuron);
    bool IsToNeuron(Guid toNeuronId);
    public bool IsFromNeuron(INeuron fromNeuron);
    void UpdateWeight(double learningRate, double delta);
}