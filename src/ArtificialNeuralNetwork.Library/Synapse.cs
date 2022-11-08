using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library;

public class Synapse : ISynapse
{
    private readonly INeuron _from;
    private readonly INeuron _to;

    public double Weight { get; private set; }
    public double PreviousWeight { get; private set; }

    public Synapse(INeuron from, INeuron to, double weight)
    {
        _from = from;
        _to = to;

        Weight = weight;
        PreviousWeight = 0;
    }

    public Synapse(INeuron from, INeuron to) : this(from, to, new Random().NextDouble())
    {
    }

    public double GetOutput() => _from.CalculateOutput();

    public bool IsFromNeuron(Guid fromNeuronId) => _from.Id == fromNeuronId;

    public bool IsFromNeuron(INeuron fromNeuron) => IsFromNeuron(fromNeuron.Id);

    public void UpdateWeight(double learningRate, double delta)
    {
        PreviousWeight = Weight;
        Weight += learningRate * delta;
    }
}