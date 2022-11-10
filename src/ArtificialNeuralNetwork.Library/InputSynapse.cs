using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library;

public class InputSynapse : ISynapse
{
    private readonly INeuron _to;

    public double Weight { get; init; }
    public double PreviousWeight { get; init; }
    public double Output { get; set; }

    public InputSynapse(INeuron to)
    {
        _to = to;
        Weight = 1;
        PreviousWeight = 1;
    }

    public InputSynapse(INeuron to, double inputValue) : this(to)
    {
        Output = inputValue;
    }

    public double GetOutput() => Output;

    public bool IsFromNeuron(Guid fromNeuronId) => false;

    public bool IsFromNeuron(INeuron fromNeuron) => IsFromNeuron(fromNeuron.Id);

    public bool IsToNeuron(Guid toNeuronId) => _to.Id == toNeuronId;

    public bool IsToNeuron(INeuron toNeuron) => IsToNeuron(toNeuron.Id);

    public void UpdateWeight(double learningRate, double delta)
    {
        throw new InvalidOperationException("Input synapses have immutable weight value of 1.0");
    }
}