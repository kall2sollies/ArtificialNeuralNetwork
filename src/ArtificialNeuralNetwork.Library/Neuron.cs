using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library;

public class Neuron : INeuron
{
    private readonly IActivationFunction _activationFunction;
    private readonly IInputFunction _inputFunction;

    public Guid Id { get; }
    public double PreviousPartialDerivate { get; set; }
    public List<ISynapse> Inputs { get; set; } = new();
    public List<ISynapse> Outputs { get; set; } = new();

    public Neuron(IActivationFunction activationFunction, 
                  IInputFunction inputFunction)
    {
        Id = Guid.NewGuid();

        _activationFunction = activationFunction;
        _inputFunction = inputFunction;
    }

    public void AddInputNeuron(INeuron inputNeuron)
    {
        ISynapse inputSynapse = new Synapse(inputNeuron, this);

        Inputs.Add(inputSynapse);

        inputNeuron.Outputs.Add(inputSynapse);
    }

    public void AddOutputNeuron(INeuron outputNeuron)
    {
        ISynapse outputSynapse = new Synapse(this, outputNeuron);

        Outputs.Add(outputSynapse);

        outputNeuron.Inputs.Add(outputSynapse);
    }

    public double CalculateOutput()
    {
        var inputValue = _inputFunction.CalculateInput(Inputs);

        return _activationFunction.CalculateOutput(inputValue);
    }

    public void AddInputSynapse(double inputValue)
    {
        ISynapse inputSynapse = new InputSynapse(this, inputValue);

        Inputs.Add(inputSynapse);
    }

    public void PushValueOnInput(double inputValue)
    {
        InputSynapse inputSynapse = (InputSynapse)Inputs.First(x => x.GetType() == typeof(InputSynapse));

        inputSynapse.Output = inputValue;
    }
}