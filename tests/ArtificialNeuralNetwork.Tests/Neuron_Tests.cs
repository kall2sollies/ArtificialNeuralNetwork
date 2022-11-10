using System.Runtime.InteropServices;
using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library;
using ArtificialNeuralNetwork.Library.Functions;
using FluentAssertions;
using Moq;

namespace ArtificialNeuralNetwork.Tests;

public class Neuron_Tests
{
    [Theory]
    [InlineData(0.5, 0.6, 0)]
    [InlineData(0.7, 0.6, 1)]
    [Trait("Category", "Unit")]
    public void CalculateOutput_Should_Activate_WhenUsingStepActivationFunction(double mockInput, double threshold, double expectedOutput)
    {
        // Arrange
        Mock<IInputFunction> inputFn = new Mock<IInputFunction>();
        inputFn.Setup(x =>
                x.CalculateInput(It.IsAny<List<ISynapse>>()))
            .Returns(mockInput);

        IActivationFunction activationFn = new StepActivationFunction(threshold);

        // Act
        INeuron sut = new Neuron(activationFn, inputFn.Object);

        // Assert
        sut.CalculateOutput().Should().Be(expectedOutput);
    }

    [Theory]
    [InlineData(0.5, 0.5)]
    [InlineData(-0.7, 0)]
    [Trait("Category", "Unit")]
    public void CalculateOutput_Should_Activate_WhenUsingRectifierActivationFunction(double mockInput, double expectedOutput)
    {
        // Arrange
        Mock<IInputFunction> inputFn = new Mock<IInputFunction>();
        inputFn.Setup(x =>
                x.CalculateInput(It.IsAny<List<ISynapse>>()))
            .Returns(mockInput);

        IActivationFunction activationFn = new RectifierActivationFunction();

        // Act
        INeuron sut = new Neuron(activationFn, inputFn.Object);

        // Assert
        sut.CalculateOutput().Should().Be(expectedOutput);
    }

    [Theory]
    [InlineData(0.5, true)]
    [InlineData(-0.7, false)]
    [Trait("Category", "Unit")]
    public void CalculateOutput_Should_Activate_WhenUsingSigmoidActivationFunction(double mockInput, bool comparison)
    {
        // Arrange
        Mock<IInputFunction> inputFn = new Mock<IInputFunction>();
        inputFn.Setup(x =>
                x.CalculateInput(It.IsAny<List<ISynapse>>()))
            .Returns(mockInput);

        IActivationFunction activationFn = new SigmoidActivationFunction(1);

        // Act
        INeuron sut = new Neuron(activationFn, inputFn.Object);

        // Assert
        (sut.CalculateOutput() > 0.5).Should().Be(comparison);
    }

    [Theory]
    [InlineData(-0.8, 0)]
    [InlineData(1.2, 1.2)]
    [InlineData(0, 0)]
    [Trait("Category", "Unit")]
    public void AddInputSynapse_Should_ProduceExpectedOutput(double input, double expectedOutput)
    {
        // Arrange
        INeuron sut = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());

        // Act
        sut.AddInputSynapse(input);

        // Assert
        sut.CalculateOutput().Should().Be(expectedOutput);
    }

    [Theory]
    [InlineData(-0.8, 0)]
    [InlineData(1.2, 1.2)]
    [InlineData(0, 0)]
    [Trait("Category", "Unit")]
    public void PushValueOnInput_Should_UpdateOutput(double input, double expectedOutput)
    {
        // Arrange
        INeuron sut = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());
        sut.AddInputSynapse(1.2);
        sut.CalculateOutput().Should().Be(1.2);

        // Act
        sut.PushValueOnInput(input);

        // Assert
        sut.CalculateOutput().Should().Be(expectedOutput);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void AddInputNeuron_Should_ConnectNeuronsBothWays()
    {
        // Arrange
        INeuron input1 = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());
        INeuron input2 = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());
        INeuron sut = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());

        // Act
        sut.AddInputNeuron(input1);
        sut.AddInputNeuron(input2);

        // Assert
        sut.Inputs.Count.Should().Be(2);
        sut.Outputs.Count.Should().Be(0);
        input1.Outputs.Count.Should().Be(1);
        input2.Outputs.Count.Should().Be(1);

        sut.Inputs.Count(x => x.IsFromNeuron(input1)).Should().Be(1);
        sut.Inputs.Count(x => x.IsFromNeuron(input2)).Should().Be(1);
        input1.Outputs.Count(x => x.IsToNeuron(sut)).Should().Be(1);
        input2.Outputs.Count(x => x.IsToNeuron(sut)).Should().Be(1);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void AddOutputNeuron_Should_ConnectNeuronsBothWays()
    {
        // Arrange
        INeuron output1 = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());
        INeuron output2 = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());
        INeuron sut = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());

        // Act
        sut.AddOutputNeuron(output1);
        sut.AddOutputNeuron(output2);

        // Assert
        sut.Outputs.Count.Should().Be(2);
        sut.Inputs.Count.Should().Be(0);
        output1.Inputs.Count.Should().Be(1);
        output2.Inputs.Count.Should().Be(1);

        sut.Outputs.Count(x => x.IsToNeuron(output1)).Should().Be(1);
        sut.Outputs.Count(x => x.IsToNeuron(output2)).Should().Be(1);
        output1.Inputs.Count(x => x.IsFromNeuron(sut)).Should().Be(1);
        output2.Inputs.Count(x => x.IsFromNeuron(sut)).Should().Be(1);
    }
}