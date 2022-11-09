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
        INeuron sut = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());

        sut.AddInputSynapse(input);

        sut.CalculateOutput().Should().Be(expectedOutput);
    }

    [Theory]
    [InlineData(-0.8, 0)]
    [InlineData(1.2, 1.2)]
    [InlineData(0, 0)]
    [Trait("Category", "Unit")]
    public void PushValueOnInput_Should_UpdateOutput(double input, double expectedOutput)
    {
        INeuron sut = new Neuron(new RectifierActivationFunction(), new WeightedSumFunction());

        sut.AddInputSynapse(1.2);

        sut.CalculateOutput().Should().Be(1.2);

        sut.PushValueOnInput(input);

        sut.CalculateOutput().Should().Be(expectedOutput);
    }
}