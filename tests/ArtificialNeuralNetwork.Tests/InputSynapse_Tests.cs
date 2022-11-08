using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library;
using FluentAssertions;
using Moq;

namespace ArtificialNeuralNetwork.Tests;

public class InputSynapse_Tests
{
    [Fact]
    [Trait("Category", "Unit")]
    public void Constructor_Should_CreateSynapse_WithNoParameter()
    {
        // Arrange
        Mock<INeuron> toNeuron = new Mock<INeuron>();

        ISynapse sut = new InputSynapse(
            to: toNeuron.Object);

        // Act
        var weight = sut.Weight;
        var previousWeight = sut.PreviousWeight;

        // Assert
        weight.Should().Be(1);
        previousWeight.Should().Be(1);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void GetOutput_Should_Return_InputValue()
    {
        // Arrange
        Mock<INeuron> toNeuron = new Mock<INeuron>();

        double input = 0.42d;

        ISynapse sut = new InputSynapse(
            to: toNeuron.Object,
            inputValue: input);

        // Act
        var output = sut.GetOutput();

        // Assert
        output.Should().Be(input);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void UpdateWeight_Should_Throw()
    {
        // Arrange
        Mock<INeuron> toNeuron = new Mock<INeuron>();
            
        var initialWeight = 0.42d;
        var learningRate = 0.02d;
        var delta = 0.99d;

        ISynapse sut = new InputSynapse(
            to: toNeuron.Object);

        // Act & Assert
        sut.Invoking(x => x.UpdateWeight(learningRate, delta)).Should().Throw<InvalidOperationException>();
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void IsFromNeuron_Should_ReturnFalse()
    {
        // Arrange
        Mock<INeuron> toNeuron = new Mock<INeuron>();
        Guid toNeuronId = Guid.NewGuid();
        toNeuron.Setup(x => x.Id).Returns(toNeuronId);

        ISynapse sut = new InputSynapse(
            to: toNeuron.Object);

        // Act & Assert
        sut.IsFromNeuron(toNeuron.Object).Should().BeFalse();
    }
}