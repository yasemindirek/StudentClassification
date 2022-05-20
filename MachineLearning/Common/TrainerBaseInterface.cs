using System;
using Microsoft.ML.Data;

namespace StudentClassification.MachineLearning.Common
{
    /// <summary>
    /// Interface description class
    /// </summary>
    public interface TrainerBaseInterface
    {
        string Name { get; }
        void Fit(string trainingFileName);
        MulticlassClassificationMetrics Evaluate();
        void Save();
    }
}
