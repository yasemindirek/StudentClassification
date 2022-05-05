using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using StudentClassification.MachineLearning.Common;

namespace StudentClassification.MachineLearning.Trainers
{
    public class LbfgsMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
    {
        public LbfgsMaximumEntropyTrainer() : base()
        {
            Name = "LBFGS Maximum Entropy";
            _model = MlContext.MulticlassClassification.Trainers
              .LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
