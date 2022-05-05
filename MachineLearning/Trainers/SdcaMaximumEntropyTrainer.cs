using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using StudentClassification.MachineLearning.Common;

namespace StudentClassification.MachineLearning.Trainers
{
    public class SdcaMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
    {
        public SdcaMaximumEntropyTrainer() : base()
        {
            Name = "Sdca Maximum Entropy";
            _model = MlContext.MulticlassClassification.Trainers
              .SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
