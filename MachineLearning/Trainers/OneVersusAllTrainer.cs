using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using StudentClassification.MachineLearning.Common;

namespace StudentClassification.MachineLearning.Trainers
{
    public class OneVersusAllTrainer : TrainerBase<OneVersusAllModelParameters>
    {
        public OneVersusAllTrainer() : base()
        {
            Name = "One Versus All";
            _model = MlContext.MulticlassClassification.Trainers
          .OneVersusAll(binaryEstimator: MlContext.BinaryClassification.Trainers.SgdCalibrated());
        }
    }
}
