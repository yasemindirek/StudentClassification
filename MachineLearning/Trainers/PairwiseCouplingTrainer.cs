using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using StudentClassification.MachineLearning.Common;

namespace StudentClassification.MachineLearning.Trainers
{
    public class PairwiseCouplingTrainer: TrainerBase<PairwiseCouplingModelParameters>
    {
        public PairwiseCouplingTrainer() : base()
        {
            Name = "PairwiseCoupling";

            _model = MlContext.MulticlassClassification.Trainers.PairwiseCoupling(binaryEstimator: MlContext.BinaryClassification.Trainers.SgdCalibrated());
        }
    }
}
