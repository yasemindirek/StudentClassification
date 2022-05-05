using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using StudentClassification.MachineLearning.Common;

namespace StudentClassification.MachineLearning.Trainers
{
    public class SdcaNonCalibratedTrainer : TrainerBase<LinearMulticlassModelParameters>
    {
        public SdcaNonCalibratedTrainer() : base()
        {
            Name = "Sdca NonCalibrated";
            _model = MlContext.MulticlassClassification.Trainers
              .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
