using System;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using StudentClassification.MachineLearning.Common;

namespace StudentClassification.MachineLearning.Trainers
{
    public class NaiveBayesTrainer : TrainerBase<NaiveBayesMulticlassModelParameters>
    {
        public NaiveBayesTrainer() : base()
        {
            Name = "Naive Bayes";
            _model = MlContext.MulticlassClassification.Trainers
                      .NaiveBayes(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
