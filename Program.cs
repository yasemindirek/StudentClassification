using System;
using StudentClassification.MachineLearning.Common;
using StudentClassification.MachineLearning.DataModels;
using StudentClassification.MachineLearning.Predictors;
using StudentClassification.MachineLearning.Trainers;

namespace StudentClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            // New sample to be tested
            var newSample = new StudentsData
            {
                CorrectAnswers = 100,
                AnswerTime = 30
         
            };


            var trainers = new List<ITrainerBase>
            {
                new LbfgsMaximumEntropyTrainer(),
                new NaiveBayesTrainer(),
                new OneVersusAllTrainer(),
                new SdcaMaximumEntropyTrainer(),
                new SdcaNonCalibratedTrainer()
            };

            trainers.ForEach(t => TrainEvaluatePredict(t, newSample));
        }

        static void TrainEvaluatePredict(ITrainerBase trainer, StudentsData newSample)
        {
            Console.WriteLine("*******************************");
            Console.WriteLine($"{ trainer.Name }");
            Console.WriteLine("*******************************");

            trainer.Fit("C:\\Users\\yasem\\source\\repos\\StudentClassification\\StudentClassification\\Data\\data.csv"); // Dataset path

            var modelMetrics = trainer.Evaluate();

            Console.WriteLine($"Macro Accuracy: {modelMetrics.MacroAccuracy:#.##}{Environment.NewLine}" +
                              $"Micro Accuracy: {modelMetrics.MicroAccuracy:#.##}{Environment.NewLine}" +
                              $"Log Loss: {modelMetrics.LogLoss:#.##}{Environment.NewLine}" +
                              $"Log Loss Reduction: {modelMetrics.LogLossReduction:#.##}{Environment.NewLine}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(newSample);
            Console.WriteLine("------------------------------");
            Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
            Console.WriteLine("------------------------------");
        }
    }
}
