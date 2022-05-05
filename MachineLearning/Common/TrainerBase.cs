using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using StudentClassification.MachineLearning.DataModels;

namespace StudentClassification.MachineLearning.Common
{
    /// <summary>
    /// Base class for Trainers.
    /// This class exposes methods for training, evaluating and saving ML Models.
    /// Classes that inherit this class need to assing concrete model and name; and to implement data pre-processing.
    /// </summary>
    public abstract class TrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; } // to add the name of the algorithm

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl"); // to store trained model

        protected readonly MLContext MlContext; // to use ML.NET functionalities

        // Loaded data split into train and test set
        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<MulticlassPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel; // resulting model

        protected TrainerBase()
        {
            MlContext = new MLContext(111);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="trainingFileName"></param>
        public void Fit(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline
                                    .Append(_model)
                                    .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>RegressionMetrics object which contain information about model performance.</returns>
        public MulticlassClassificationMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return MlContext.MulticlassClassification.Evaluate(testSetTransform);
        }

        /// <summary>
        /// Save Model into defined file path.
        /// </summary>
        public void Save()
        {
            MlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        /// <summary>
        /// Data pre-processing.
        /// </summary>
        /// <returns>Data Processing Pipeline.</returns>
        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(StudentsData.Label), outputColumnName: "Label")
               .Append(MlContext.Transforms.Concatenate("Features",nameof(StudentsData.CorrectAnswer), nameof(StudentsData.AnswerTime)))
               .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
               .AppendCacheCheckpoint(MlContext);

            return dataProcessPipeline;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<StudentsData>(trainingFileName, hasHeader: true, separatorChar: ',');
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}
